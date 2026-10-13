#!/usr/bin/env python3
"""Recognize GDS label templates and fit a rigid microscope-image alignment."""
import argparse
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd
from scipy.spatial import cKDTree


def geometry(path):
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / 'geometry.json'
        env = dict(os.environ, ALIGN_GDS=str(path.resolve()),
                   ALIGN_GEOMETRY=str(target), QT_QPA_PLATFORM='offscreen')
        subprocess.run(['klayout', '-b', '-r', str(Path(__file__).parent / 'tools/export_geometry.py')],
                       env=env, check=True, capture_output=True, text=True)
        return [np.array(q) for q in json.loads(target.read_text())]


def label_library(polygons, scale):
    chars = [q for q in polygons if 8 < np.ptp(q, axis=0)[1] < 9]
    if not chars:
        raise ValueError('No expected 8.4 micron GDS lettering found')
    ys = sorted(set(q[:, 1].min() for q in chars), reverse=True)
    # Separate columns by the large gaps between character origins.
    origins = sorted(set(q[:, 0].min() for q in chars))
    columns = []
    for x in origins:
        if not columns or x - columns[-1][-1] > 20:
            columns.append([])
        columns[-1].append(x)
    groups = {}
    for q in chars:
        col = next(i for i, c in enumerate(columns) if q[:, 0].min() in c)
        groups.setdefault((col, ys.index(q[:, 1].min())), []).append(q)
    squares = [q for q in polygons if np.allclose(np.ptp(q, axis=0), [2, 2])]
    centers = np.array([q.mean(axis=0) for q in squares])
    library = []
    for (col, row), qs in sorted(groups.items()):
        a = np.concatenate(qs)
        center = (a.min(0) + a.max(0)) / 2
        # Associated marker is above and to the left of the lettering.
        nearby = [q for q, c in zip(squares, centers)
                  if center[0]-25 < c[0] < center[0]-7 and 0 < c[1]-center[1] < 20]
        if len(nearby) != 2:
            raise ValueError(f'Expected a diagonal square pair beside label {row},{col}')
        anchor = np.mean([(q.min(0) + q.max(0)) / 2 for q in nearby], axis=0)
        image = Image.new('L', (40, 40))
        draw = ImageDraw.Draw(image)
        for q in qs:
            draw.polygon([((x-center[0])/scale+20, (center[1]-y)/scale+20) for x,y in q], fill=255)
        template = nd.gaussian_filter(np.array(image, float), .7)
        template -= template.mean()
        template /= np.linalg.norm(template)
        library.append(dict(name=chr(65+row)+str(col+1), center=center,
                            anchor=anchor, squares=nearby, template=template))
    return library


def detect(image, library):
    gray = np.array(image.convert('L'), float)
    contrast = nd.gaussian_filter(gray, 5)-gray
    components, _ = nd.label(nd.binary_dilation(contrast > 15, structure=np.ones((3,5))))
    templates = np.array([v['template'] for v in library])
    detections = []
    for sl in nd.find_objects(components):
        h, w = sl[0].stop-sl[0].start, sl[1].stop-sl[1].start
        if not (10 <= h <= 25 and 13 <= w <= 32):
            continue
        x,y = (sl[1].start+sl[1].stop)//2, (sl[0].start+sl[0].stop)//2
        if min(x,y) < 24 or x > gray.shape[1]-24 or y > gray.shape[0]-24:
            continue
        best = (-1, None, None)
        for dx,dy in itertools.product(range(-2,3), repeat=2):
            patch = contrast[y+dy-20:y+dy+20, x+dx-20:x+dx+20].copy()
            patch -= patch.mean()
            patch /= max(np.linalg.norm(patch), 1)
            scores = np.einsum('ijk,jk->i', templates, patch)
            k = int(scores.argmax())
            if scores[k] > best[0]:
                best = (float(scores[k]), k, [x+dx,y+dy])
        if best[0] > .65:
            detections.append(dict(score=best[0], index=best[1], pixel=best[2]))
    return detections, contrast


def rigid(p, q, scale):
    a,b = p.mean(0),q.mean(0)
    u,_,vt = np.linalg.svd((p-a).T @ (q-b))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    matrix = scale*r
    return matrix, b-matrix@a


def fit_labels(detections, library, size, scale):
    if len(detections) < 3:
        raise ValueError('Fewer than three recognizable labels; refusing an ambiguous alignment')
    p = np.array([d['pixel'] for d in detections], float)
    p = (p-np.array(size)/2)*[1,-1]
    q = np.array([library[d['index']]['center'] for d in detections])
    best = None
    for i,j in itertools.combinations(range(len(p)),2):
        if np.linalg.norm(p[i]-p[j]) < 100:
            continue
        matrix,t = rigid(p[[i,j]],q[[i,j]],scale)
        errors = np.linalg.norm(p@matrix.T+t-q,axis=1)
        good = errors < 8
        rank = (int(good.sum()), -float(np.median(errors[good])) if good.any() else -1e9)
        if best is None or rank > best[0]:
            best = rank,good
    if best is None or best[0][0] < 3:
        raise ValueError('No consistent set of three labels found')
    good = best[1]
    for _ in range(3):
        matrix,t = rigid(p[good],q[good],scale)
        good = np.linalg.norm(p@matrix.T+t-q,axis=1) < 8
    return matrix,t,good


def refine_markers(matrix,t,contrast,library,size,scale):
    """Locate the actual square pairs, using the recognized grid as an initial pose."""
    points,targets,names,scores = [],[],[],[]
    half = np.array(size)/2
    for entry in library:
        predicted = np.linalg.solve(matrix,entry['anchor']-t)*[1,-1]+half
        x,y = np.rint(predicted).astype(int)
        if x<18 or y<18 or x>=size[0]-18 or y>=size[1]-18:
            continue
        im = Image.new('L',(24,24)); draw=ImageDraw.Draw(im)
        for q in entry['squares']:
            local = np.linalg.solve(matrix,(q-entry['anchor']).T).T*[1,-1]+12
            draw.polygon([tuple(v) for v in local],fill=255)
        template=nd.gaussian_filter(np.array(im,float),.7)
        template-=template.mean();template/=np.linalg.norm(template)
        best=(-1,None)
        for dx,dy in itertools.product(range(-6,7),repeat=2):
            patch=contrast[y+dy-12:y+dy+12,x+dx-12:x+dx+12].copy()
            patch-=patch.mean();patch/=max(np.linalg.norm(patch),1)
            score=float(np.sum(patch*template))
            if score>best[0]:best=(score,[x+dx,y+dy])
        if best[0]>.5:
            points.append(best[1]);targets.append(entry['anchor']);names.append(entry['name']);scores.append(best[0])
    if len(points)<3:
        raise ValueError('Fewer than three square pairs verified')
    p=(np.array(points)-half)*[1,-1];q=np.array(targets)
    good=np.ones(len(p),bool)
    for _ in range(3):
        matrix,t=rigid(p[good],q[good],scale)
        errors=np.linalg.norm(p@matrix.T+t-q,axis=1)
        good=errors<4
    if good.sum()<3:raise ValueError('Marker verification failed')
    return matrix,t,[dict(label=names[i],pixel=[int(v) for v in points[i]],gds_um=targets[i].tolist(),
                         residual_um=float(errors[i]),score=scores[i]) for i in np.where(good)[0]]


def preview(image, polygons, matrix, t, path):
    out=image.convert('RGB');draw=ImageDraw.Draw(out);half=np.array(image.size)/2
    for q in polygons:
        pixels=np.linalg.solve(matrix,(q-t).T).T*[1,-1]+half
        if pixels[:,0].max()<0 or pixels[:,1].max()<0 or pixels[:,0].min()>image.width or pixels[:,1].min()>image.height:
            continue
        vertices=[tuple(v) for v in pixels]
        draw.line(vertices+[vertices[0]],fill=(0,255,255),width=1)
    out.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gds',type=Path,default=Path('Marker_4x4_new.gds'))
    parser.add_argument('--images',type=Path,default=Path('pictures'))
    parser.add_argument('--session',type=Path,default=Path('aligned_images.lys'),help='Session template; its image poses are ignored')
    parser.add_argument('--scale',type=float,help='Fixed calibrated microns/pixel; default: mean area-equivalent scale in session')
    parser.add_argument('--output',type=Path,default=Path('output'))
    args=parser.parse_args()
    tree=ET.parse(args.session)
    reference={}
    for ann in tree.findall('.//annotation'):
        if ann.findtext('class')!='img::Object':continue
        value=ann.findtext('value')
        rows=re.search(r'matrix=\((.*?)\) \((.*?)\)',value)
        if not rows:continue
        a,b=[list(map(float,rows.group(i).split(','))) for i in (1,2)]
        filename=Path(re.search(r"file='([^']+)'",value).group(1)).name
        reference[filename]=np.array([a,b])
    scale=args.scale or float(np.mean([np.sqrt(abs(np.linalg.det(m[:,:2]))) for m in reference.values()]))
    if not np.isfinite(scale) or scale<=0:raise ValueError('A positive calibrated --scale is required')
    polygons=geometry(args.gds);library=label_library(polygons,scale)
    args.output.mkdir(parents=True,exist_ok=True)
    annotations=tree.find('.//view/annotations')
    if annotations is None:raise ValueError('Session has no annotation container')
    for ann in list(annotations):
        if ann.findtext('class')=='img::Object':annotations.remove(ann)
    tree.find('.//layout/file-path').text=str(args.gds.resolve())
    report=dict(scale_um_per_pixel=scale,images=[])
    files=sorted(args.images.glob('*.png'))
    if not files:raise ValueError(f'No PNG images in {args.images}')
    for z,path in enumerate(files,1):
        image=Image.open(path)
        detections,contrast=detect(image,library)
        matrix,t,inliers=fit_labels(detections,library,image.size,scale)
        matrix,t,marks=refine_markers(matrix,t,contrast,library,image.size,scale)
        result=dict(file=path.name,rotation_deg=float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0]))),
                    translation_um=t.tolist(),matrix=np.column_stack([matrix,t]).tolist(),
                    recognized_label_inliers=int(inliers.sum()),markers=marks,
                    marker_rms_um=float(np.sqrt(np.mean([m['residual_um']**2 for m in marks]))))
        if path.name in reference:
            corners=np.array(list(itertools.product([-image.width/2,image.width/2],[-image.height/2,image.height/2])))
            ref=reference[path.name]
            differences=corners@matrix.T+t-(corners@ref[:,:2].T+ref[:,2])
            result['example_corner_rms_um']=float(np.sqrt(np.mean(np.sum(differences**2,axis=1))))
            result['example_center_difference_um']=float(np.linalg.norm(t-ref[:,2]))
        report['images'].append(result)
        ann=ET.SubElement(annotations,'annotation');ET.SubElement(ann,'class').text='img::Object'
        ET.SubElement(ann,'value').text=(f'color:matrix=({matrix[0,0]:.12g},{matrix[0,1]:.12g},{t[0]:.12g}) '
            f'({matrix[1,0]:.12g},{matrix[1,1]:.12g},{t[1]:.12g}) (0,0,1);min_value=0;max_value=255;'
            f'is_visible=true;z_position={z};brightness=0;contrast=0;gamma=1;red_gain=1;green_gain=1;blue_gain=1;'
            f"landmarks=[];color_mapping=[0,'#000000';1,'#ffffff';];file='{path.resolve()}'")
        preview(image,polygons,matrix,t,args.output/(path.stem+'_overlay.png'))
        print(f"{path.name}: {len(marks)} markers, RMS {result['marker_rms_um']:.3f} um, rotation {result['rotation_deg']:.3f} deg",flush=True)
    tree.write(args.output/'aligned_auto.lys',encoding='utf-8',xml_declaration=True)
    (args.output/'alignment_report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()

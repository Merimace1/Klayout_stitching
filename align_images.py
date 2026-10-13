#!/usr/bin/env python3
"""Recognize GDS label templates and fit a rigid microscope-image alignment."""

import argparse
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import asdict

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd
from scipy.spatial import cKDTree

from alignment_config import AlignmentConfig
from gds_io import load_geometry
from session_io import build_session, reference_matrices, image_annotation


def infer_dimensions(polygons, config):
    extents = np.array([np.ptp(q, axis=0) for q in polygons])
    square_extents = extents[
        np.isclose(extents[:, 0], extents[:, 1]) & (extents[:, 0] > 0)
    ]
    if not len(square_extents):
        raise ValueError("GDS contains no square marker candidates")
    sizes, counts = np.unique(np.round(square_extents[:, 0], 6), return_counts=True)
    square_size = config.square_size_um or float(sizes[counts.argmax()])
    heights = extents[
        (extents[:, 1] > 2 * square_size) & (extents[:, 0] < extents[:, 1]), 1
    ]
    if not len(heights) and config.letter_height_um is None:
        raise ValueError(
            "Cannot infer lettering height; set letter_height_um in --config"
        )
    values, counts = np.unique(np.round(heights, 6), return_counts=True)
    letter_height = config.letter_height_um or float(values[counts.argmax()])
    return letter_height, square_size


def label_library(polygons, scale, config=AlignmentConfig()):
    letter_height, square_size = infer_dimensions(polygons, config)
    chars = [
        q
        for q in polygons
        if np.isclose(np.ptp(q, axis=0)[1], letter_height, rtol=0.02)
    ]
    if not chars:
        raise ValueError("No polygons match the inferred lettering height")
    ys = sorted(set(q[:, 1].min() for q in chars), reverse=True)
    # Separate columns by the large gaps between character origins.
    origins = sorted(set(q[:, 0].min() for q in chars))
    columns = []
    for x in origins:
        if not columns or x - columns[-1][-1] > 2.4 * letter_height:
            columns.append([])
        columns[-1].append(x)
    groups = {}
    for q in chars:
        col = next(i for i, c in enumerate(columns) if q[:, 0].min() in c)
        groups.setdefault((col, ys.index(q[:, 1].min())), []).append(q)
    squares = [
        q
        for q in polygons
        if np.allclose(np.ptp(q, axis=0), [square_size, square_size], rtol=0.02)
    ]
    centers = np.array([q.mean(axis=0) for q in squares])
    library = []
    square_tree = cKDTree(centers)
    for (col, row), qs in sorted(groups.items()):
        a = np.concatenate(qs)
        center = (a.min(0) + a.max(0)) / 2
        # Find a compact diagonal pair near the label, regardless of which
        # side it is on. Remote grid squares are not considered.
        indices = square_tree.query_ball_point(center, 4 * letter_height)
        pairs = []
        for i, j in itertools.combinations(indices, 2):
            separation = np.abs(centers[i] - centers[j])
            if np.all(
                (separation > 0.8 * square_size) & (separation < 2.5 * square_size)
            ):
                distance = np.linalg.norm((centers[i] + centers[j]) / 2 - center)
                pairs.append((distance, i, j))
        if not pairs:
            raise ValueError(f"No diagonal square pair beside label {row},{col}")
        _, i, j = min(pairs)
        nearby = [squares[i], squares[j]]
        anchor = np.mean([(q.min(0) + q.max(0)) / 2 for q in nearby], axis=0)
        image = Image.new("L", (40, 40))
        draw = ImageDraw.Draw(image)
        for q in qs:
            draw.polygon(
                [
                    ((x - center[0]) / scale + 20, (center[1] - y) / scale + 20)
                    for x, y in q
                ],
                fill=255,
            )
        template = nd.gaussian_filter(np.array(image, float), 0.7)
        template -= template.mean()
        template /= np.linalg.norm(template)
        library.append(
            dict(
                name=chr(65 + row) + str(col + 1),
                center=center,
                anchor=anchor,
                squares=nearby,
                template=template,
            )
        )
    return library


def detect(image, library, config=AlignmentConfig()):
    gray = np.array(image.convert("L"), float)
    contrast = (nd.gaussian_filter(gray, config.background_sigma_px) - gray) * (
        1 if config.polarity == "dark" else -1
    )
    components, _ = nd.label(
        nd.binary_dilation(
            contrast > config.contrast_threshold, structure=np.ones((3, 5))
        )
    )
    templates = np.array([v["template"] for v in library])
    detections = []
    for sl in nd.find_objects(components):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (10 <= h <= 25 and 13 <= w <= 32):
            continue
        x, y = (sl[1].start + sl[1].stop) // 2, (sl[0].start + sl[0].stop) // 2
        if min(x, y) < 24 or x > gray.shape[1] - 24 or y > gray.shape[0] - 24:
            continue
        best = (-1, None, None)
        for dx, dy in itertools.product(range(-2, 3), repeat=2):
            patch = contrast[
                y + dy - 20 : y + dy + 20, x + dx - 20 : x + dx + 20
            ].copy()
            patch -= patch.mean()
            patch /= max(np.linalg.norm(patch), 1)
            scores = np.einsum("ijk,jk->i", templates, patch)
            k = int(scores.argmax())
            if scores[k] > best[0]:
                best = (float(scores[k]), k, [x + dx, y + dy])
        if best[0] > config.label_score:
            detections.append(dict(score=best[0], index=best[1], pixel=best[2]))
    return detections, contrast


def rigid(p, q, scale):
    a, b = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - a).T @ (q - b))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    matrix = scale * r
    return matrix, b - matrix @ a


def fit_labels(detections, library, size, scale, config=AlignmentConfig()):
    if len(detections) < 3:
        raise ValueError(
            "Fewer than three recognizable labels; refusing an ambiguous alignment"
        )
    p = np.array([d["pixel"] for d in detections], float)
    p = (p - np.array(size) / 2) * [1, -1]
    q = np.array([library[d["index"]]["center"] for d in detections])
    best = None
    for i, j in itertools.combinations(range(len(p)), 2):
        if np.linalg.norm(p[i] - p[j]) < 100:
            continue
        matrix, t = rigid(p[[i, j]], q[[i, j]], scale)
        errors = np.linalg.norm(p @ matrix.T + t - q, axis=1)
        good = errors < config.label_tolerance_um
        rank = (
            int(good.sum()),
            -float(np.median(errors[good])) if good.any() else -1e9,
        )
        if best is None or rank > best[0]:
            best = rank, good
    if best is None or best[0][0] < 3:
        raise ValueError("No consistent set of three labels found")
    good = best[1]
    for _ in range(3):
        if good.sum() < 3:
            raise ValueError(
                "Too few consistent correspondences remain after refinement"
            )
        matrix, t = rigid(p[good], q[good], scale)
        good = np.linalg.norm(p @ matrix.T + t - q, axis=1) < config.label_tolerance_um
    return matrix, t, good


def refine_markers(matrix, t, contrast, library, size, scale, config=AlignmentConfig()):
    """Locate the actual square pairs, using the recognized grid as an initial pose."""
    points, targets, names, scores = [], [], [], []
    half = np.array(size) / 2
    search = config.marker_search_px
    margin = 12 + search
    for entry in library:
        predicted = np.linalg.solve(matrix, entry["anchor"] - t) * [1, -1] + half
        x, y = np.rint(predicted).astype(int)
        if x < margin or y < margin or x >= size[0] - margin or y >= size[1] - margin:
            continue
        im = Image.new("L", (24, 24))
        draw = ImageDraw.Draw(im)
        for q in entry["squares"]:
            local = np.linalg.solve(matrix, (q - entry["anchor"]).T).T * [1, -1] + 12
            draw.polygon([tuple(v) for v in local], fill=255)
        template = nd.gaussian_filter(np.array(im, float), 0.7)
        template -= template.mean()
        template /= np.linalg.norm(template)
        best = (-1, None)
        for dx, dy in itertools.product(range(-search, search + 1), repeat=2):
            patch = contrast[
                y + dy - 12 : y + dy + 12, x + dx - 12 : x + dx + 12
            ].copy()
            patch -= patch.mean()
            patch /= max(np.linalg.norm(patch), 1)
            score = float(np.sum(patch * template))
            if score > best[0]:
                best = (score, [x + dx, y + dy])
        if best[0] > config.marker_score:
            points.append(best[1])
            targets.append(entry["anchor"])
            names.append(entry["name"])
            scores.append(best[0])
    if len(points) < 3:
        raise ValueError("Fewer than three square pairs verified")
    p = (np.array(points) - half) * [1, -1]
    q = np.array(targets)
    good = np.ones(len(p), bool)
    for _ in range(3):
        if good.sum() < 3:
            raise ValueError(
                "Too few consistent correspondences remain after refinement"
            )
        matrix, t = rigid(p[good], q[good], scale)
        errors = np.linalg.norm(p @ matrix.T + t - q, axis=1)
        good = errors < config.marker_tolerance_um
    if good.sum() < 3:
        raise ValueError("Marker verification failed")
    return (
        matrix,
        t,
        [
            dict(
                label=names[i],
                pixel=[int(v) for v in points[i]],
                gds_um=targets[i].tolist(),
                residual_um=float(errors[i]),
                score=scores[i],
            )
            for i in np.where(good)[0]
        ],
    )


def preview(image, polygons, matrix, t, path):
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    half = np.array(image.size) / 2
    for q in polygons:
        pixels = np.linalg.solve(matrix, (q - t).T).T * [1, -1] + half
        if (
            pixels[:, 0].max() < 0
            or pixels[:, 1].max() < 0
            or pixels[:, 0].min() > image.width
            or pixels[:, 1].min() > image.height
        ):
            continue
        vertices = [tuple(v) for v in pixels]
        draw.line(vertices + [vertices[0]], fill=(0, 255, 255), width=1)
    out.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gds", type=Path, required=True, help="Marker GDS file")
    parser.add_argument("--images", type=Path, required=True, help="Image folder")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output folder (default: ./output)",
    )
    parser.add_argument("--scale", type=float, help="Fixed microns/pixel calibration")
    parser.add_argument(
        "--session", type=Path, help="Optional calibration/reference session"
    )
    parser.add_argument(
        "--klayout-executable",
        help="Desktop KLayout executable path; overrides Python backend",
    )
    parser.add_argument(
        "--cell", help="GDS top cell (required when the GDS has multiple top cells)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON recognition parameters; see config.example.json",
    )
    args = parser.parse_args()
    try:
        if not args.images.is_dir():
            raise ValueError(f"Image folder does not exist: {args.images}")
        if not args.gds.is_file():
            raise ValueError(f"GDS file does not exist: {args.gds}")
        config = AlignmentConfig.from_file(args.config)
        # Preserve convenient calibration for the supplied example, but allow
        # operation with no example session at all when --scale is supplied.
        calibration = args.session
        if calibration is None and args.scale is None:
            calibration = Path(__file__).resolve().parent / "aligned_images.lys"
        reference = reference_matrices(ET.parse(calibration) if calibration else None)
        scales = [np.sqrt(abs(np.linalg.det(m[:, :2]))) for m in reference.values()]
        scale = (
            args.scale
            if args.scale is not None
            else float(np.mean(scales))
            if scales
            else None
        )
        if scale is None or not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                "Provide a positive --scale, or a --session with calibrated image matrices"
            )
        files = sorted(
            p
            for p in args.images.iterdir()
            if p.is_file()
            and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        )
        if not files:
            raise ValueError(f"No supported images in {args.images}")
        data = load_geometry(args.gds, args.klayout_executable, args.cell)
        polygons = data["polygons"]
        letter_height, square_size = infer_dimensions(polygons, config)
        # Normalize sampling, not physical geometry: matching windows retain
        # their meaning across acquisition magnifications and design sizes.
        normalized_scale = letter_height / config.target_letter_height_px
        resize_ratio = scale / normalized_scale
        if abs(resize_ratio - 1) < 0.02:
            resize_ratio = 1.0
        working_scale = scale / resize_ratio
        library = label_library(polygons, working_scale, config)
        tree = build_session(args.gds, data["cell"], polygons, data["layers"])
        annotations = tree.find(".//view/annotations")
        args.output.mkdir(parents=True, exist_ok=True)
    except (ValueError, RuntimeError, OSError, ET.ParseError) as error:
        parser.error(str(error))
    report = dict(
        scale_um_per_pixel=scale,
        geometry_backend=data["backend"],
        cell=data["cell"],
        letter_height_um=letter_height,
        square_size_um=square_size,
        config=asdict(config),
        images=[],
    )
    for z, path in enumerate(files, 1):
        try:
            image = Image.open(path)
            image.load()
        except OSError as error:
            parser.error(f"Cannot read image {path}: {error}")
        working = image
        if resize_ratio != 1.0:
            # A tiny dimension-rounding difference remains below one working pixel.
            working = image.resize(
                (
                    max(1, round(image.width * resize_ratio)),
                    max(1, round(image.height * resize_ratio)),
                ),
                getattr(Image, "Resampling", Image).LANCZOS,
            )
        try:
            detections, contrast = detect(working, library, config)
            matrix, t, inliers = fit_labels(
                detections, library, working.size, working_scale, config
            )
            matrix, t, marks = refine_markers(
                matrix, t, contrast, library, working.size, working_scale, config
            )
        except ValueError as error:
            parser.error(
                f"{path.name}: {error}. Check --scale and recognition settings in --config"
            )
        matrix *= resize_ratio
        for mark in marks:
            centered = (
                np.array(mark["pixel"]) - np.array(working.size) / 2
            ) / resize_ratio
            mark["pixel"] = (centered + np.array(image.size) / 2).tolist()
        result = dict(
            file=path.name,
            rotation_deg=float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
            translation_um=t.tolist(),
            matrix=np.column_stack([matrix, t]).tolist(),
            recognized_label_inliers=int(inliers.sum()),
            markers=marks,
            marker_rms_um=float(
                np.sqrt(np.mean([m["residual_um"] ** 2 for m in marks]))
            ),
        )
        if path.name in reference:
            corners = np.array(
                list(
                    itertools.product(
                        [-image.width / 2, image.width / 2],
                        [-image.height / 2, image.height / 2],
                    )
                )
            )
            ref = reference[path.name]
            differences = corners @ matrix.T + t - (corners @ ref[:, :2].T + ref[:, 2])
            result["example_corner_rms_um"] = float(
                np.sqrt(np.mean(np.sum(differences**2, axis=1)))
            )
            result["example_center_difference_um"] = float(
                np.linalg.norm(t - ref[:, 2])
            )
        report["images"].append(result)
        ann = ET.SubElement(annotations, "annotation")
        ET.SubElement(ann, "class").text = "img::Object"
        ET.SubElement(ann, "value").text = image_annotation(path, matrix, t, z)
        preview(image, polygons, matrix, t, args.output / (path.stem + "_overlay.png"))
        print(
            f"{path.name}: {len(marks)} markers, RMS {result['marker_rms_um']:.3f} um, rotation {result['rotation_deg']:.3f} deg",
            flush=True,
        )
    tree.write(args.output / "aligned_auto.lys", encoding="utf-8", xml_declaration=True)
    (args.output / "alignment_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

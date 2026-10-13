"""Anchor images with GDS labels, then grow an overlap registration graph."""

from dataclasses import asdict
import itertools
import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from session_io import build_session, image_annotation
from stitching import (
    extract_features,
    register_pair,
    refine_registration,
    compose_pose,
    mosaic,
)


def align_batch(
    files,
    polygons,
    candidates,
    config,
    marker_aligner,
    scale_map=None,
    estimate_scale=True,
    stitch=True,
):
    scale_map = scale_map or {}
    images = {}
    poses = {}
    failures = {}
    features = {}
    links = {}
    for path in files:
        try:
            image = Image.open(path)
            image.load()
            images[path.name] = image.convert("RGB")
        except OSError as error:
            failures[path.name] = dict(reason=f"Cannot read image: {error}")
            continue
        print(f"{path.name}: trying GDS markers", flush=True)
        attempts = []
        best = None
        scales = [scale_map[path.name]] if path.name in scale_map else candidates
        fixed = path.name in scale_map or not estimate_scale
        for scale in scales:
            try:
                result = marker_aligner(
                    images[path.name], polygons, scale, config, estimate_scale=not fixed
                )
                rank = (
                    result["recognized_label_inliers"],
                    len(result["markers"]),
                    -result["marker_rms_um"],
                )
                if best is None or rank > best[0]:
                    best = rank, result
                # A substantial set of labels settles the absolute placement.
                if result["recognized_label_inliers"] >= 6:
                    break
            except ValueError as error:
                attempts.append(dict(scale_um_per_pixel=scale, reason=str(error)))
        if best:
            result = best[1]
            result.update(
                method="markers", anchor=path.name, depth=0, marker_attempts=attempts
            )
            poses[path.name] = result
            print(
                f"{path.name}: GDS anchor, {len(result['markers'])} markers, scale {np.sqrt(np.linalg.det(result['matrix'])):.6g} um/px",
                flush=True,
            )
        else:
            failures[path.name] = dict(
                reason="No reliable GDS label alignment", marker_attempts=attempts
            )
    if not stitch or not poses:
        return images, poses, failures

    def get_features(name):
        if name not in features:
            features[name] = extract_features(
                images[name],
                max_dimension=config.stitching_max_dimension,
                levels=config.stitching_feature_levels,
                per_level=config.stitching_features_per_level,
            )
        return features[name]

    while True:
        additions = []
        parents = list(poses)
        for child in sorted(set(images) - set(poses)):
            best = None
            for parent in parents:
                key = (child, parent)
                if key not in links:
                    fixed_ratio = None
                    if child in scale_map:
                        fixed_ratio = scale_map[child] / np.sqrt(
                            np.linalg.det(poses[parent]["matrix"])
                        )
                    pair = register_pair(
                        get_features(child),
                        get_features(parent),
                        images[child],
                        images[parent],
                        fixed_scale=fixed_ratio,
                        min_inliers=config.stitching_min_inliers,
                        ratio=config.stitching_descriptor_ratio,
                        threshold=config.stitching_threshold_px,
                        minimum_score=config.stitching_min_gradient_score,
                    )
                    if pair is None and fixed_ratio is None:
                        # Sparse close-ups need a wider initialization tolerance;
                        # require a stronger dense image check after refinement.
                        pair = register_pair(
                            get_features(child),
                            get_features(parent),
                            images[child],
                            images[parent],
                            min_inliers=config.stitching_min_inliers,
                            ratio=config.stitching_descriptor_ratio,
                            threshold=config.stitching_sparse_threshold_px,
                            minimum_score=min(0.1, config.stitching_min_gradient_score),
                        )
                    if (
                        pair is not None
                        and fixed_ratio is None
                        and (pair["inliers"] < 8 or pair["gradient_score"] < 0.4)
                    ):
                        pair = refine_registration(images[child], images[parent], pair)
                        if pair["gradient_score"] < config.stitching_min_gradient_score:
                            pair = None
                    links[key] = pair
                pair = links[key]
                if pair is None:
                    continue
                rank = (
                    pair["gradient_score"]
                    * np.sqrt(pair["inliers"])
                    / (1 + poses[parent]["depth"]),
                    -pair["rms_px"],
                )
                if best is None or rank > best[0]:
                    best = rank, parent, pair
            if best:
                _, parent, pair = best
                matrix, translation = compose_pose(
                    poses[parent]["matrix"],
                    poses[parent]["translation"],
                    images[parent].size,
                    images[child].size,
                    pair,
                )
                clean = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in pair.items()
                    if key
                    not in ("matrix", "translation", "source_points", "target_points")
                }
                clean["limited_features"] = pair["inliers"] < 8
                result = dict(
                    matrix=matrix,
                    translation=translation,
                    method="stitching",
                    parent=parent,
                    anchor=poses[parent]["anchor"],
                    depth=poses[parent]["depth"] + 1,
                    registration=clean,
                    relative_matrix=np.column_stack(
                        [pair["matrix"], pair["translation"]]
                    ).tolist(),
                    marker_attempts=failures[child].get("marker_attempts", []),
                )
                additions.append((child, result))
        if not additions:
            break
        for name, result in additions:
            poses[name] = result
            failures.pop(name, None)
            print(
                f"{name}: stitched to {result['parent']}, scale {np.sqrt(np.linalg.det(result['matrix'])):.6g} um/px, {result['registration']['inliers']} feature matches",
                flush=True,
            )
    for name in set(images) - set(poses):
        failures[name]["reason"] = "No verified overlap path to a GDS-anchored image"
    return images, poses, failures


def write_outputs(
    files,
    images,
    poses,
    failures,
    data,
    gds,
    output,
    config,
    reference,
    preview,
    candidates,
):
    tree = build_session(gds, data["cell"], data["polygons"], data["layers"])
    annotations = tree.find(".//view/annotations")
    # Larger physical fields are lower in both the KLayout stack and preview.
    order = sorted(
        poses,
        key=lambda name: (
            -images[name].width
            * images[name].height
            * np.linalg.det(poses[name]["matrix"]),
            name,
        ),
    )
    report = dict(
        geometry_backend=data["backend"],
        cell=data["cell"],
        config=asdict(config),
        scale_candidates_um_per_pixel=candidates,
        images=[],
        unresolved=failures,
        draw_order_large_to_small=order,
    )
    paths = {p.name: p for p in files}
    for z, name in enumerate(order, 1):
        pose = poses[name]
        matrix = pose["matrix"]
        t = pose["translation"]
        image = images[name]
        record = {
            key: value
            for key, value in pose.items()
            if key not in ("matrix", "translation")
        }
        record.update(
            file=name,
            scale_um_per_pixel=float(np.sqrt(np.linalg.det(matrix))),
            rotation_deg=float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
            translation_um=t.tolist(),
            matrix=np.column_stack([matrix, t]).tolist(),
            physical_area_um2=float(image.width * image.height * np.linalg.det(matrix)),
            z_position=z,
        )
        if name in reference:
            corners = np.array(
                list(
                    itertools.product(
                        [-image.width / 2, image.width / 2],
                        [-image.height / 2, image.height / 2],
                    )
                )
            )
            ref = reference[name]
            expected = corners @ ref[:2, :2].T + ref[:2, 2]
            if ref.shape == (3, 3):
                expected /= (corners @ ref[2, :2] + ref[2, 2])[:, None]
            differences = corners @ matrix.T + t - expected
            record["example_corner_rms_um"] = float(
                np.sqrt(np.mean(np.sum(differences**2, axis=1)))
            )
            record["example_center_difference_um"] = float(
                np.linalg.norm(
                    t - ref[:2, 2] / (ref[2, 2] if ref.shape == (3, 3) else 1.0)
                )
            )
        report["images"].append(record)
        annotation = ET.SubElement(annotations, "annotation")
        ET.SubElement(annotation, "class").text = "img::Object"
        ET.SubElement(annotation, "value").text = image_annotation(
            paths[name], matrix, t, z
        )
        preview(
            image,
            data["polygons"],
            matrix,
            t,
            output / (paths[name].stem + "_overlay.png"),
        )
    if order:
        report["mosaic"] = mosaic(
            [images[name] for name in order],
            [(poses[name]["matrix"], poses[name]["translation"]) for name in order],
            output / "mosaic.png",
        )
    tree.write(output / "aligned_auto.lys", encoding="utf-8", xml_declaration=True)
    (output / "alignment_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report

"""Multiscale gradient-feature registration without trained models.

Descriptors follow the SIFT principle (oriented spatial gradient histograms),
but this is a small SciPy implementation, not the original SIFT detector.
"""

from dataclasses import dataclass
import itertools

import numpy as np
from PIL import Image
from scipy import ndimage as nd
from scipy.spatial import cKDTree, ConvexHull


@dataclass
class Features:
    points: np.ndarray
    descriptors: np.ndarray
    size: tuple


def extract_features(image, max_dimension=1600, levels=9, per_level=450):
    gray = np.asarray(image.convert("L"), dtype=float) / 255
    base = min(1.0, max_dimension / max(image.size))
    points, descriptors = [], []
    grid_y, grid_x = np.mgrid[-15.5:16.5, -15.5:16.5]
    weights = np.exp(-(grid_x**2 + grid_y**2) / (2 * 12**2))
    for level in range(levels):
        ratio = base * 2 ** (-level / 2)
        if min(gray.shape) * ratio < 72:
            break
        scaled = nd.zoom(gray, ratio, order=1)
        gx = nd.gaussian_filter(scaled, 1, order=(0, 1))
        gy = nd.gaussian_filter(scaled, 1, order=(1, 0))
        xx, yy, xy = [nd.gaussian_filter(v, 1.5) for v in (gx * gx, gy * gy, gx * gy)]
        response = xx * yy - xy * xy - 0.045 * (xx + yy) ** 2
        response[:24] = 0
        response[-24:] = 0
        response[:, :24] = 0
        response[:, -24:] = 0
        maxima = (response == nd.maximum_filter(response, 9)) & (
            response > response.max() * 0.0005
        )
        y, x = np.where(maxima)
        order = np.argsort(response[y, x])[::-1][:per_level]
        for k in order:
            cy, cx = int(y[k]), int(x[k])
            sx = gx[cy - 10 : cy + 11, cx - 10 : cx + 11]
            sy = gy[cy - 10 : cy + 11, cx - 10 : cx + 11]
            angles = np.arctan2(sy, sx)
            magnitude = np.hypot(sx, sy)
            orient = np.bincount(
                ((angles + np.pi) * 36 / (2 * np.pi)).astype(int).ravel() % 36,
                weights=magnitude.ravel(),
                minlength=36,
            )
            orient = nd.gaussian_filter1d(orient, 1, mode="wrap")
            theta = (orient.argmax() + 0.5) * 2 * np.pi / 36 - np.pi
            c, s = np.cos(theta), np.sin(theta)
            px = cx + c * grid_x - s * grid_y
            py = cy + s * grid_x + c * grid_y
            dx = nd.map_coordinates(gx, [py, px], order=1)
            dy = nd.map_coordinates(gy, [py, px], order=1)
            magnitude = np.hypot(dx, dy) * weights
            angle = (np.arctan2(dy, dx) - theta) % (2 * np.pi) * 8 / (2 * np.pi)
            lower = np.floor(angle).astype(int) % 8
            fraction = angle - np.floor(angle)
            hist = np.zeros((4, 4, 8))
            for iy, ix in itertools.product(range(4), repeat=2):
                sl = (slice(iy * 8, iy * 8 + 8), slice(ix * 8, ix * 8 + 8))
                bins = lower[sl].ravel()
                mag = magnitude[sl].ravel()
                f = fraction[sl].ravel()
                hist[iy, ix] = np.bincount(
                    bins, weights=mag * (1 - f), minlength=8
                ) + np.bincount((bins + 1) % 8, weights=mag * f, minlength=8)
            descriptor = hist.ravel()
            descriptor /= max(np.linalg.norm(descriptor), 1e-12)
            descriptor = np.minimum(descriptor, 0.2)
            descriptor /= max(np.linalg.norm(descriptor), 1e-12)
            # Root descriptor improves robustness to illumination changes.
            descriptor = np.sqrt(descriptor / max(descriptor.sum(), 1e-12))
            points.append([(cx + 0.5) / ratio - 0.5, (cy + 0.5) / ratio - 0.5])
            descriptors.append(descriptor)
    return Features(
        np.array(points).reshape(-1, 2),
        np.array(descriptors).reshape(-1, 128),
        image.size,
    )


def similarity(p, q, fixed_scale=None):
    """Fit q = A p + b with a positive uniform scale and proper rotation."""
    pc, qc = p.mean(0), q.mean(0)
    u, values, vt = np.linalg.svd((p - pc).T @ (q - qc))
    correction = np.ones(2)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1] = -1
    rotation = vt.T @ np.diag(correction) @ u.T
    denominator = np.sum((p - pc) ** 2)
    if denominator < 1e-12:
        raise ValueError("Degenerate feature positions")
    scale = (
        fixed_scale
        if fixed_scale is not None
        else float(np.dot(values, correction) / denominator)
    )
    if scale <= 0:
        raise ValueError("Degenerate similarity scale")
    matrix = scale * rotation
    return matrix, qc - matrix @ pc


def spread(points, size):
    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume / (size[0] * size[1]))
    except Exception:
        return 0.0


def feature_matches(source, target, ratio=0.8):
    if min(len(source.points), len(target.points)) < 2:
        return np.empty((0, 2), int)
    distances, indices = cKDTree(target.descriptors).query(source.descriptors, k=2)
    reverse = cKDTree(source.descriptors).query(target.descriptors, k=1)[1]
    good = (distances[:, 0] < ratio * distances[:, 1]) & (
        reverse[indices[:, 0]] == np.arange(len(source.points))
    )
    pairs = np.column_stack([np.where(good)[0], indices[good, 0]])
    # Pyramid observations of the same corner are not independent evidence.
    unique = []
    for i, j in pairs:
        if all(
            np.linalg.norm(source.points[i] - source.points[a]) > 4
            and np.linalg.norm(target.points[j] - target.points[b]) > 4
            for a, b in unique
        ):
            unique.append((i, j))
    return np.array(unique, dtype=int).reshape(-1, 2)


def photometric_score(source_image, target_image, matrix, translation):
    """Check gradient agreement in the predicted common region."""
    source = np.array(source_image.convert("L"), float)
    target = np.array(target_image.convert("L"), float)
    ys = np.linspace(2, source.shape[0] - 3, 100)
    xs = np.linspace(2, source.shape[1] - 3, 100)
    y, x = np.meshgrid(ys, xs, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel()])
    mapped = points @ matrix.T + translation
    valid = (
        (mapped[:, 0] > 2)
        & (mapped[:, 0] < target.shape[1] - 3)
        & (mapped[:, 1] > 2)
        & (mapped[:, 1] < target.shape[0] - 3)
    )
    if valid.sum() < 100:
        return -1.0, float(valid.mean())
    scale = np.sqrt(np.linalg.det(matrix))
    # Smooth to corresponding physical resolution before comparing gradients.
    sigma_target = max(3.0, 3.0 * scale)
    sigma_source = sigma_target / scale
    sg = [
        (
            nd.gaussian_filter(source, sigma_source, order=axis)
            - nd.gaussian_filter(source, sigma_source * 6, order=axis)
        )
        for axis in [(0, 1), (1, 0)]
    ]
    tg = [
        (
            nd.gaussian_filter(target, sigma_target, order=axis)
            - nd.gaussian_filter(target, sigma_target * 6, order=axis)
        )
        for axis in [(0, 1), (1, 0)]
    ]
    s = np.column_stack(
        [
            nd.map_coordinates(v, [points[valid, 1], points[valid, 0]], order=1)
            for v in sg
        ]
    )
    t = np.column_stack(
        [
            nd.map_coordinates(v, [mapped[valid, 1], mapped[valid, 0]], order=1)
            for v in tg
        ]
    )
    rotation = matrix / scale
    s = s @ rotation.T
    # Clip extreme edge amplitudes to keep large debris from dominating.
    for v in (s, t):
        norms = np.linalg.norm(v, axis=1)
        v *= np.minimum(1, np.percentile(norms, 95) / np.maximum(norms, 1e-12))[:, None]
    score = float(np.sum(s * t) / max(np.linalg.norm(s) * np.linalg.norm(t), 1e-12))
    return score, float(valid.mean())


def register_pair(
    source,
    target,
    source_image,
    target_image,
    min_inliers=5,
    ratio=0.95,
    threshold=4.0,
    fixed_scale=None,
    minimum_score=0.25,
):
    matches = feature_matches(source, target, ratio)
    if len(matches) < min_inliers:
        return None
    p = source.points[matches[:, 0]]
    q = target.points[matches[:, 1]]
    rng = np.random.default_rng(0)
    if len(p) * (len(p) - 1) // 2 <= 1800:
        pairs = list(itertools.combinations(range(len(p)), 2))
    else:
        pairs = sorted(
            {tuple(sorted(rng.choice(len(p), 2, replace=False))) for _ in range(2200)}
        )
    best = None
    for i, j in pairs:
        if np.linalg.norm(p[i] - p[j]) < 15 or np.linalg.norm(q[i] - q[j]) < 15:
            continue
        a, b = similarity(p[[i, j]], q[[i, j]], fixed_scale)
        errors = np.linalg.norm(p @ a.T + b - q, axis=1)
        good = errors < threshold
        rank = (
            int(good.sum()),
            -float(np.median(errors[good])) if good.any() else -1e9,
        )
        if best is None or rank > best[0]:
            best = rank, good
    if best is None or best[0][0] < min_inliers:
        return None
    good = best[1]
    for _ in range(4):
        if good.sum() < min_inliers:
            return None
        a, b = similarity(p[good], q[good], fixed_scale)
        errors = np.linalg.norm(p @ a.T + b - q, axis=1)
        good = errors < threshold
    if good.sum() < min_inliers:
        return None
    fraction_source = spread(p[good], source.size)
    fraction_target = spread(q[good], target.size)
    if max(fraction_source, fraction_target) < 0.01:
        return None
    score, overlap = photometric_score(source_image, target_image, a, b)
    if score < minimum_score or overlap < 0.02:
        return None
    return dict(
        matrix=a,
        translation=b,
        inliers=int(good.sum()),
        matches=len(matches),
        rms_px=float(np.sqrt(np.mean(errors[good] ** 2))),
        source_spread=fraction_source,
        target_spread=fraction_target,
        gradient_score=score,
        overlap_fraction=overlap,
        source_points=p[good],
        target_points=q[good],
    )


def compose_pose(parent_matrix, parent_translation, parent_size, child_size, pair):
    """Compose a top-left pixel registration with a centered, y-up GDS pose."""
    flip = np.diag([1.0, -1.0])
    matrix = parent_matrix @ flip @ pair["matrix"] @ flip
    offset = (
        pair["matrix"] @ (np.array(child_size) / 2)
        + pair["translation"]
        - np.array(parent_size) / 2
    )
    translation = parent_translation + parent_matrix @ flip @ offset
    return matrix, translation


def mosaic(images, poses, output, max_dimension=2400):
    """Render in footprint order: large fields first, close-ups last."""
    flip = np.array([1.0, -1.0])
    bounds = []
    for image, (matrix, t) in zip(images, poses):
        corners = np.array(list(itertools.product([0, image.width], [0, image.height])))
        bounds.append(((corners - np.array(image.size) / 2) * flip) @ matrix.T + t)
    points = np.concatenate(bounds)
    low, high = points.min(0), points.max(0)
    sampling = max(high - low) / max_dimension
    size = np.maximum(1, np.ceil((high - low) / sampling).astype(int))
    canvas = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    areas = [
        image.width * image.height * np.linalg.det(pose[0])
        for image, pose in zip(images, poses)
    ]
    for index in sorted(range(len(images)), key=lambda i: (-areas[i], i)):
        image = images[index]
        matrix, t = poses[index]
        poly = bounds[index]
        left, bottom = np.floor((poly.min(0) - low) / sampling).astype(int)
        right, top = np.ceil((poly.max(0) - low) / sampling).astype(int)
        left = max(0, left)
        bottom = max(0, bottom)
        right = min(size[0], right)
        top = min(size[1], top)
        yy, xx = np.mgrid[bottom:top, left:right]
        q = np.column_stack(
            [
                low[0] + (xx.ravel() + 0.5) * sampling,
                low[1] + (yy.ravel() + 0.5) * sampling,
            ]
        )
        pixels = np.linalg.solve(matrix, (q - t).T).T * flip + np.array(image.size) / 2
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < image.width - 1)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < image.height - 1)
        )
        rgb = np.array(image.convert("RGB"))
        values = np.column_stack(
            [
                nd.map_coordinates(
                    rgb[:, :, channel], [pixels[:, 1], pixels[:, 0]], order=1
                )
                for channel in range(3)
            ]
        )
        region = canvas[bottom:top, left:right].copy().reshape(-1, 3)
        region[valid] = values[valid]
        canvas[bottom:top, left:right] = region.reshape(top - bottom, right - left, 3)
    Image.fromarray(canvas[::-1]).save(output)
    return dict(
        origin_um=low.tolist(), pixel_size_um=float(sampling), size_px=size.tolist()
    )


def refine_registration(source_image, target_image, pair):
    """Refine a feature initialization by coarse-to-fine gradient correlation."""
    from scipy.optimize import minimize

    source = np.asarray(source_image.convert("L"), float)
    target = np.asarray(target_image.convert("L"), float)
    scale = np.sqrt(np.linalg.det(pair["matrix"]))
    angle = np.arctan2(pair["matrix"][1, 0], pair["matrix"][0, 0])
    initial = np.array([np.log(scale), angle, *pair["translation"]])
    ys = np.linspace(4, source.shape[0] - 5, 120)
    xs = np.linspace(4, source.shape[1] - 5, 160)
    y, x = np.meshgrid(ys, xs, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel()])
    params = initial.copy()
    for sigma in [6.0, 3.0]:
        sg = [
            nd.gaussian_filter(source, sigma / scale, order=axis)
            - nd.gaussian_filter(source, 6 * sigma / scale, order=axis)
            for axis in [(0, 1), (1, 0)]
        ]
        tg = [
            nd.gaussian_filter(target, sigma, order=axis)
            - nd.gaussian_filter(target, 6 * sigma, order=axis)
            for axis in [(0, 1), (1, 0)]
        ]
        s = np.column_stack(
            [nd.map_coordinates(v, [points[:, 1], points[:, 0]], order=1) for v in sg]
        )
        norms = np.linalg.norm(s, axis=1)
        s *= np.minimum(1, np.percentile(norms, 98) / np.maximum(norms, 1e-12))[:, None]

        def objective(values):
            sc = np.exp(values[0])
            c, ss = np.cos(values[1]), np.sin(values[1])
            rotation = np.array([[c, -ss], [ss, c]])
            mapped = points @ (sc * rotation).T + values[2:]
            valid = (
                (mapped[:, 0] > 2)
                & (mapped[:, 0] < target.shape[1] - 3)
                & (mapped[:, 1] > 2)
                & (mapped[:, 1] < target.shape[0] - 3)
            )
            if valid.mean() < 0.2:
                return 1.0
            t = np.column_stack(
                [
                    nd.map_coordinates(v, [mapped[valid, 1], mapped[valid, 0]], order=1)
                    for v in tg
                ]
            )
            norms = np.linalg.norm(t, axis=1)
            t *= np.minimum(1, np.percentile(norms, 98) / np.maximum(norms, 1e-12))[
                :, None
            ]
            transformed = s[valid] @ rotation.T
            return -np.sum(transformed * t) / max(
                np.linalg.norm(transformed) * np.linalg.norm(t), 1e-12
            )

        bounds = [
            (initial[0] - 0.16, initial[0] + 0.16),
            (initial[1] - 0.06, initial[1] + 0.06),
            (initial[2] - 50, initial[2] + 50),
            (initial[3] - 50, initial[3] + 50),
        ]
        # Powell can handle the resampling objective without numerical gradients.
        result = minimize(
            objective,
            params,
            method="Powell",
            bounds=bounds,
            options=dict(maxiter=35, xtol=0.001, ftol=1e-5),
        )
        if objective(result.x) < objective(params):
            params = result.x
    sc = np.exp(params[0])
    c, s = np.cos(params[1]), np.sin(params[1])
    matrix = sc * np.array([[c, -s], [s, c]])
    score, overlap = photometric_score(source_image, target_image, matrix, params[2:])
    if score < pair["gradient_score"]:
        return pair
    result = dict(
        pair,
        matrix=matrix,
        translation=params[2:],
        gradient_score=score,
        overlap_fraction=overlap,
    )
    result["refined"] = True
    if "source_points" in pair:
        errors = np.linalg.norm(
            pair["source_points"] @ matrix.T + params[2:] - pair["target_points"],
            axis=1,
        )
        result["rms_px"] = float(np.sqrt(np.mean(errors**2)))
    return result

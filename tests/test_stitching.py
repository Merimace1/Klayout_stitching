import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy import ndimage as nd

from alignment_config import AlignmentConfig
from align_images import fit_labels
from batch_alignment import align_batch, write_outputs
from stitching import similarity, compose_pose, extract_features, register_pair, mosaic


class StitchingTest(unittest.TestCase):
    def test_similarity_estimates_magnification_without_shear(self):
        points = np.array([[0.0, 0.0], [100, 0], [0, 200], [200, 150]])
        angle = 0.07
        expected = 0.23 * np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        matrix, translation = similarity(points, points @ expected.T + [20, 40])
        np.testing.assert_allclose(matrix, expected, atol=1e-12)
        np.testing.assert_allclose(translation, [20, 40], atol=1e-12)
        singular = np.linalg.svd(matrix, compute_uv=False)
        self.assertAlmostEqual(singular[0], singular[1])

    def test_automatic_marker_fit_skips_duplicate_labels(self):
        pixels = np.array(
            [[100, 100], [500, 120], [120, 500], [520, 520], [800, 800]], float
        )
        p = (pixels - [500, 500]) * [1, -1]
        q = p[:4] * 0.7 + [100, 200]
        library = [{"center": point} for point in q]
        detections = [
            dict(index=i, pixel=point.tolist()) for i, point in enumerate(pixels[:4])
        ]
        detections.append(dict(index=0, pixel=pixels[-1].tolist()))
        matrix, t, good = fit_labels(
            detections, library, (1000, 1000), 0.9, estimate_scale=True
        )
        np.testing.assert_allclose(matrix, 0.7 * np.eye(2), atol=1e-12)
        np.testing.assert_allclose(t, [100, 200], atol=1e-12)
        self.assertFalse(good[-1])

    def test_pose_composition_respects_center_and_y_direction(self):
        parent_matrix = np.array([[0.7, 0.02], [-0.02, 0.7]])
        parent_translation = np.array([120.0, 400.0])
        relative = np.array([[0.2, -0.01], [0.01, 0.2]])
        offset = np.array([300.0, 450.0])
        parent_size = (2000, 1500)
        child_size = (800, 600)
        a, t = compose_pose(
            parent_matrix,
            parent_translation,
            parent_size,
            child_size,
            dict(matrix=relative, translation=offset),
        )
        p = np.array([[10.0, 20.0], [700.0, 450.0]])
        expected = (
            ((p @ relative.T + offset) - np.array(parent_size) / 2) * [1, -1]
        ) @ parent_matrix.T + parent_translation
        actual = ((p - np.array(child_size) / 2) * [1, -1]) @ a.T + t
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_multiscale_registration_of_textured_crop(self):
        rng = np.random.default_rng(3)
        pixels = nd.gaussian_filter(rng.normal(size=(400, 500)), 2)
        pixels = (pixels - pixels.min()) / np.ptp(pixels) * 255
        parent = Image.fromarray(pixels.astype("uint8"))
        child = parent.crop((110, 90, 310, 240)).resize(
            (400, 300), getattr(Image, "Resampling", Image).LANCZOS
        )
        pair = register_pair(
            extract_features(child), extract_features(parent), child, parent
        )
        self.assertIsNotNone(pair)
        np.testing.assert_allclose(pair["matrix"], 0.5 * np.eye(2), atol=0.008)
        np.testing.assert_allclose(pair["translation"], [110, 90], atol=2)

    def test_small_physical_footprint_is_visible_on_top(self):
        overview = Image.new("RGB", (40, 40), (0, 0, 255))
        closeup = Image.new("RGB", (80, 80), (255, 0, 0))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mosaic.png"
            # Put the closeup first in the input list to test explicit sorting.
            mosaic(
                [closeup, overview],
                [(0.1 * np.eye(2), np.zeros(2)), (np.eye(2), np.zeros(2))],
                path,
                max_dimension=100,
            )
            rendered = np.asarray(Image.open(path))
            np.testing.assert_array_equal(rendered[50, 50], [255, 0, 0])
            np.testing.assert_array_equal(rendered[20, 20], [0, 0, 255])

    def test_export_handles_projective_reference_and_sorted_z_positions(self):
        images = {
            "large.png": Image.new("RGB", (40, 40), "blue"),
            "small.png": Image.new("RGB", (80, 80), "red"),
        }
        poses = {
            name: dict(
                matrix=scale * np.eye(2), translation=np.zeros(2), method="markers"
            )
            for name, scale in (("large.png", 1.0), ("small.png", 0.1))
        }
        reference = {
            "large.png": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.001, 0.0, 1.0]])
        }
        polygons = [np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]])]
        data = dict(
            cell="Test",
            polygons=polygons,
            layers=[dict(layer=1, datatype=0)],
            backend="test",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with patch("batch_alignment.mosaic", return_value={}):
                report = write_outputs(
                    [path / name for name in images],
                    images,
                    poses,
                    {},
                    data,
                    path / "pattern.gds",
                    path,
                    AlignmentConfig(),
                    reference,
                    lambda *args: None,
                    [1.0],
                )
            self.assertEqual(
                report["draw_order_large_to_small"], ["large.png", "small.png"]
            )
            self.assertEqual(
                [image["z_position"] for image in report["images"]], [1, 2]
            )
            self.assertEqual(report["images"][0]["example_center_difference_um"], 0.0)
            self.assertGreater(report["images"][0]["example_corner_rms_um"], 0.0)

    def test_failed_image_does_not_discard_anchor(self):
        with tempfile.TemporaryDirectory() as folder:
            files = [Path(folder) / name for name in ("anchor.png", "unmatched.png")]
            for i, path in enumerate(files):
                Image.new("RGB", (40, 40), (i * 100, 0, 0)).save(path)

            def align(image, polygons, scale, config, estimate_scale=False):
                if image.getpixel((0, 0))[0]:
                    raise ValueError("No labels")
                return dict(
                    matrix=np.eye(2),
                    translation=np.zeros(2),
                    markers=[dict(residual_um=0)],
                    marker_rms_um=0,
                    recognized_label_inliers=6,
                )

            with patch("batch_alignment.register_pair", return_value=None):
                _, poses, failed = align_batch(
                    files, [], [1.0], AlignmentConfig(), align
                )
            self.assertIn("anchor.png", poses)
            self.assertIn("unmatched.png", failed)


if __name__ == "__main__":
    unittest.main()

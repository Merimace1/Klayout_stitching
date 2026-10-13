import unittest
import numpy as np
from pathlib import Path
from PIL import Image
from align_images import rigid, fit_labels, align_marker_image
from alignment_config import AlignmentConfig
from gds_io import load_geometry


class AlignmentTest(unittest.TestCase):
    def test_gold_photographs_anchor_to_expected_rows(self):
        root = Path(__file__).resolve().parents[1]
        folder = root / "new_examples" / "gold_markers"
        if not all((folder / name).is_file() for name in ("1.jpg", "2.jpg", "3.jpg")):
            self.skipTest("Optional gold photographs are not present")
        polygons = load_geometry(root / "Marker_4x4_new.gds")["polygons"]
        for name, rows in (("1.jpg", "DE"), ("2.jpg", "FG"), ("3.jpg", "FG")):
            with self.subTest(image=name), Image.open(folder / name) as image:
                pose = align_marker_image(
                    image, polygons, 0.34484, AlignmentConfig(marker_type="gold"), True
                )
                self.assertGreaterEqual(len(pose["markers"]), 3)
                self.assertTrue(all(m["label"][0] in rows for m in pose["markers"]))
                self.assertLess(pose["marker_rms_um"], 2)
                self.assertAlmostEqual(
                    np.sqrt(np.linalg.det(pose["matrix"])), 0.353, delta=0.003
                )

    def test_fixed_scale_rigid_transform(self):
        p = np.array([[0.0, 0.0], [100, 0], [0, 200], [150, 170]])
        angle = 0.13
        expected = 0.7 * np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        shift = np.array([23.0, -41.0])
        matrix, t = rigid(p, p @ expected.T + shift, 0.7)
        np.testing.assert_allclose(matrix, expected, atol=1e-12)
        np.testing.assert_allclose(t, shift, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(matrix), 0.7**2)

    def test_wrong_label_is_rejected(self):
        pixels = np.array(
            [[100, 100], [500, 120], [120, 500], [520, 520], [800, 800]], float
        )
        p = (pixels - [500, 500]) * [1, -1]
        angle = -0.04
        matrix = 0.683 * np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        q = p @ matrix.T + [100, 200]
        q[-1] += [400, 600]
        lib = [{"center": v} for v in q]
        detections = [{"index": i, "pixel": v.tolist()} for i, v in enumerate(pixels)]
        recovered, t, good = fit_labels(detections, lib, (1000, 1000), 0.683)
        self.assertEqual(good.tolist(), [True, True, True, True, False])
        np.testing.assert_allclose(recovered, matrix, atol=1e-12)
        np.testing.assert_allclose(t, [100, 200], atol=1e-12)


if __name__ == "__main__":
    unittest.main()

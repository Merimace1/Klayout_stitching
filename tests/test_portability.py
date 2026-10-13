import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from alignment_config import AlignmentConfig
from align_images import infer_dimensions, label_library
from gds_io import find_executable, load_geometry
from session_io import build_session, reference_matrices


def rectangle(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)


class PortabilityTest(unittest.TestCase):
    def test_explicit_executable_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="path with spaces ") as folder:
            executable = Path(folder) / "klayout_app.exe"
            executable.touch()
            self.assertEqual(
                find_executable(str(executable)), str(executable.resolve())
            )

    def test_windows_executable_on_path(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("gds_io.shutil.which") as which,
        ):
            which.side_effect = lambda name: (
                "C:/KLayout/klayout_app.exe" if name == "klayout_app.exe" else None
            )
            self.assertEqual(find_executable(), "C:/KLayout/klayout_app.exe")

    def test_missing_dependency_has_actionable_error(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("gds_io.shutil.which", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "--klayout-executable"):
                find_executable()

    def test_python_backend_does_not_spawn_a_process(self):
        data = {"polygons": [[[0, 0], [1, 0], [1, 1], [0, 1]]], "cell": "DifferentCell"}
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("gds_io.importlib.import_module"),
            patch("gds_io.extract_layout", return_value=data),
            patch("gds_io.subprocess.run") as run,
        ):
            result = load_geometry(Path("example.gds"))
            self.assertEqual(result["backend"], "klayout.db")
            run.assert_not_called()

    def test_geometry_sizes_and_marker_side_are_not_fixed(self):
        # Larger letters and square pair to the right, rather than upper-left.
        polygons = [
            rectangle(0, 0, 6, 12),
            rectangle(8, 0, 6, 12),
            rectangle(22, 14, 3, 3),
            rectangle(25, 17, 3, 3),
        ]
        height, side = infer_dimensions(polygons, AlignmentConfig())
        self.assertEqual((height, side), (12, 3))
        library = label_library(polygons, 12 / 12.3)
        self.assertEqual(len(library), 1)
        np.testing.assert_allclose(library[0]["anchor"], [25, 17])

    def test_session_uses_actual_cell_and_cross_platform_filename(self):
        tree = build_session(
            Path("renamed.gds"),
            "ActualCell",
            [rectangle(0, 0, 10, 10)],
            [dict(layer=7, datatype=3, name="Markers")],
        )
        self.assertEqual(tree.findtext(".//source"), "7/3@1")
        self.assertEqual(tree.findtext(".//cellname"), "ActualCell")
        ann = ET.SubElement(tree.find(".//annotations"), "annotation")
        ET.SubElement(ann, "class").text = "img::Object"
        ET.SubElement(
            ann, "value"
        ).text = "color:matrix=(1,0,1) (0,1,2);file='C:\\pictures\\example.PNG'"
        self.assertIn("example.PNG", reference_matrices(tree))

    def test_unknown_config_option_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"typo": 2}))
            with self.assertRaisesRegex(ValueError, "Unknown configuration"):
                AlignmentConfig.from_file(path)

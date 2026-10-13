"""Portable GDS reading through the Python package or desktop KLayout."""

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def extract_layout(db, path, cell=None):
    layout = db.Layout()
    layout.read(str(path))
    if cell:
        top = layout.cell(cell)
        if top is None:
            raise ValueError(f"GDS cell not found: {cell}")
    else:
        tops = list(layout.top_cells())
        if len(tops) != 1:
            raise ValueError("GDS has multiple top cells; select one with --cell")
        top = tops[0]
    polygons = []
    for layer in layout.layer_indexes():
        iterator = top.begin_shapes_rec(layer)
        while not iterator.at_end():
            shape = iterator.shape()
            if shape.is_polygon() or shape.is_box() or shape.is_path():
                polygon = shape.polygon
                polygon = polygon.transformed(iterator.trans())
                polygons.append(
                    [
                        [p.x * layout.dbu, p.y * layout.dbu]
                        for p in polygon.each_point_hull()
                    ]
                )
            iterator.next()
    if not polygons:
        raise ValueError("Selected GDS cell contains no polygon geometry")
    return dict(
        polygons=polygons,
        cell=top.name,
        layers=[
            dict(
                layer=layout.get_info(index).layer,
                datatype=layout.get_info(index).datatype,
                name=layout.get_info(index).name,
            )
            for index in layout.layer_indexes()
        ],
    )


def find_executable(explicit=None):
    requested = explicit or os.environ.get("KLAYOUT_EXECUTABLE")
    if requested:
        path = Path(requested).expanduser()
        found = str(path.resolve()) if path.is_file() else shutil.which(str(requested))
        if not found:
            raise ValueError(f"KLayout executable does not exist: {requested}")
        return found
    for name in ("klayout", "klayout_app.exe", "klayout.exe"):
        found = shutil.which(name)
        if found:
            return found
    # Windows installers often do not put the application on PATH.
    for root in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA", "APPDATA"):
        if not os.environ.get(root):
            continue
        for folder in ("KLayout", "klayout", "Programs/KLayout", "Programs/klayout"):
            for name in ("klayout_app.exe", "klayout.exe"):
                candidate = Path(os.environ[root]) / folder / name
                if candidate.is_file():
                    return str(candidate)
    raise RuntimeError(
        "KLayout is unavailable. Install its Python package with "
        '"python -m pip install klayout", or specify the desktop executable with '
        '--klayout-executable "C:\\path\\to\\klayout_app.exe". '
        "You can also set KLAYOUT_EXECUTABLE."
    )


def load_geometry(path, executable=None, cell=None):
    """Prefer klayout.db unless an executable is explicitly selected."""
    import numpy as np

    if not executable and not os.environ.get("KLAYOUT_EXECUTABLE"):
        try:
            db = importlib.import_module("klayout.db")
        except (ImportError, OSError):
            db = None
        if db is not None:
            data = extract_layout(db, path.resolve(), cell)
            data["backend"] = "klayout.db"
            data["polygons"] = [np.asarray(q, dtype=float) for q in data["polygons"]]
            return data
    application = find_executable(executable)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "geometry.json"
        env = dict(
            os.environ,
            ALIGN_GDS=str(path.resolve()),
            ALIGN_GEOMETRY=str(target),
            ALIGN_CELL=cell or "",
            QT_QPA_PLATFORM="offscreen",
        )
        try:
            result = subprocess.run(
                [
                    application,
                    "-b",
                    "-r",
                    str(Path(__file__).parent / "tools/export_geometry.py"),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "KLayout did not finish reading the GDS within 120 seconds"
            ) from error
        except OSError as error:
            raise RuntimeError(
                f"Cannot start KLayout at {application}: {error}"
            ) from error
        if result.returncode or not target.is_file():
            raise RuntimeError(
                "KLayout could not read the GDS:\n"
                + (result.stderr or result.stdout or "No geometry was exported")
            )
        data = json.loads(target.read_text(encoding="utf-8"))
        data["polygons"] = [np.asarray(q, dtype=float) for q in data["polygons"]]
        data["backend"] = application
        return data

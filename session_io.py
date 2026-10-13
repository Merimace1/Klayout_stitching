"""Read optional calibration sessions and generate machine-independent views."""

from pathlib import PureWindowsPath
import re
import xml.etree.ElementTree as ET

import numpy as np


def reference_matrices(tree):
    reference = {}
    if tree is None:
        return reference
    for annotation in tree.findall(".//annotation"):
        if annotation.findtext("class") != "img::Object":
            continue
        value = annotation.findtext("value", "")
        rows = re.search(r"matrix=\((.*?)\) \((.*?)\)", value)
        filename = re.search(r"file='([^']+)'", value)
        if rows and filename:
            matrix = np.array(
                [list(map(float, rows.group(i).split(","))) for i in (1, 2)]
            )
            if matrix.shape == (2, 3):
                reference[PureWindowsPath(filename.group(1)).name] = matrix
    return reference


def build_session(gds, cell, polygons, layers=()):
    """Create a fresh session; never copy another computer's window state."""
    root = ET.Element("session")
    ET.SubElement(root, "current-view").text = "0"
    layout = ET.SubElement(root, "layout")
    ET.SubElement(layout, "name").text = gds.name
    ET.SubElement(layout, "file-path").text = gds.resolve().as_posix()
    view = ET.SubElement(root, "view")
    ET.SubElement(view, "active-cellview-index").text = "0"
    display = ET.SubElement(view, "display")
    points = np.concatenate(polygons)
    low, high = points.min(0), points.max(0)
    for key, value in zip(
        ("x-left", "x-right", "y-bottom", "y-top"), (low[0], high[0], low[1], high[1])
    ):
        ET.SubElement(display, key).text = str(value)
    ET.SubElement(display, "min-hier").text = "0"
    ET.SubElement(display, "max-hier").text = "100"
    cellpath = ET.SubElement(ET.SubElement(display, "cellpaths"), "cellpath")
    ET.SubElement(cellpath, "cellname").text = cell
    cv = ET.SubElement(ET.SubElement(view, "cellviews"), "cellview")
    ET.SubElement(cv, "layout-ref").text = gds.name
    tabs = ET.SubElement(view, "layer-properties-tabs")
    properties = ET.SubElement(tabs, "layer-properties")
    for layer in layers:
        item = ET.SubElement(properties, "properties")
        values = {
            "expanded": "false",
            "frame-color": "#00ffff",
            "fill-color": "#00ffff",
            "frame-brightness": "0",
            "fill-brightness": "0",
            "dither-pattern": "I9",
            "valid": "true",
            "visible": "true",
            "transparent": "false",
            "marked": "false",
            "xfill": "false",
            "animation": "0",
            "name": layer.get("name", ""),
            "source": f"{layer['layer']}/{layer['datatype']}@1",
        }
        for key, value in values.items():
            ET.SubElement(item, key).text = value
    ET.SubElement(view, "annotations")
    return ET.ElementTree(root)


def image_annotation(path, matrix, translation, z):
    # KLayout has its own string parser inside the XML value. Normalize Windows
    # separators and escape apostrophes separately from XML serialization.
    filename = path.resolve().as_posix().replace("'", "\\'")
    return (
        f"color:matrix=({matrix[0, 0]:.12g},{matrix[0, 1]:.12g},{translation[0]:.12g}) "
        f"({matrix[1, 0]:.12g},{matrix[1, 1]:.12g},{translation[1]:.12g}) (0,0,1);"
        f"min_value=0;max_value=255;is_visible=true;z_position={z};brightness=0;contrast=0;"
        "gamma=1;red_gain=1;green_gain=1;blue_gain=1;landmarks=[];"
        f"color_mapping=[0,'#000000';1,'#ffffff';];file='{filename}'"
    )

"""Run with KLayout -b -r; all input paths come from environment variables."""

import json
import os
from pathlib import Path
import sys
import pya

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gds_io import extract_layout

data = extract_layout(
    pya, Path(os.environ["ALIGN_GDS"]), os.environ.get("ALIGN_CELL") or None
)
with open(os.environ["ALIGN_GEOMETRY"], "w", encoding="utf-8") as output:
    json.dump(data, output)

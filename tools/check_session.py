"""Validate an exported session using KLayout's own image loader.

klayout -z -nc -rx -rd session_path=/absolute/path.lys -r tools/check_session.py
"""

import traceback
import pya

try:
    window = pya.Application.instance().main_window()
    window.restore_session(session_path)
    view = window.current_view()
    images = list(view.each_image())
    iterator = view.begin_layers()
    layers = 0
    while not iterator.at_end():
        layers += 1
        iterator.next()
    if not layers:
        raise ValueError("Session has no visible GDS layer entries")
    if not images:
        raise ValueError("Session contains no images")
    print(f"Loaded {len(images)} images and {layers} GDS layers", flush=True)
    for image in images:
        print(image.matrix, flush=True)
except Exception:
    traceback.print_exc()
    pya.Application.instance().exit(1)
else:
    pya.Application.instance().exit(0)

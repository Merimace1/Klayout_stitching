"""Validate an exported session using KLayout's own image loader.

klayout -z -nc -rx -rd session_path=/absolute/path.lys -r tools/check_session.py
"""
import traceback
import pya

try:
    window = pya.Application.instance().main_window()
    window.restore_session(session_path)
    images = list(window.current_view().each_image())
    if not images:
        raise ValueError('Session contains no images')
    print(f'Loaded {len(images)} images', flush=True)
    for image in images:
        print(image.matrix, flush=True)
except Exception:
    traceback.print_exc()
    pya.Application.instance().exit(1)
else:
    pya.Application.instance().exit(0)

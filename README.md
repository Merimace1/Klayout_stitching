# Automatic alignment of microscope images to the marker GDS

`align_images.py` recognizes the letter/number labels in `pictures/*.png`, locates their adjacent diagonal square pairs, and writes a KLayout session with the images aligned to `Marker_4x4_new.gds`.

The program was run and checked on both supplied images. The original GDS, images, and manually aligned session are preserved.

## Run

Requirements: Python 3.10+, KLayout on PATH with Python macro support, NumPy, SciPy, and Pillow. A separate `klayout` Python package, OpenCV, Tesseract, and downloaded recognition models are unnecessary.

```bash
cd klayout_stitching
# If the Python dependencies are missing:
python3 -m pip install -r requirements.txt
python3 align_images.py --images pictures --gds Marker_4x4_new.gds
klayout -u output/aligned_auto.lys
```

Alternatively, use File → Restore Session in KLayout and choose `output/aligned_auto.lys`.

Outputs:

- `output/aligned_auto.lys`: GDS and automatically positioned images.
- `output/Snap-20015_overlay.png` and `output/Snap-20016_overlay.png`: cyan GDS outlines over the original photographs, for inspection.
- `output/alignment_report.json`: scale, rotation, image-center translation, transformation matrix, accepted marker coordinates, template scores, and fitting residuals.

The `--images` folder and `--gds` file arguments are required. They may be absolute paths or paths relative to your current working directory. By default, output goes into `./output` in that directory; use `--output` to choose another location. The bundled session template is located relative to the script, so you can run it from any directory.

For example, from another working directory:

```bash
python3 /path/to/klayout_stitching/align_images.py \
  --images /path/to/pictures --gds /path/to/Marker_4x4_new.gds \
  --output /path/to/results
```

Custom paths and calibrated scale:

```bash
python3 align_images.py --gds Marker_4x4_new.gds --images pictures \
  --session aligned_images.lys --scale 0.683061072542717 --output output
```

The actual folder supplied is named `pictures`. `--images` also allows a folder named `pcitures` or any other name. The program processes lowercase `*.png` files. Output sessions use absolute paths; rerun after moving the project. Output files in the selected output directory are overwritten on subsequent runs.

## Method

1. KLayout reads and flattens the GDS hierarchy into polygon coordinates in microns. The helper macro `tools/export_geometry.py` runs in a separate batch process.
2. Extract the 8.4 µm high lettering polygons and group adjacent characters into labels. For this particular GDS, columns run 1–9 from left to right and rows A–L from top to bottom. Each label has a pair of 2 × 2 µm squares above and to its left. The lettering is polygon geometry, rather than GDS text records.
3. Rasterize all 108 GDS labels at the calibrated microscope pixel size and blur them slightly. These exact-font templates replace a generic trained OCR model.
4. Subtract image intensity from a Gaussian-smoothed background to emphasize dark lettering and markers over the slowly varying colored substrate. Connected components provide candidate label locations. Normalized template correlation, with a small position search, recognizes labels. Some characters are ambiguous, so recognition alone does not establish a correspondence.
5. Try every pair of recognized labels as an initial rigid pose. Score each pose against all recognized labels and retain the largest consistent set within 8 µm. Refine using all accepted correspondences with a fixed-scale, proper-rotation least-squares fit (SVD). Wrong label readings and unrelated substrate features are rejected through geometric consistency.
6. Project the complete labeled marker grid into the photograph. For each visible labeled position, match a rendered diagonal square pair in a ±6 pixel neighborhood. Accept correlations above 0.5, refit using the actual square-pair centers, and reject residuals over 4 µm. At least three consistent labels and three verified marker pairs are required; ambiguous images fail instead of silently receiving a placement.
7. Insert the resulting image matrices into a copy of the session template. Remove its previous image annotations, retain its view/layer settings, and write overlay previews and numerical diagnostics.

All accepted marker pairs contribute to the final least-squares fit. No random selection is used. Labels determine the absolute grid position, avoiding the translation ambiguity of the repetitive unlabeled marker grid.

## Rotation and translation only

Scale is **fixed**, not fitted. By default, the pixel calibration is the mean of `sqrt(abs(det(A)))` for the image matrices in the supplied session: **0.683061072542717 µm/pixel**. Use `--scale` for a different microscope calibration. This is the only alignment-related information used from the reference session during estimation: its rotations, translations, and landmarks are not used to initialize or determine the new poses. Reference poses are used afterwards to report differences for matching filenames.

For pixel column `u`, pixel row `v`, image width `W`, and height `H`, the exported KLayout transform is:

```text
p = [u - W/2, H/2 - v]       # centered pixels, y upwards
q = scale * R(theta) * p + t # GDS coordinates in microns
```

The report's translation is the image center in GDS coordinates. It is not the upper-left image corner. The 2 × 2 matrix has equal singular values and positive determinant: no fitted shear, unequal axis scaling, reflection, or perspective warp is introduced. KLayout's image matrix convention is documented in its [Image API](https://www.klayout.org/downloads/master/doc-qt4/code/class_Image.html); session loading is described in the [session manual](https://www.klayout.org/downloads/master/doc-qt5/manual/session.html).

## Results on the supplied example

| Image | Recognized label inliers | Verified marker pairs | Rotation | Marker RMS residual | Center difference from example | Corner RMS difference from example |
|---|---:|---:|---:|---:|---:|---:|
| Snap-20015.png | 9 | 19 | −2.3281° | 0.906 µm | 0.486 µm | 1.584 µm |
| Snap-20016.png | 10 | 14 | −2.3784° | 1.735 µm | 0.363 µm | 3.889 µm |

The supplied session contains general affine matrices, including small shear and unequal axis scaling. Consequently, a rigid transform cannot reproduce every point of that session exactly. The recovered centers agree within 0.5 µm; the corner differences above quantify the remaining mismatch over the full image extent. Marker residuals measure consistency with detected photographic features, while example differences measure agreement with the saved manual placement; neither is an independent measurement of physical fabrication accuracy.

Verification performed:

- Full alignment on both supplied PNGs and inspection of the GDS overlay previews.
- Loading the generated `.lys` into installed KLayout and verifying that it contains two images with the exported matrices.
- An additional translated-image experiment: moving Snap-20015 by `(31, -17)` pixels recovered the predicted image-center translation within 0.060 µm and changed the angle by less than 0.005°. That file had a different filename and no reference pose in the session.
- Unit tests for fixed-scale rigid fitting and rejection of an incorrect label correspondence:

```bash
python3 -m unittest discover -s tests -v
```

A KLayout round-trip validation helper is also included:

```bash
QT_QPA_PLATFORM=offscreen klayout -z -nc -rx \
  -rd session_path="$PWD/output/aligned_auto.lys" -r tools/check_session.py
```

## Scope and limitations

This implementation is tailored to the supplied marker design, exact polygon font, dark features, and approximately 0.68 µm/pixel microscope images. The initial label templates are upright, so the tested case is a few degrees of rotation. Arbitrary orientation, strongly different magnification, altered lettering, severe occlusion, and contrast reversal are not validated. Supporting them would require a multi-angle/multi-scale template search or a broader recognition model. Merely supplying a new scale does not retune the candidate-component size thresholds.

Optical distortion and the manual session's affine deformation can leave residuals under the requested rigid model. Inspect the overlay PNGs and per-marker report before using a placement for subsequent experimental work. Recognition and marker thresholds are explicit in `align_images.py` and can be tuned for a new acquisition setup.

This is a local Git repository; no remote hosting or publication has been configured. Inputs are included in the initial snapshot, and generated `output/` files are ignored because they can be reproduced with the command above.

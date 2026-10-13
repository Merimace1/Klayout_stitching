# Automatic alignment of microscope images to a marker GDS

Recognize the polygon lettering in a GDS marker grid, locate the adjacent diagonal square pairs in microscope photographs, and export a KLayout session with rotation and translation only. The physical pixel scale is calibrated separately and held fixed.

## Install and run

Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
python align_images.py --gds ./Marker_4x4_new.gds --images ./pictures
```

The preferred GDS reader is the `klayout.db` Python package, which does not need the desktop application installed or on PATH. The desktop application is needed only to view the exported `.lys` file. See the [official Python package documentation](https://www.klayout.org/klayout-pypi/).

For the supplied example, the command above uses pixel calibration from the bundled `aligned_images.lys`. For another microscope setup, provide its calibrated microns per pixel:

```bash
python align_images.py --gds /path/to/pattern.gds --images /path/to/images \
  --scale 0.683061072542717 --output /path/to/results
```

With `--scale`, no example session is required. All input paths can be absolute or relative to the current working directory. Output defaults to `./output` in that directory. PNG, JPEG, TIFF, and BMP files are accepted, with case-insensitive extensions. A folder should contain images from the same pixel calibration.

### Windows: resolving WinError 2

The earlier version launched an executable named `klayout`. `WinError 2` at `subprocess.Popen` means Windows could not find that executable; it does not indicate that the GDS or photographs are missing.

The refactored reader tries these alternatives:

1. Import `klayout.db` directly.
2. Discover desktop KLayout on PATH, including `klayout_app.exe` and `klayout.exe`.
3. Search common Windows installation folders under Program Files and AppData.

An explicit executable path or `KLAYOUT_EXECUTABLE` environment variable overrides automatic selection. For example, in PowerShell, using the actual location of your installation:

```powershell
python align_images.py --gds .\Marker_4x4_new.gds --images .\pictures --klayout-executable "C:\Program Files\KLayout\klayout_app.exe"
```

The executable can also be in an extracted ZIP installation. KLayout documents the Windows executable name in its [installation documentation](https://www.klayout.de/build.html).

If `pip install klayout` cannot install a compatible package for your Python version, the desktop fallback remains available. Install only the processing dependencies in that case:

```powershell
python -m pip install numpy scipy Pillow
```

Then run with `--klayout-executable`. Python 3.14 on Windows has not been tested in this workspace; both GDS-reader APIs were exercised through the installed Linux KLayout environment, and Windows discovery behavior is covered by mocked tests. Missing dependencies and incorrect executable paths now produce actionable CLI errors instead of the original unhandled traceback.

## Options

| Argument | Meaning |
|---|---|
| `--gds FILE` | Required GDS input; arbitrary path and filename |
| `--images FOLDER` | Required image folder |
| `--output FOLDER` | Output location; default `./output` |
| `--scale NUMBER` | Fixed calibrated µm/pixel; no reference session needed |
| `--session FILE` | Optional calibration session and numerical comparison reference |
| `--klayout-executable FILE` | Use a particular desktop KLayout installation |
| `--cell NAME` | Select a GDS cell; required when there are multiple top cells |
| `--config FILE` | JSON recognition settings; see `config.example.json` |

When `--scale` is omitted, calibration is the mean `sqrt(abs(det(A)))` over the session's image matrices. The bundled session is used if `--session` is also omitted. No reference rotations, translations, or landmarks are used during estimation. An explicitly supplied session can also be used for post-fit comparisons of matching image filenames, even when `--scale` is set.

## Outputs

- `aligned_auto.lys`: fresh session pointing to the selected GDS cell and aligned images.
- `<image-name>_overlay.png`: cyan GDS outlines on the original photograph.
- `alignment_report.json`: geometry backend, inferred feature dimensions, configuration, rotations, image-center translations, matrices, accepted markers, and residuals.

Open the session with File → Restore Session in KLayout. On systems with the application on PATH:

```bash
klayout -u output/aligned_auto.lys
```

Session paths are absolute and Windows paths are normalized for KLayout's string parser. Rerun after moving the project. Existing output files with matching names are overwritten. Images with the same stem but different extensions share an overlay filename; use distinct stems. The original inputs are never changed.

## Method and organization

- `gds_io.py` reads the selected cell through the Python database API or a desktop batch macro. It flattens hierarchy into micron coordinates and includes polygon, box, and path geometry. Multiple top cells require an explicit selection.
- `alignment_config.py` defines and validates recognition parameters. The most frequent square size and taller character-polygon height are inferred from the design. They can be overridden in JSON for a design with competing geometry.
- `align_images.py` groups adjacent lettering polygons into a library of exact GDS label templates and associates each with the nearest compact diagonal square pair. Pair association no longer requires a particular side of the lettering. The library depends on the supplied GDS, rather than a trained model or a hardcoded list of image labels.
- Images are resampled for detection so lettering has a standard working pixel height. This lets candidate sizes and matching windows work across different image resolutions and uniform design scaling. Fits are mapped back into original pixel coordinates; exported scale remains fixed at the supplied calibration.
- Background subtraction emphasizes small dark features. Connected components propose label locations; normalized template correlation recognizes their GDS shapes. All pairs of recognized labels are tried to find the largest geometrically consistent set, then a proper-rotation least-squares fit rejects incorrect readings.
- The pose projects labeled marker locations into the photograph. Local template matching verifies the actual diagonal square pairs, and all accepted pair centers contribute to the final rigid fit. At least three consistent labels and three verified marker pairs are required.
- `session_io.py` generates a fresh session from the actual selected GDS cell and bounds. It does not copy the supplied session's cell name, window geometry, absolute input paths, layer state, or landmarks.

Defaults are explicit in `config.example.json`. For example, override `polarity` to `bright` for bright markers, reduce `contrast_threshold` for weak contrast, or set `letter_height_um` and `square_size_um` when automatic dimension inference selects unrelated features. Correlation scores range up to 1; geometric residual limits are in physical microns. If the entire pattern is enlarged, scale those residual limits as appropriate. Configuration keys are validated; unknown keys are rejected.

Grid labels in reports are diagnostic row/column identifiers (A1, A2, etc.), not a transcription by a generic OCR engine. Correspondences are established by matching each GDS polygon template to the photograph. The number of rows, columns, actual dimensions, grid spacing, cell name, and filenames are not fixed to the example.

## Coordinate convention

For pixel column `u`, row `v`, image width `W`, and height `H`:

```text
p = [u - W/2, H/2 - v]
q = scale * R(theta) * p + t
```

`q` is in GDS microns; `t` is the image center in the layout, not its upper-left corner. The matrix has equal singular values and positive determinant: no fitted shear, reflection, perspective, or unequal axis scaling. See the [KLayout Image API](https://www.klayout.org/downloads/master/doc-qt4/code/class_Image.html).

## Validation

The refactor retains the original supplied-example results:

| Image | Label inliers | Verified pairs | Rotation | Marker RMS | Example center difference | Example corner RMS difference |
|---|---:|---:|---:|---:|---:|---:|
| Snap-20015.png | 9 | 19 | −2.3281° | 0.906 µm | 0.486 µm | 1.584 µm |
| Snap-20016.png | 10 | 14 | −2.3784° | 1.735 µm | 0.363 µm | 3.889 µm |

The manual example includes slight shear and unequal axis scaling, so a rigid placement cannot reproduce every point exactly. Fitting residuals measure consistency with photographic detections, not independent fabrication accuracy.

Additional checks:

- Both GDS-reader APIs were exercised on the supplied GDS, returning 5,621 polygons.
- Fresh exported sessions load into installed KLayout with the correct images and matrices.
- A renamed, half-resolution photograph aligned with doubled µm/pixel calibration; the center differed from the original result by 0.139 µm and rotation by 0.0076°.
- A translated, uniformly enlarged GDS with a different cell name was recognized using inferred 25.2 µm lettering and 6 µm squares. The corresponding calibrated scale and residual thresholds were supplied explicitly.
- A previous translated-image experiment recovered a `(31, -17)` pixel shift within 0.060 µm, with angle change below 0.005°.

Run the tests:

```bash
python -m unittest discover -s tests -v
```

They cover rigid fitting, false correspondence rejection, inferred dimensions, markers on another side of a label, Windows executable names, paths with spaces, the direct Python backend, session cell selection, cross-platform filename parsing, and configuration validation.

Optional desktop session-loading check on Linux:

```bash
QT_QPA_PLATFORM=offscreen klayout -z -nc -rx \
  -rd session_path="$PWD/output/aligned_auto.lys" -r tools/check_session.py
```

## Remaining assumptions

This is a template matcher for a regular grid of polygon lettering and small diagonal square pairs. Character polygons must have a common height and row baseline, and labels must be separated enough to group by their origins. Designs using GDS text objects, arbitrary lettering arrangements, polygon holes, unrelated dominant square geometry, or significantly different label proportions may need additional extraction logic or configuration.

Templates are initially upright; small rotations are tested, arbitrary orientations are not. Heavy occlusion, perspective distortion, and significant optical distortion remain outside the validated scope. The design contains the recognition reference, so changing font shapes can work if grouping and marker association still meet these assumptions; this is not a universal OCR solution.

Generated `output/` files are ignored by Git. Inputs and reproducible scripts are tracked. The repository remote is configured; publishing depends on network access and GitHub authentication in the environment running `git push`.

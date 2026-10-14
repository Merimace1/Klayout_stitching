# Automatic alignment of microscope images to a marker GDS

Align mixed-magnification microscope photographs to a GDS marker grid. Images with enough readable labels anchor directly to the GDS; other images register through overlapping surface features in those anchors or intermediate photographs. The output uses rotation, translation, and a uniform pixel scale, with no shear or perspective warp. Smaller physical fields are placed on top of larger ones.

# How to use

- the code does not do a great job of scaling the images that it uses as anchors, therefore, it is recommended to use --scale-candidates (for 5x its ~ 0.68 um/pixel and for 10x its ~0.34 um/pixel)
- the programme needs recognizable letters to anchor first few images, thus letters need to be clearly recognizable from the pictures
- from my experience using 5x magnification images as anchors works the best
- most of the testing was done on developed markers, thus it is assumed it works better for developed marker system rather than gold ones, for gold markers one should use special setting, see below.

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

With `--scale`, no example session is required. All input paths can be absolute or relative to the current working directory. Output defaults to `./output` in that directory. PNG, JPEG, TIFF, and BMP files are accepted, with case-insensitive extensions. Mixed magnifications are supported within a folder.

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
| `--scale NUMBER` | Fixed µm/pixel for direct marker attempts; overlap registration can infer other images' scales |
| `--scale-map FILE` | JSON map of individual image filenames to fixed calibrated µm/pixel |
| `--scale-candidates LIST` | Comma-separated scale seeds for automatic magnification search |
| `--marker-type developed\|gold` | Marker appearance; defaults to the existing developed detector |
| `--no-stitch` | Disable overlap registration; save successful marker alignments |
| `--session FILE` | Optional calibration session and numerical comparison reference |
| `--klayout-executable FILE` | Use a particular desktop KLayout installation |
| `--cell NAME` | Select a GDS cell; required when there are multiple top cells |
| `--config FILE` | JSON recognition settings; see `config.example.json` |

Without `--scale`, the widest-field calibration in the session seeds a multiscale label search, together with other session calibrations and half-octave scale candidates. The bundled session supplies an overview seed if `--session` is omitted. Calibrations from different magnifications are **not averaged**. Once labels establish correspondences, their GDS separation estimates the image's uniform scale. This replaces the previous single-scale default. No reference translations, rotations, or landmarks are used during alignment; full reference matrices are used only for post-fit comparisons.

To operate without any session and search your own scales:

```bash
python align_images.py --gds pattern.gds --images pictures \
  --scale-candidates 0.7,0.35,0.07
```

If calibration is known, prefer a scale map, for example:

```json
{"overview.png": 0.683, "medium.png": 0.344, "closeup.png": 0.070}
```

Use `--scale-map scales.json`. Entries fix both direct marker scale and stitched relative scale. Images absent from the map use automatic estimation. With just `--scale`, that value is fixed for direct marker attempts on every image; images that fail can still obtain another scale from overlap registration. Use `--no-stitch --scale NUMBER` for the previous fixed-scale marker-only behavior.

## Outputs

- `aligned_auto.lys`: fresh session pointing to the selected GDS cell and aligned images.
- `mosaic.png`: combined overview, preserving original image colors and explicit overlay boundaries.
- `<image-name>_overlay.png`: cyan GDS outlines on the original photograph.
- `alignment_report.json`: geometry backend, inferred feature dimensions, configuration, rotations, image-center translations, matrices, accepted markers, and residuals.

Open the session with File → Restore Session in KLayout. On systems with the application on PATH:

```bash
klayout -u output/aligned_auto.lys
```

Session paths are absolute and Windows paths are normalized for KLayout's string parser. Rerun after moving the project. Existing output files with matching names are overwritten. Images with the same stem but different extensions share an overlay filename; use distinct stems. The original inputs are never changed. Physical footprint is `width × height × determinant(pixel-to-GDS matrix)`, rather than pixel count or filename order. Images are serialized largest to smallest with increasing KLayout `z_position`, and the mosaic uses the same order.

## Method and organization

- `gds_io.py` reads the selected cell through the Python database API or a desktop batch macro. It flattens hierarchy into micron coordinates and includes polygon, box, and path geometry. Multiple top cells require an explicit selection.
- `alignment_config.py` defines and validates recognition parameters. The most frequent square size and taller character-polygon height are inferred from the design. They can be overridden in JSON for a design with competing geometry.
- `align_images.py` groups adjacent lettering polygons into a library of exact GDS label templates and associates each with the nearest compact diagonal square pair. Pair association no longer requires a particular side of the lettering. The library depends on the supplied GDS, rather than a trained model or a hardcoded list of image labels.
- Images are resampled for detection so lettering has a standard working pixel height. This lets candidate sizes and matching windows work across different image resolutions and uniform design scaling. Fits are mapped back into original pixel coordinates; supplied per-image calibrations are held fixed, while automatic mode estimates a uniform scale from verified correspondences.
- Background subtraction emphasizes small dark features. Connected components propose label locations; normalized template correlation recognizes their GDS shapes. All pairs of recognized labels are tried to find the largest geometrically consistent set, then a proper-rotation least-squares fit rejects incorrect readings.
- The pose projects labeled marker locations into the photograph. Local template matching verifies the actual diagonal square pairs, and all accepted pair centers contribute to the final rigid fit. At least three consistent labels and three verified marker pairs are required.
- `stitching.py` extracts Harris corners over an image pyramid and builds orientation-normalized 4 × 4 × 8 gradient-histogram descriptors. This follows the SIFT descriptor principle but is a small SciPy implementation, not the original SIFT detector or a pretrained model. Mutual nearest-neighbor matching and a descriptor ratio check produce tentative correspondences. Deterministic RANSAC and least-squares fitting estimate relative uniform scale, rotation, and translation.
- Registration verifies correspondence count, spatial spread, overlap, and band-pass image-gradient agreement. Sparse close-ups get a wider feature initialization tolerance followed by bounded coarse-to-fine gradient-correlation refinement. At least five independent feature matches and a final gradient score of 0.25 are required by default. The report flags registrations with fewer than eight inliers as `limited_features`.
- `batch_alignment.py` tries markers on every image, keeps successful anchors, then grows an overlap graph in rounds. A close-up can register through an intermediate photograph even if it does not overlap an overview enough to match directly. Anchors remain fixed; the report records each image's parent, anchor, graph depth, inferred scale, and match diagnostics. No training or model downloads are required.
- `session_io.py` generates a fresh session from the actual selected GDS cell and bounds. It does not copy the supplied session's cell name, window geometry, absolute input paths, layer state, or landmarks.

Defaults are explicit in `config.example.json`. For example, override `polarity` to `bright` for bright markers, reduce `contrast_threshold` for weak contrast, or set `letter_height_um` and `square_size_um` when automatic dimension inference selects unrelated features. Correlation scores range up to 1; geometric residual limits are in physical microns. If the entire pattern is enlarged, scale those residual limits as appropriate. Stitching parameters (`stitching_min_inliers`, descriptor ratio, ordinary and sparse residual limits, gradient threshold, pyramid levels, feature count, and working resolution) are also configurable. Stitch residuals are in pixels of the parent image. Configuration keys are validated; unknown keys are rejected.

One failed image no longer stops the entire batch. Successful placements and a session are saved, and the report's `unresolved` section explains failures. Exit status is 0 when all inputs align, 1 for partial success, and 2 when no image anchors or the command is invalid. Images without a verified overlap path to an anchor remain unresolved.

Grid labels in reports are diagnostic row/column identifiers (A1, A2, etc.), not a transcription by a generic OCR engine. Correspondences are established by matching each GDS polygon template to the photograph. The number of rows, columns, actual dimensions, grid spacing, cell name, and filenames are not fixed to the example.

## Coordinate convention

For pixel column `u`, row `v`, image width `W`, and height `H`:

```text
p = [u - W/2, H/2 - v]
q = scale * R(theta) * p + t
```

`q` is in GDS microns; `t` is the image center in the layout, not its upper-left corner. The matrix has equal singular values and positive determinant: no fitted shear, reflection, perspective, or unequal axis scaling. See the [KLayout Image API](https://www.klayout.org/downloads/master/doc-qt4/code/class_Image.html).

## Mixed-magnification examples

Real inputs were tested from `/media/merlin/INTENSO/matus_tmp/stitching_programme/new_examples`. Outputs are written inside this project, not back to the external drive:

```bash
python align_images.py --gds Marker_4x4_new.gds \
  --images /media/merlin/INTENSO/matus_tmp/stitching_programme/new_examples/pictures \
  --session /media/merlin/INTENSO/matus_tmp/stitching_programme/new_examples/session.lys \
  --output output_multimag1

python align_images.py --gds Marker_4x4_new.gds \
  --images /media/merlin/INTENSO/matus_tmp/stitching_programme/new_examples/pictures2 \
  --session /media/merlin/INTENSO/matus_tmp/stitching_programme/new_examples/session2.lys \
  --output output_multimag2
```

The first dataset aligns five of five images: Snap-20015 and Snap-20016 are GDS anchors; Snap-20014 registers to Snap-20015, Snap-20012 through Snap-20014, and Snap-20013 through Snap-20012. Snap-20013 is an additional tighter close-up without a placement in the supplied reference session. The second dataset aligns four of four: Snap-20019, Snap-20020, and Snap-20021 are GDS anchors; Snap-20018 registers through Snap-20019.

The examples cover approximately 0.68, 0.345, 0.070, and 0.035 µm/pixel. Source image pixel dimensions are all 2560 × 1920, so sorting by pixel dimensions would fail to establish the required layer order. The tightest close-up is on top in each session and mosaic. Snap-20018 has limited feature evidence; inspect its overlay and gradient score when applying this method to another sample.

See `validation/multimag_results.json` for measured comparisons with the manual sessions. Some reference placements contain shear or perspective; full reference matrices are used for numerical validation, whereas generated placements remain similarities. Center differences for the eight photographs with reference placements are below 1.4 µm. Corner RMS differences range from 0.4 to 22.2 µm; the largest is Snap-20021, whose reference includes perspective deformation that a similarity cannot reproduce. These differences measure agreement with manual placement, not independent physical accuracy. The second-folder run aligns its four available files; the manual session2 also references images from the first folder.

## Validation

For the supplied gold-marker photographs, run:

```bash
python3 align_images.py --gds Marker_4x4_new.gds \
  --images new_examples/gold_markers --marker-type gold \
  --scale-candidates 0.34484 --output output_gold
```

Gold mode uses bright red-channel contrast and fills small holes in the GDS label templates to account for the photographed gold lettering. It retains several similar label hypotheses for the existing geometric fit to disambiguate. Square-pair verification uses the existing search radius and a gold contrast threshold (`gold_marker_score`, default 0.4; developed mode retains `marker_score`, default 0.5). The same three-label/three-pair requirements and physical residual limits still apply. Developed mode retains the previous contrast, templates, and anchoring behavior; `polarity` controls developed mode only. The mode can also be set as `"marker_type": "gold"` in JSON; the CLI overrides it.

`--scale-candidates` supplies a seed, allowing the marker fit and overlaps to estimate calibration. Use `--scale` to fix direct marker calibration. No reference placements were supplied for the gold photographs, so overlays and residuals establish internal consistency rather than independent absolute accuracy.

This command aligns all five supplied gold images: `1.jpg`, `2.jpg`, and `3.jpg` anchor directly with 3, 6, and 6 verified pairs (RMS 0.521, 0.727, and 1.026 µm). `4.jpg` and `mid.jpg` register through `3.jpg`. The overview scales converge to 0.3528–0.3538 µm/pixel; `mid.jpg` converges to 0.07082 µm/pixel, indicating a smaller physical field than the other photographs.

Fixed-scale marker-only mode retains the original supplied-example results:

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

Tests cover rigid and similarity fitting, duplicate-label rejection, multiscale crop registration, image-to-GDS pose composition, smallest-footprint visibility, partial-batch failures, and the portability/configuration checks described above.

Optional desktop session-loading check on Linux:

```bash
QT_QPA_PLATFORM=offscreen klayout -z -nc -rx \
  -rd session_path="$PWD/output/aligned_auto.lys" -r tools/check_session.py
```

## Remaining assumptions

This is a template matcher for a regular grid of polygon lettering and small diagonal square pairs. Character polygons must have a common height and row baseline, and labels must be separated enough to group by their origins. Designs using GDS text objects, arbitrary lettering arrangements, polygon holes, unrelated dominant square geometry, or significantly different label proportions may need additional extraction logic or configuration.

GDS templates are initially upright; small rotations are tested for direct anchoring. Overlap descriptors normalize orientation, but arbitrary rotations of real photographs have not been validated. Heavy occlusion, perspective distortion, significant optical distortion, no shared texture, and disconnected image groups remain outside the validated scope. Repeated grids can create false matches; descriptor filtering, spatial spread, and gradient checks reduce that risk but do not eliminate it. Relative errors accumulate along long registration chains. The mosaic deliberately preserves close-up visibility and exposure/color differences rather than blending away the image boundaries. The design contains the recognition reference, so changing font shapes can work if grouping and marker association still meet these assumptions; this is not a universal OCR solution.

Generated `output/` and `output_multimag*/` files are ignored by Git. Inputs and reproducible scripts are tracked. The repository remote is configured; publishing depends on network access and GitHub authentication in the environment running `git push`.

# QuantifiableCracks Annotation Tool

A desktop tool for semi-automatic, pixel-level crack-mask annotation in calibrated images. The annotator corrects lens distortion, maps an ArUco reference marker to metric coordinates, extracts a fixed-size physical region of interest (ROI), proposes a crack mask, and lets the operator correct it with brush tools.

[Chinese documentation / 中文说明](README_zh-CN.md)

## Features

- Folder-selection window when `annotate_crack_masks.py` is run without arguments.
- Selectable image folder, camera-calibration file, ArUco-corner file, and output folder.
- Fixed metric ROI, independent of image distance and in-plane perspective under the coplanar-surface assumption.
- Automatic dark-ridge mask proposal after the user clicks the target crack centre.
- Interactive add, erase, undo, zoom, ROI resize, and centre reselection controls.
- Automatic progress saving after every image and safe resumption.
- ROI masks, full-image masks, overlays, and geometric transformation metadata.
- Unicode-safe image paths on Windows.

## Scope and assumptions

This tool is designed for approximately planar crack surfaces photographed together with a square ArUco marker. The marker and target crack should be approximately coplanar. The same camera/lens configuration used for intrinsic calibration must be retained during image acquisition.

The automatic proposal is an annotation aid, not an unattended ground-truth generator. Every saved mask should be checked and corrected by a human annotator.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/levyw3847-arch/QuantifiableCracks-Annotation-Tool.git
cd QuantifiableCracks-Annotation-Tool
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python annotate_crack_masks.py
```

Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python annotate_crack_masks.py
```

`tkinter` is included with standard Windows Python distributions. On some Linux distributions it must be installed through the system package manager.

## Required inputs

The annotation stage uses four inputs:

1. A folder containing the source crack images.
2. A camera-calibration `.npz` file containing `camera_matrix`, `distortion_coefficients`, and `image_size`.
3. An ArUco-corner `.json` file keyed by the source image filenames.
4. An output folder.

The calibration and ArUco files must correspond to the selected images. An example corner file is provided in [`examples/aruco_corners.example.json`](examples/aruco_corners.example.json).

## Recommended workflow

### 1. Calibrate the fixed camera/lens configuration

Capture at least 10 checkerboard views. Twenty to thirty views distributed across the frame are preferable.

```bash
python calibrate_camera.py \
  --images-dir path/to/checkerboard_images \
  --output-dir camera_calibration \
  --corners-x 9 \
  --corners-y 6 \
  --square-size-mm 19
```

The command saves `camera_calibration.npz`, a JSON summary, an OpenCV YAML file, reprojection errors, and corner-detection previews.

### 2. Detect or manually mark the ArUco reference

The supplied utility detects marker ID 0 from `DICT_4X4_50`. Failed detections can be corrected manually.

```bash
python annotate_aruco.py \
  --images-dir path/to/crack_images \
  --calibration camera_calibration/camera_calibration.npz \
  --output quantification_results/aruco_corners.json \
  --marker-size-mm 31 \
  --images-per-group 3
```

Set `--images-per-group 1` when every image is an independent specimen. Files are grouped in filename-sorted order.

### 3. Start crack-mask annotation

Double-click `run_annotator.bat` on Windows, or run:

```bash
python annotate_crack_masks.py
```

The setup window lets you choose all input and output paths. It also exposes the physical ROI size and output resolution in pixels/mm.

For a reproducible command-line run:

```bash
python annotate_crack_masks.py \
  --images-dir path/to/crack_images \
  --calibration camera_calibration/camera_calibration.npz \
  --corners quantification_results/aruco_corners.json \
  --output-dir annotations \
  --roi-size-mm 80 \
  --pixels-per-mm 10
```

Use `--select` to show the setup window while retaining command-line values as defaults. Use `--retry-skipped` to revisit skipped images and `--dry-run` to validate inputs without opening the editor.

## Annotation controls

### Full-image view

| Control | Action |
|---|---|
| Left click | Select the centre of the target crack |
| `E` / `D` | Expand / reduce the metric ROI |
| `N` | Skip the current image |
| `Esc` or `Q` | Save progress and quit |

### ROI annotation view

| Control | Action |
|---|---|
| Left drag | Add crack pixels |
| Right drag | Erase pixels |
| Mouse wheel | Zoom around the cursor |
| Middle drag | Pan |
| `A` | Restore the automatic proposal |
| `C` | Clear the mask |
| `Z` | Undo |
| `[` / `]` | Reduce / increase brush size |
| `E` / `D` | Expand / reduce the ROI while preserving the mask |
| `B` | Return to the full image and reselect the crack centre |
| `Enter` | Save the annotation and continue |

## Output structure

```text
annotations/
|-- annotation_progress.json
|-- roi_images/
|-- roi_masks/
|-- roi_overlays/
|-- full_masks/
`-- full_overlays/
```

Masks use `0` for background and `255` for crack pixels. `annotation_progress.json` records the source folder, calibration files, metric scale, ROI transform, completion state, and output paths. If an output folder already contains progress from another source-image folder, the program stops instead of mixing the two datasets.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Data policy

No source images, annotations, camera calibration files, or trained models are included in this repository. Users are responsible for image-use permissions and for reviewing all generated annotations.

## License

This software is released under the MIT License. See [LICENSE](LICENSE).

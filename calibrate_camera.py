import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate a camera from checkerboard photographs."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing checkerboard photographs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("camera_calibration"),
        help="Directory for calibration files and corner previews.",
    )
    parser.add_argument(
        "--corners-x",
        type=int,
        default=9,
        help="Number of checkerboard inner corners horizontally.",
    )
    parser.add_argument(
        "--corners-y",
        type=int,
        default=6,
        help="Number of checkerboard inner corners vertically.",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=20.0,
        help="Measured checkerboard square side length in millimetres.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for images.",
    )
    return parser.parse_args()


def load_image(path):
    # Pillow handles Unicode paths and applies phone EXIF orientation reliably.
    with Image.open(path) as pil_image:
        rgb = ImageOps.exif_transpose(pil_image).convert("RGB")
        return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Could not encode preview image: {path}")
    encoded.tofile(path)


def find_images(directory, recursive):
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def detect_corners(gray, pattern_size):
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
        return found, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if found:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            50,
            1e-4,
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def calculate_view_errors(object_points, image_points, rvecs, tvecs, matrix, distortion):
    errors = []
    for object_point, image_point, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            object_point, rvec, tvec, matrix, distortion
        )
        residual = image_point.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
    return errors


def write_yaml(path, matrix, distortion, image_size, rms):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise RuntimeError(f"Could not write OpenCV YAML file: {path}")
    storage.write("camera_matrix", matrix)
    storage.write("distortion_coefficients", distortion)
    storage.write("image_width", image_size[0])
    storage.write("image_height", image_size[1])
    storage.write("rms_reprojection_error_px", rms)
    storage.release()


def main():
    args = parse_args()
    if args.square_size_mm <= 0:
        raise ValueError("--square-size-mm must be positive.")
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.images_dir}")

    image_paths = find_images(args.images_dir, args.recursive)
    if not image_paths:
        raise FileNotFoundError(
            f"No supported calibration images were found in {args.images_dir}"
        )

    pattern_size = (args.corners_x, args.corners_y)
    object_template = np.zeros((args.corners_x * args.corners_y, 3), np.float32)
    object_template[:, :2] = np.mgrid[
        0 : args.corners_x, 0 : args.corners_y
    ].T.reshape(-1, 2)
    object_template *= args.square_size_mm

    object_points = []
    image_points = []
    accepted_paths = []
    rejected_paths = []
    expected_size = None
    preview_dir = args.output_dir / "corner_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        image = load_image(path)
        image_size = (image.shape[1], image.shape[0])
        if expected_size is None:
            expected_size = image_size
        if image_size != expected_size:
            raise ValueError(
                "All calibration images must have the same resolution and orientation. "
                f"Expected {expected_size}, but {path.name} is {image_size}."
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = detect_corners(gray, pattern_size)
        preview = image.copy()
        if found:
            object_points.append(object_template.copy())
            image_points.append(corners.astype(np.float32))
            accepted_paths.append(path)
            cv2.drawChessboardCorners(preview, pattern_size, corners, True)
            preview_name = f"accepted_{path.stem}.jpg"
        else:
            rejected_paths.append(path)
            cv2.putText(
                preview,
                "CHECKERBOARD NOT DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            preview_name = f"rejected_{path.stem}.jpg"
        save_image(preview_dir / preview_name, preview)

    if len(accepted_paths) < 10:
        raise RuntimeError(
            f"Only {len(accepted_paths)} valid views were detected. "
            "Use at least 10; 20-30 well-distributed views are recommended."
        )

    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        expected_size,
        None,
        None,
    )
    view_errors = calculate_view_errors(
        object_points, image_points, rvecs, tvecs, matrix, distortion
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "camera_calibration.npz",
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        image_size=np.asarray(expected_size, dtype=np.int32),
        pattern_size=np.asarray(pattern_size, dtype=np.int32),
        square_size_mm=np.asarray(args.square_size_mm, dtype=np.float64),
        rms_reprojection_error_px=np.asarray(rms, dtype=np.float64),
    )
    write_yaml(
        args.output_dir / "camera_calibration.yaml",
        matrix,
        distortion,
        expected_size,
        rms,
    )

    summary = {
        "image_size": {"width": expected_size[0], "height": expected_size[1]},
        "pattern_inner_corners": {
            "horizontal": args.corners_x,
            "vertical": args.corners_y,
        },
        "square_size_mm": args.square_size_mm,
        "accepted_image_count": len(accepted_paths),
        "rejected_image_count": len(rejected_paths),
        "rms_reprojection_error_px": float(rms),
        "mean_view_rmse_px": float(np.mean(view_errors)),
        "max_view_rmse_px": float(np.max(view_errors)),
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "quality_note": (
            "Prefer RMS below 0.5 px; inspect images above 1.0 px and repeat "
            "calibration if errors are spatially concentrated."
        ),
    }
    with (args.output_dir / "camera_calibration.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    with (args.output_dir / "view_reprojection_errors.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image", "reprojection_rmse_px"])
        for path, error in zip(accepted_paths, view_errors):
            writer.writerow([str(path), f"{error:.6f}"])

    print(f"Accepted views: {len(accepted_paths)} / {len(image_paths)}")
    print(f"Image size: {expected_size[0]} x {expected_size[1]}")
    print(f"RMS reprojection error: {rms:.4f} px")
    print("Camera matrix K:\n", matrix)
    print("Distortion coefficients:\n", distortion.reshape(-1))
    print(f"Saved calibration files to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

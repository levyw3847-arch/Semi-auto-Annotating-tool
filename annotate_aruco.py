import argparse
import json
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk


DEFAULT_CALIBRATION = Path("camera_calibration/camera_calibration.npz")
DEFAULT_OUTPUT = Path("quantification_results/aruco_corners.json")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatically detect ArUco ID 0 and manually annotate failures."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--marker-size-mm", type=float, default=31.0)
    parser.add_argument(
        "--images-per-group",
        type=int,
        default=3,
        help="Consecutive images assigned to one specimen group (default: 3).",
    )
    parser.add_argument("--retry-skipped", action="store_true")
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Record automatic detections and mark failures as skipped without opening the GUI.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_raw_image(path):
    with Image.open(path) as image:
        orientation = int(image.getexif().get(274, 1))
        rgb = np.array(image.convert("RGB"), copy=True)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), orientation


def apply_orientation(image, orientation):
    if orientation == 1:
        return image
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported EXIF orientation: {orientation}")


def load_records(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("images", {})


def save_records(path, records, marker_size_mm):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aruco_dictionary": "DICT_4X4_50",
        "marker_id": 0,
        "marker_size_mm": marker_size_mm,
        "corner_order": "top-left, top-right, bottom-right, bottom-left",
        "coordinate_system": "undistorted_then_exif_oriented_image",
        "images": records,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def create_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect_marker(image, detector):
    height, width = image.shape[:2]
    scale = min(1.0, 2000.0 / max(width, height))
    if scale < 1.0:
        detection_image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    else:
        detection_image = image
    gray = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or 0 not in ids.reshape(-1):
        return None
    index = list(ids.reshape(-1)).index(0)
    return (corners[index].reshape(4, 2) / scale).astype(np.float32)


class ManualAnnotator:
    def __init__(self, window_name="Manual ArUco annotation"):
        self.window_name = window_name
        self.image = None
        self.filename = ""
        self.mode = "overview"
        self.center = None
        self.crop_bounds = None
        self.points = []
        self.display_scale = 1.0
        self.display_image = None
        self.finished_action = None

    @staticmethod
    def add_banner(image, lines):
        output = image.copy()
        line_height = 31
        banner_height = 18 + line_height * len(lines)
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (output.shape[1], banner_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, output, 0.28, 0, output)
        for index, line in enumerate(lines):
            cv2.putText(
                output,
                line,
                (16, 29 + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return output

    def render_overview(self):
        max_width, max_height = 1500, 900
        height, width = self.image.shape[:2]
        self.display_scale = min(max_width / width, max_height / height, 1.0)
        display = cv2.resize(
            self.image,
            (round(width * self.display_scale), round(height * self.display_scale)),
            interpolation=cv2.INTER_AREA,
        )
        self.display_image = self.add_banner(
            display,
            [
                f"{self.filename} | Click the CENTER of the black ArUco square to zoom",
                "N: no complete marker / skip    Q or Esc: save progress and quit",
            ],
        )

    def render_zoom(self):
        x0, y0, x1, y1 = self.crop_bounds
        crop = self.image[y0:y1, x0:x1]
        max_width, max_height = 1200, 900
        height, width = crop.shape[:2]
        self.display_scale = min(max_width / width, max_height / height, 2.5)
        display = cv2.resize(
            crop,
            (round(width * self.display_scale), round(height * self.display_scale)),
            interpolation=cv2.INTER_CUBIC if self.display_scale > 1 else cv2.INTER_AREA,
        )
        for index, point in enumerate(self.points):
            local_x = round((point[0] - x0) * self.display_scale)
            local_y = round((point[1] - y0) * self.display_scale)
            cv2.circle(display, (local_x, local_y), 7, (0, 255, 255), -1)
            cv2.putText(
                display,
                str(index + 1),
                (local_x + 9, local_y - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if len(self.points) >= 2:
            local_points = np.array(
                [
                    [
                        round((point[0] - x0) * self.display_scale),
                        round((point[1] - y0) * self.display_scale),
                    ]
                    for point in self.points
                ],
                dtype=np.int32,
            )
            cv2.polylines(display, [local_points], False, (0, 255, 255), 2)
        if len(self.points) == 4:
            cv2.line(
                display,
                tuple(local_points[-1]),
                tuple(local_points[0]),
                (0, 255, 255),
                2,
            )
        next_corner = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
        prompt = (
            f"Click {next_corner[len(self.points)]} outer black corner"
            if len(self.points) < 4
            else "Four corners ready - press Enter to save"
        )
        self.display_image = self.add_banner(
            display,
            [
                f"{self.filename} | {prompt}",
                "Right click/U: undo    R: reset    B: back to overview    N: skip",
            ],
        )

    def mouse_callback(self, event, x, y, _flags, _parameter):
        banner_height = 80
        if y < banner_height:
            return
        if event == cv2.EVENT_RBUTTONDOWN and self.mode == "zoom":
            if self.points:
                self.points.pop()
                self.render_zoom()
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.mode == "overview":
            full_x = x / self.display_scale
            full_y = y / self.display_scale
            height, width = self.image.shape[:2]
            radius = 700
            x0 = max(0, round(full_x - radius))
            y0 = max(0, round(full_y - radius))
            x1 = min(width, round(full_x + radius))
            y1 = min(height, round(full_y + radius))
            self.crop_bounds = (x0, y0, x1, y1)
            self.points = []
            self.mode = "zoom"
            self.render_zoom()
        elif len(self.points) < 4:
            x0, y0, _, _ = self.crop_bounds
            full_x = x0 + x / self.display_scale
            full_y = y0 + y / self.display_scale
            self.points.append([float(full_x), float(full_y)])
            self.render_zoom()

    def annotate(self, image, filename):
        self.image = image
        self.filename = filename
        self.mode = "overview"
        self.center = None
        self.crop_bounds = None
        self.points = []
        self.finished_action = None
        self.render_overview()

        root = tk.Tk()
        root.title(self.window_name)
        root.configure(background="#202020")
        canvas = tk.Canvas(root, highlightthickness=0, background="#202020")
        canvas.pack(fill=tk.BOTH, expand=True)
        result = {"action": "quit", "corners": None}
        photo_holder = {"image": None}

        def refresh():
            rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            photo_holder["image"] = ImageTk.PhotoImage(pil_image)
            canvas.configure(width=pil_image.width, height=pil_image.height)
            canvas.delete("all")
            canvas.create_image(0, 0, image=photo_holder["image"], anchor=tk.NW)
            root.geometry(f"{pil_image.width}x{pil_image.height}")
            root.update_idletasks()

        def left_click(event):
            self.mouse_callback(cv2.EVENT_LBUTTONDOWN, event.x, event.y, 0, None)
            refresh()

        def right_click(event):
            self.mouse_callback(cv2.EVENT_RBUTTONDOWN, event.x, event.y, 0, None)
            refresh()

        def finish(action, corners=None):
            result["action"] = action
            result["corners"] = corners
            root.destroy()

        def key_press(event):
            key = event.keysym.lower()
            if key in ("escape", "q"):
                finish("quit")
            elif key == "n":
                finish("skip")
            elif key == "b" and self.mode == "zoom":
                self.mode = "overview"
                self.points = []
                self.render_overview()
                refresh()
            elif key == "r":
                self.points = []
                if self.mode == "zoom":
                    self.render_zoom()
                else:
                    self.render_overview()
                refresh()
            elif key == "u" and self.mode == "zoom" and self.points:
                self.points.pop()
                self.render_zoom()
                refresh()
            elif key in ("return", "kp_enter") and len(self.points) == 4:
                finish("save", np.asarray(self.points, dtype=np.float32))

        canvas.bind("<Button-1>", left_click)
        canvas.bind("<Button-3>", right_click)
        root.bind("<Key>", key_press)
        root.protocol("WM_DELETE_WINDOW", lambda: finish("quit"))
        refresh()
        root.focus_force()
        root.mainloop()
        return result["action"], result["corners"]


def main():
    args = parse_args()
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.images_dir}")
    if not args.calibration.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {args.calibration}")
    if args.marker_size_mm <= 0:
        raise ValueError("Marker size must be positive.")
    if args.images_per_group <= 0:
        raise ValueError("Images per group must be positive.")
    image_paths = sorted(
        path
        for path in args.images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    calibration = np.load(args.calibration)
    camera_matrix = calibration["camera_matrix"]
    distortion = calibration["distortion_coefficients"]
    expected_size = tuple(int(value) for value in calibration["image_size"])
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        camera_matrix,
        expected_size,
        cv2.CV_32FC1,
    )
    records = load_records(args.output)
    print(f"Images: {len(image_paths)}")
    print(f"Existing records: {len(records)}")
    print(f"Marker size: {args.marker_size_mm:.3f} mm")
    if args.dry_run:
        print("Dry run completed; no GUI opened.")
        return

    detector = create_detector()
    annotator = ManualAnnotator()
    try:
        for image_index, path in enumerate(image_paths):
            existing = records.get(path.name)
            if existing and not (
                args.retry_skipped and existing.get("source") == "skipped"
            ):
                continue

            raw, orientation = read_raw_image(path)
            raw_size = (raw.shape[1], raw.shape[0])
            if raw_size != expected_size:
                raise ValueError(
                    f"{path.name}: raw size {raw_size} does not match calibration "
                    f"size {expected_size}."
                )
            undistorted = cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR)
            processed = apply_orientation(undistorted, orientation)
            corners = detect_marker(processed, detector)

            common = {
                "group_id": f"specimen_{image_index // args.images_per_group + 1:03d}",
                "image_in_group": image_index % args.images_per_group + 1,
                "marker_id": 0,
                "marker_size_mm": args.marker_size_mm,
                "exif_orientation": orientation,
                "processed_image_size": [processed.shape[1], processed.shape[0]],
            }
            if corners is not None:
                records[path.name] = {
                    **common,
                    "source": "automatic",
                    "corners_px": corners.tolist(),
                }
                save_records(args.output, records, args.marker_size_mm)
                print(f"AUTO {path.name}")
                continue

            if args.auto_only:
                records[path.name] = {**common, "source": "skipped", "corners_px": None}
                save_records(args.output, records, args.marker_size_mm)
                print(f"AUTO_FAIL {path.name}")
                continue

            action, manual_corners = annotator.annotate(processed, path.name)
            if action == "quit":
                print("Stopped by user; progress was preserved.")
                break
            if action == "skip":
                records[path.name] = {**common, "source": "skipped", "corners_px": None}
                print(f"SKIP {path.name}")
            else:
                records[path.name] = {
                    **common,
                    "source": "manual",
                    "corners_px": manual_corners.tolist(),
                }
                print(f"MANUAL {path.name}")
            save_records(args.output, records, args.marker_size_mm)
    finally:
        pass

    automatic = sum(record.get("source") == "automatic" for record in records.values())
    manual = sum(record.get("source") == "manual" for record in records.values())
    skipped = sum(record.get("source") == "skipped" for record in records.values())
    print(f"Saved: automatic={automatic}, manual={manual}, skipped={skipped}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()

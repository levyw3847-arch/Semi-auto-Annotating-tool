import argparse
import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


DEFAULT_CALIBRATION = Path("camera_calibration/camera_calibration.npz")
DEFAULT_CORNERS = Path("quantification_results/aruco_corners.json")
DEFAULT_OUTPUT = Path("quantification_annotations")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrated brush annotation for thin crack segmentation masks."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Image directory. If omitted, a configuration window is opened.",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--corners", type=Path, default=DEFAULT_CORNERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roi-size-mm", type=float, default=80.0)
    parser.add_argument("--pixels-per-mm", type=float, default=10.0)
    parser.add_argument("--retry-skipped", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--select",
        action="store_true",
        help="Open the configuration window even when command-line paths are provided.",
    )
    return parser.parse_args()


def select_configuration(args):
    """Collect paths and annotation parameters before opening the image editor."""
    root = tk.Tk()
    root.title("QuantifiableCracks annotation setup")
    root.resizable(True, False)

    values = {
        "images_dir": tk.StringVar(value=str(args.images_dir or "")),
        "calibration": tk.StringVar(value=str(args.calibration)),
        "corners": tk.StringVar(value=str(args.corners)),
        "output_dir": tk.StringVar(value=str(args.output_dir)),
        "roi_size_mm": tk.StringVar(value=f"{args.roi_size_mm:g}"),
        "pixels_per_mm": tk.StringVar(value=f"{args.pixels_per_mm:g}"),
        "retry_skipped": tk.BooleanVar(value=bool(args.retry_skipped)),
    }
    selected = {}

    container = ttk.Frame(root, padding=16)
    container.grid(row=0, column=0, sticky="nsew")
    container.columnconfigure(1, weight=1)
    root.columnconfigure(0, weight=1)

    def initial_directory(value):
        candidate = Path(value.get()).expanduser() if value.get().strip() else Path.cwd()
        if candidate.is_file():
            candidate = candidate.parent
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return str(candidate)

    def choose_directory(variable, title):
        chosen = filedialog.askdirectory(
            parent=root,
            title=title,
            initialdir=initial_directory(variable),
        )
        if chosen:
            variable.set(chosen)

    def choose_file(variable, title, filetypes):
        chosen = filedialog.askopenfilename(
            parent=root,
            title=title,
            initialdir=initial_directory(variable),
            filetypes=filetypes,
        )
        if chosen:
            variable.set(chosen)

    path_rows = [
        (
            "Image folder",
            "images_dir",
            lambda: choose_directory(values["images_dir"], "Select image folder"),
        ),
        (
            "Camera calibration (.npz)",
            "calibration",
            lambda: choose_file(
                values["calibration"],
                "Select camera calibration",
                [("NumPy calibration", "*.npz"), ("All files", "*.*")],
            ),
        ),
        (
            "ArUco corners (.json)",
            "corners",
            lambda: choose_file(
                values["corners"],
                "Select ArUco corner annotations",
                [("JSON files", "*.json"), ("All files", "*.*")],
            ),
        ),
        (
            "Output folder",
            "output_dir",
            lambda: choose_directory(values["output_dir"], "Select output folder"),
        ),
    ]
    for row, (label, key, command) in enumerate(path_rows):
        ttk.Label(container, text=label).grid(row=row, column=0, padx=(0, 10), pady=6, sticky="w")
        ttk.Entry(container, textvariable=values[key], width=72).grid(
            row=row, column=1, pady=6, sticky="ew"
        )
        ttk.Button(container, text="Browse...", command=command).grid(
            row=row, column=2, padx=(10, 0), pady=6
        )

    parameter_row = len(path_rows)
    ttk.Label(container, text="ROI size (mm)").grid(
        row=parameter_row, column=0, padx=(0, 10), pady=(14, 6), sticky="w"
    )
    ttk.Entry(container, textvariable=values["roi_size_mm"], width=14).grid(
        row=parameter_row, column=1, pady=(14, 6), sticky="w"
    )
    ttk.Label(container, text="Pixels per mm").grid(
        row=parameter_row + 1, column=0, padx=(0, 10), pady=6, sticky="w"
    )
    ttk.Entry(container, textvariable=values["pixels_per_mm"], width=14).grid(
        row=parameter_row + 1, column=1, pady=6, sticky="w"
    )
    ttk.Checkbutton(
        container,
        text="Retry images previously marked as skipped",
        variable=values["retry_skipped"],
    ).grid(row=parameter_row + 2, column=1, pady=6, sticky="w")

    hint = (
        "The calibration and ArUco-corner files must correspond to the selected images. "
        "Progress is saved after every image."
    )
    ttk.Label(container, text=hint, foreground="#555555", wraplength=650).grid(
        row=parameter_row + 3, column=0, columnspan=3, pady=(10, 4), sticky="w"
    )

    def start():
        images_dir = Path(values["images_dir"].get().strip()).expanduser()
        calibration = Path(values["calibration"].get().strip()).expanduser()
        corners = Path(values["corners"].get().strip()).expanduser()
        output_dir = Path(values["output_dir"].get().strip()).expanduser()
        if not images_dir.is_dir():
            messagebox.showerror("Invalid image folder", "Select an existing image folder.", parent=root)
            return
        if not calibration.is_file():
            messagebox.showerror("Invalid calibration", "Select an existing .npz calibration file.", parent=root)
            return
        if not corners.is_file():
            messagebox.showerror("Invalid ArUco data", "Select an existing ArUco-corners JSON file.", parent=root)
            return
        if not str(output_dir):
            messagebox.showerror("Invalid output folder", "Select or enter an output folder.", parent=root)
            return
        try:
            roi_size_mm = float(values["roi_size_mm"].get())
            pixels_per_mm = float(values["pixels_per_mm"].get())
        except ValueError:
            messagebox.showerror("Invalid parameters", "ROI size and pixels/mm must be numbers.", parent=root)
            return
        if roi_size_mm <= 0 or pixels_per_mm <= 0:
            messagebox.showerror("Invalid parameters", "ROI size and pixels/mm must be positive.", parent=root)
            return
        selected.update(
            images_dir=images_dir,
            calibration=calibration,
            corners=corners,
            output_dir=output_dir,
            roi_size_mm=roi_size_mm,
            pixels_per_mm=pixels_per_mm,
            retry_skipped=bool(values["retry_skipped"].get()),
        )
        root.destroy()

    buttons = ttk.Frame(container)
    buttons.grid(row=parameter_row + 4, column=0, columnspan=3, pady=(14, 0), sticky="e")
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side=tk.RIGHT, padx=(8, 0))
    ttk.Button(buttons, text="Start annotation", command=start).pack(side=tk.RIGHT)
    root.bind("<Return>", lambda _event: start())
    root.bind("<Escape>", lambda _event: root.destroy())
    root.update_idletasks()
    width = max(820, root.winfo_reqwidth())
    height = root.winfo_reqheight()
    x = max(0, (root.winfo_screenwidth() - width) // 2)
    y = max(0, (root.winfo_screenheight() - height) // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.mainloop()

    if not selected:
        return None
    for key, value in selected.items():
        setattr(args, key, value)
    return args


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


def image_to_metric_homography(corners, marker_size_mm):
    destination = np.array(
        [
            [0.0, 0.0],
            [marker_size_mm, 0.0],
            [marker_size_mm, marker_size_mm],
            [0.0, marker_size_mm],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(np.asarray(corners, np.float32), destination)


def imwrite(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".png", image)
    if not success:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(path)


def load_progress(path):
    if not path.exists():
        return {"images": {}}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_progress(path, records, configuration):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**configuration, "images": records}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def automatic_proposal(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )
    dark_ridge = cv2.subtract(background, gray)
    centre_y, centre_x = np.array(gray.shape) // 2
    seed_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(seed_mask, (int(centre_x), int(centre_y)), 14, 1, thickness=-1)
    seed_values = dark_ridge[seed_mask > 0]
    seed_strength = float(np.percentile(seed_values, 75)) if seed_values.size else 0.0
    threshold = max(8.0, min(70.0, seed_strength + 1.0))
    candidate = (dark_ridge >= threshold).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    touching = np.unique(labels[seed_mask > 0])
    touching = [int(value) for value in touching if value != 0 and stats[value, 4] >= 8]
    if not touching and count > 1:
        centre = np.array([centre_x, centre_y], dtype=np.float64)
        distances = [
            (float(np.linalg.norm(centroids[index] - centre)), index)
            for index in range(1, count)
            if stats[index, 4] >= 8
        ]
        if distances and min(distances)[0] <= 80:
            touching = [min(distances)[1]]
    output = np.isin(labels, touching).astype(np.uint8) * 255
    if int((output > 0).sum()) > int(0.10 * output.size):
        output.fill(0)
    return output


def overlay_mask(image, mask):
    output = image.astype(np.float32).copy()
    selected = mask > 0
    output[selected] = 0.35 * output[selected] + 0.65 * np.array([0, 0, 255])
    return output.astype(np.uint8)


def remove_tiny_components(mask, minimum_area=8):
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area:
            output[labels == index] = 255
    return output


class CrackMaskAnnotator:
    def __init__(self, roi_size_mm, pixels_per_mm):
        self.roi_size_mm = float(roi_size_mm)
        self.pixels_per_mm = float(pixels_per_mm)
        self.roi_size_px = int(round(self.roi_size_mm * self.pixels_per_mm))
        self.image = None
        self.filename = ""
        self.marker_corners = None
        self.image_to_metric = None
        self.centre_px = None
        self.centre_mm = None
        self.image_to_roi = None
        self.roi = None
        self.mask = None
        self.proposal = None
        self.mode = "overview"
        self.brush_radius = max(2, int(round(0.35 * self.pixels_per_mm)))
        self.zoom = 1.0
        self.overview_scale = 1.0
        self.undo_stack = []
        self.stroke_point = None
        self.action = "quit"
        self.root = None
        self.canvas = None
        self.photo = None
        self.image_item = None
        self.status = None

    def rebuild_roi(self, preserve_mask=False):
        old_mask = self.mask.copy() if preserve_mask and self.mask is not None else None
        old_transform = (
            self.image_to_roi.copy()
            if preserve_mask and self.image_to_roi is not None
            else None
        )
        self.roi_size_px = int(round(self.roi_size_mm * self.pixels_per_mm))
        half = self.roi_size_mm / 2.0
        minimum = self.centre_mm - half
        metric_to_roi = np.array(
            [
                [self.pixels_per_mm, 0.0, -minimum[0] * self.pixels_per_mm],
                [0.0, self.pixels_per_mm, -minimum[1] * self.pixels_per_mm],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.image_to_roi = metric_to_roi @ self.image_to_metric
        self.roi = cv2.warpPerspective(
            self.image,
            self.image_to_roi,
            (self.roi_size_px, self.roi_size_px),
            flags=cv2.INTER_LINEAR,
        )
        self.proposal = automatic_proposal(self.roi)
        if old_mask is not None and old_transform is not None:
            old_to_new = self.image_to_roi @ np.linalg.inv(old_transform)
            self.mask = cv2.warpPerspective(
                old_mask,
                old_to_new,
                (self.roi_size_px, self.roi_size_px),
                flags=cv2.INTER_NEAREST,
            )
        else:
            self.mask = self.proposal.copy()
        self.undo_stack = []
        self.mode = "annotate"
        self.zoom = 1.0
        self.render_annotation()

    def prepare_roi(self, x, y):
        self.centre_px = np.array([float(x), float(y)], dtype=np.float64)
        transformed = cv2.perspectiveTransform(
            self.centre_px.astype(np.float32).reshape(1, 1, 2), self.image_to_metric
        )[0, 0]
        self.centre_mm = transformed.astype(np.float64)
        self.rebuild_roi(preserve_mask=False)

    def resize_roi(self, delta_mm):
        new_size = max(40.0, min(200.0, self.roi_size_mm + delta_mm))
        if abs(new_size - self.roi_size_mm) < 1e-6:
            return
        self.roi_size_mm = new_size
        if self.mode == "annotate" and self.centre_mm is not None:
            self.rebuild_roi(preserve_mask=True)
        else:
            self.roi_size_px = int(round(self.roi_size_mm * self.pixels_per_mm))
            self.render_overview()

    def set_status(self, message):
        self.status.configure(text=message)

    def render_overview(self):
        height, width = self.image.shape[:2]
        max_width, max_height = 1450, 820
        self.overview_scale = min(max_width / width, max_height / height, 1.0)
        display = cv2.resize(
            self.image,
            (round(width * self.overview_scale), round(height * self.overview_scale)),
            interpolation=cv2.INTER_AREA,
        )
        scaled_corners = np.round(self.marker_corners * self.overview_scale).astype(np.int32)
        cv2.polylines(display, [scaled_corners], True, (0, 255, 255), 3)
        self.show_image(display, zoom=1.0)
        self.set_status(
            f"{self.filename} | ROI {self.roi_size_mm:.0f} mm | Click the centre of the target crack | "
            "E: expand ROI  D: shrink ROI  N: skip  Esc: save and quit"
        )

    def render_annotation(self):
        display = overlay_mask(self.roi, self.mask)
        self.show_image(display, zoom=self.zoom)
        self.set_status(
            f"{self.filename} | ROI {self.roi_size_mm:.0f} mm | brush {2*self.brush_radius/self.pixels_per_mm:.1f} mm diameter | "
            "Left: add  Right: erase  E/D: resize ROI  A: auto  C: clear  Z: undo  [ ]: brush  Wheel: zoom  Enter: save  B: reselect"
        )

    def move_view_to(self, left, top, full_width, full_height):
        viewport_width = self.canvas.winfo_width()
        viewport_height = self.canvas.winfo_height()
        maximum_left = max(0.0, full_width - viewport_width)
        maximum_top = max(0.0, full_height - viewport_height)
        left = max(0.0, min(maximum_left, left))
        top = max(0.0, min(maximum_top, top))
        self.canvas.xview_moveto(left / max(1.0, full_width))
        self.canvas.yview_moveto(top / max(1.0, full_height))

    def show_image(self, image, zoom):
        old_x = self.canvas.canvasx(self.canvas.winfo_width() / 2.0)
        old_y = self.canvas.canvasy(self.canvas.winfo_height() / 2.0)
        old_zoom = max(getattr(self, "zoom", 1.0), 1e-6)
        centre_image_x, centre_image_y = old_x / old_zoom, old_y / old_zoom

        if zoom != 1.0:
            image = cv2.resize(
                image,
                None,
                fx=zoom,
                fy=zoom,
                interpolation=cv2.INTER_NEAREST,
            )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        if self.image_item is None:
            self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        else:
            self.canvas.itemconfigure(self.image_item, image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, image.shape[1], image.shape[0]))
        self.canvas.update_idletasks()
        if self.mode == "annotate":
            target_x = centre_image_x * zoom - self.canvas.winfo_width() / 2.0
            target_y = centre_image_y * zoom - self.canvas.winfo_height() / 2.0
            self.move_view_to(target_x, target_y, image.shape[1], image.shape[0])

    def canvas_to_image(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if self.mode == "overview":
            return x / self.overview_scale, y / self.overview_scale
        return x / self.zoom, y / self.zoom

    def begin_stroke(self, event, value):
        if self.mode == "overview":
            if value == 255:
                x, y = self.canvas_to_image(event)
                if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                    self.prepare_roi(x, y)
            return
        self.undo_stack.append(self.mask.copy())
        self.undo_stack = self.undo_stack[-20:]
        self.stroke_point = self.canvas_to_image(event)
        self.paint_to(event, value)

    def paint_to(self, event, value):
        if self.mode != "annotate" or self.stroke_point is None:
            return
        current = self.canvas_to_image(event)
        x0, y0 = (int(round(number)) for number in self.stroke_point)
        x1, y1 = (int(round(number)) for number in current)
        cv2.line(
            self.mask,
            (x0, y0),
            (x1, y1),
            int(value),
            thickness=2 * self.brush_radius + 1,
            lineType=cv2.LINE_8,
        )
        cv2.circle(self.mask, (x1, y1), self.brush_radius, int(value), thickness=-1)
        self.stroke_point = current
        self.render_annotation()

    def end_stroke(self, _event):
        self.stroke_point = None

    def zoom_at_cursor(self, event):
        if self.mode != "annotate":
            return
        image_x, image_y = self.canvas_to_image(event)
        factor = 1.25 if event.delta > 0 else 0.8
        new_zoom = max(0.75, min(4.0, self.zoom * factor))
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        self.zoom = new_zoom
        display = overlay_mask(self.roi, self.mask)
        self.show_image(display, zoom=self.zoom)
        target_x = image_x * self.zoom - event.x
        target_y = image_y * self.zoom - event.y
        full_width = self.roi_size_px * self.zoom
        full_height = self.roi_size_px * self.zoom
        self.move_view_to(target_x, target_y, full_width, full_height)

    def key_press(self, event):
        key = event.keysym.lower()
        if key in ("escape", "q"):
            self.action = "quit"
            self.root.destroy()
        elif key == "n":
            self.action = "skip"
            self.root.destroy()
        elif key in ("return", "kp_enter") and self.mode == "annotate":
            if not self.mask.any():
                self.set_status("Mask is empty. Draw the crack, restore A, or press N to skip.")
                return
            self.action = "save"
            self.root.destroy()
        elif key == "a" and self.mode == "annotate":
            self.undo_stack.append(self.mask.copy())
            self.mask = self.proposal.copy()
            self.render_annotation()
        elif key == "c" and self.mode == "annotate":
            self.undo_stack.append(self.mask.copy())
            self.mask.fill(0)
            self.render_annotation()
        elif key == "z" and self.mode == "annotate" and self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.render_annotation()
        elif key == "b" and self.mode == "annotate":
            self.mode = "overview"
            self.image_item = None
            self.canvas.delete("all")
            self.render_overview()
        elif key == "e":
            self.resize_roi(20.0)
        elif key == "d":
            self.resize_roi(-20.0)
        elif key in ("bracketleft", "minus") and self.mode == "annotate":
            self.brush_radius = max(1, self.brush_radius - 1)
            self.render_annotation()
        elif key in ("bracketright", "plus", "equal") and self.mode == "annotate":
            self.brush_radius = min(80, self.brush_radius + 1)
            self.render_annotation()

    def annotate(self, image, filename, marker_corners, marker_size_mm):
        self.image = image
        self.filename = filename
        self.marker_corners = np.asarray(marker_corners, dtype=np.float32)
        self.image_to_metric = image_to_metric_homography(
            self.marker_corners, marker_size_mm
        )
        self.mode = "overview"
        self.action = "quit"
        self.image_item = None

        self.root = tk.Tk()
        self.root.title("Calibrated crack mask annotation")
        self.root.geometry("1450x950")
        self.status = tk.Label(
            self.root,
            text="",
            anchor="w",
            padx=10,
            pady=8,
            bg="#202020",
            fg="white",
            font=("Segoe UI", 11),
        )
        self.status.pack(fill=tk.X)
        frame = tk.Frame(self.root, bg="#202020")
        frame.pack(fill=tk.BOTH, expand=True)
        x_scroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        y_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(
            frame,
            bg="#202020",
            highlightthickness=0,
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set,
        )
        x_scroll.configure(command=self.canvas.xview)
        y_scroll.configure(command=self.canvas.yview)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", lambda event: self.begin_stroke(event, 255))
        self.canvas.bind("<B1-Motion>", lambda event: self.paint_to(event, 255))
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        self.canvas.bind("<ButtonPress-3>", lambda event: self.begin_stroke(event, 0))
        self.canvas.bind("<B3-Motion>", lambda event: self.paint_to(event, 0))
        self.canvas.bind("<ButtonRelease-3>", self.end_stroke)
        self.canvas.bind("<MouseWheel>", self.zoom_at_cursor)
        self.canvas.bind("<ButtonPress-2>", lambda event: self.canvas.scan_mark(event.x, event.y))
        self.canvas.bind(
            "<B2-Motion>", lambda event: self.canvas.scan_dragto(event.x, event.y, gain=1)
        )
        self.root.bind("<Key>", self.key_press)
        self.root.update_idletasks()
        self.render_overview()
        self.root.focus_force()
        self.root.mainloop()
        return {
            "action": self.action,
            "centre_px": self.centre_px,
            "centre_mm": self.centre_mm,
            "image_to_roi": self.image_to_roi,
            "roi": self.roi,
            "mask": self.mask,
            "roi_size_mm": self.roi_size_mm,
            "pixels_per_mm": self.pixels_per_mm,
            "roi_size_px": self.roi_size_px,
        }


def save_annotation(output_dir, image, image_path, result, record, configuration):
    stem = image_path.stem
    roi_image_path = output_dir / "roi_images" / f"{stem}.png"
    roi_mask_path = output_dir / "roi_masks" / f"{stem}_mask.png"
    full_mask_path = output_dir / "full_masks" / f"{stem}_mask.png"
    roi_overlay_path = output_dir / "roi_overlays" / f"{stem}_overlay.png"
    full_overlay_path = output_dir / "full_overlays" / f"{stem}_overlay.jpg"

    roi = result["roi"]
    mask = remove_tiny_components(result["mask"])
    inverse = np.linalg.inv(result["image_to_roi"])
    full_mask = cv2.warpPerspective(
        mask,
        inverse,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    full_mask = (full_mask > 127).astype(np.uint8) * 255
    full_overlay = overlay_mask(image, full_mask)
    cv2.polylines(
        full_overlay,
        [np.round(record["corners_px"]).astype(np.int32)],
        True,
        (0, 255, 255),
        4,
    )
    preview_scale = min(1.0, 1800.0 / max(full_overlay.shape[:2]))
    if preview_scale < 1.0:
        full_overlay = cv2.resize(
            full_overlay,
            None,
            fx=preview_scale,
            fy=preview_scale,
            interpolation=cv2.INTER_AREA,
        )

    imwrite(roi_image_path, roi)
    imwrite(roi_mask_path, mask)
    imwrite(full_mask_path, full_mask)
    imwrite(roi_overlay_path, overlay_mask(roi, mask))
    imwrite(full_overlay_path, full_overlay)
    return {
        "status": "saved",
        "group_id": record["group_id"],
        "image_in_group": record["image_in_group"],
        "centre_px": result["centre_px"].tolist(),
        "centre_mm": result["centre_mm"].tolist(),
        "processed_image_size": [image.shape[1], image.shape[0]],
        "roi_size_mm": result["roi_size_mm"],
        "pixels_per_mm": result["pixels_per_mm"],
        "roi_size_px": result["roi_size_px"],
        "image_to_roi_homography": result["image_to_roi"].tolist(),
        "crack_pixels_roi": int((mask > 0).sum()),
        "roi_image": str(roi_image_path.resolve()),
        "roi_mask": str(roi_mask_path.resolve()),
        "full_mask": str(full_mask_path.resolve()),
        "roi_overlay": str(roi_overlay_path.resolve()),
        "full_overlay": str(full_overlay_path.resolve()),
        "human_verified": True,
    }


def main():
    args = parse_args()
    if args.select or args.images_dir is None:
        args = select_configuration(args)
        if args is None:
            print("Annotation cancelled.", flush=True)
            return
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.images_dir}")
    if not args.calibration.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {args.calibration}")
    if not args.corners.is_file():
        raise FileNotFoundError(f"ArUco-corners file does not exist: {args.corners}")
    if args.roi_size_mm <= 0 or args.pixels_per_mm <= 0:
        raise ValueError("ROI size and pixels/mm must be positive.")

    with args.corners.open("r", encoding="utf-8") as file:
        corner_payload = json.load(file)
    marker_records = corner_payload["images"]
    marker_size_mm = float(corner_payload["marker_size_mm"])
    calibration = np.load(args.calibration)
    camera_matrix = calibration["camera_matrix"]
    distortion = calibration["distortion_coefficients"]
    images = sorted(
        path for path in args.images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No supported images were found in {args.images_dir}")
    progress_path = args.output_dir / "annotation_progress.json"
    progress = load_progress(progress_path)
    records = progress.get("images", {})
    source_images_dir = str(args.images_dir.resolve())
    stored_source = progress.get("source_images_dir")
    if stored_source and Path(stored_source).resolve() != args.images_dir.resolve():
        raise ValueError(
            "The selected output folder already contains progress for another image folder: "
            f"{stored_source}. Select a different output folder."
        )
    configuration = {
        "format_version": 1,
        "source_images_dir": source_images_dir,
        "calibration_file": str(args.calibration.resolve()),
        "aruco_corners_file": str(args.corners.resolve()),
        "coordinate_system": "undistorted_then_exif_oriented_image",
        "marker_size_mm": marker_size_mm,
        "roi_size_mm": float(args.roi_size_mm),
        "pixels_per_mm": float(args.pixels_per_mm),
        "mask_values": {"background": 0, "crack": 255},
        "annotation_rule": "visible crack opening area; exclude shadows, scratches, specimen edges, and marker",
    }
    pending = [
        image
        for image in images
        if image.name in marker_records
        and (
            image.name not in records
            or (args.retry_skipped and records[image.name].get("status") == "skipped")
        )
    ]
    print(f"Images: {len(images)}", flush=True)
    print(f"Images with ArUco records: {sum(image.name in marker_records for image in images)}", flush=True)
    print(f"Completed records: {len(records)}", flush=True)
    print(f"Pending: {len(pending)}", flush=True)
    print(
        f"ROI: {args.roi_size_mm:.1f} mm, {args.pixels_per_mm:.1f} px/mm, "
        f"{round(args.roi_size_mm * args.pixels_per_mm)} px",
        flush=True,
    )
    if args.dry_run:
        print("Dry run completed; no GUI opened.", flush=True)
        return

    annotator = CrackMaskAnnotator(args.roi_size_mm, args.pixels_per_mm)
    for index, image_path in enumerate(pending, start=1):
        raw, orientation = read_raw_image(image_path)
        image = apply_orientation(cv2.undistort(raw, camera_matrix, distortion), orientation)
        marker_record = marker_records[image_path.name]
        result = annotator.annotate(
            image,
            f"[{index}/{len(pending)}] {image_path.name}",
            marker_record["corners_px"],
            marker_size_mm,
        )
        if result["action"] == "quit":
            save_progress(progress_path, records, configuration)
            print("Progress saved; annotation stopped.", flush=True)
            return
        if result["action"] == "skip":
            records[image_path.name] = {
                "status": "skipped",
                "group_id": marker_record["group_id"],
                "image_in_group": marker_record["image_in_group"],
            }
            print(f"SKIPPED {image_path.name}", flush=True)
        else:
            records[image_path.name] = save_annotation(
                args.output_dir,
                image,
                image_path,
                result,
                marker_record,
                configuration,
            )
            print(
                f"SAVED {image_path.name}: {records[image_path.name]['crack_pixels_roi']} crack pixels",
                flush=True,
            )
        save_progress(progress_path, records, configuration)
    saved = sum(record.get("status") == "saved" for record in records.values())
    skipped = sum(record.get("status") == "skipped" for record in records.values())
    print(f"Finished: saved={saved}, skipped={skipped}", flush=True)
    print(f"Output: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

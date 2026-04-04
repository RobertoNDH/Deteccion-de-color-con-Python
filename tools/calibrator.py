import argparse
import os
import sys

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.camera import Camera, CameraError
from src.config_manager import ConfigError, ConfigManager
from src.detector import ColorDetector

_WIN_CTRL   = "Calibrator - Controls"
_WIN_RESULT = "Calibrator - Preview"

_TB = {
    "L_H": "Lower H",
    "L_S": "Lower S",
    "L_V": "Lower V",
    "U_H": "Upper H",
    "U_S": "Upper S",
    "U_V": "Upper V",
}


def _nothing(_):
    pass


def _parse_args():
    p = argparse.ArgumentParser(
        prog="calibrator",
        description="Interactive HSV calibration tool for ColorTracker.",
    )
    p.add_argument("--source", default=None)
    p.add_argument("--config", default="config.json")
    p.add_argument("--color", default=None)
    return p.parse_args()


def _create_trackbars(window: str, lower: list, upper: list):
    cv2.createTrackbar(_TB["L_H"], window, lower[0], 179, _nothing)
    cv2.createTrackbar(_TB["L_S"], window, lower[1], 255, _nothing)
    cv2.createTrackbar(_TB["L_V"], window, lower[2], 255, _nothing)
    cv2.createTrackbar(_TB["U_H"], window, upper[0], 179, _nothing)
    cv2.createTrackbar(_TB["U_S"], window, upper[1], 255, _nothing)
    cv2.createTrackbar(_TB["U_V"], window, upper[2], 255, _nothing)


def _read_trackbars(window: str):
    lh = cv2.getTrackbarPos(_TB["L_H"], window)
    ls = cv2.getTrackbarPos(_TB["L_S"], window)
    lv = cv2.getTrackbarPos(_TB["L_V"], window)
    uh = cv2.getTrackbarPos(_TB["U_H"], window)
    us = cv2.getTrackbarPos(_TB["U_S"], window)
    uv = cv2.getTrackbarPos(_TB["U_V"], window)
    return [lh, ls, lv], [uh, us, uv]


def _set_trackbars(window: str, lower: list, upper: list):
    cv2.setTrackbarPos(_TB["L_H"], window, lower[0])
    cv2.setTrackbarPos(_TB["L_S"], window, lower[1])
    cv2.setTrackbarPos(_TB["L_V"], window, lower[2])
    cv2.setTrackbarPos(_TB["U_H"], window, upper[0])
    cv2.setTrackbarPos(_TB["U_S"], window, upper[1])
    cv2.setTrackbarPos(_TB["U_V"], window, upper[2])


def _apply_mask(hsv: np.ndarray,
                lower: list, upper: list) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array(lower, dtype=np.uint8)
    hi = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ColorDetector._MORPH_KERNEL_OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ColorDetector._MORPH_KERNEL_CLOSE)
    return mask, mask


def _overlay_instructions(frame: np.ndarray, color_name: str,
                            color_list: list[str], saved: bool):
    idx = color_list.index(color_name)
    lines = [
        f"Color: {color_name} ({idx + 1}/{len(color_list)})",
        "c - next color",
        "w - save to config.json",
        "click - pick HSV from pixel",
        "q / ESC - quit",
    ]
    if saved:
        lines.append("Saved!")

    pad, lh = 8, 18
    h = len(lines) * lh + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, frame.shape[0] - h), (260, frame.shape[0]),
                  (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for i, line in enumerate(lines):
        y = frame.shape[0] - h + pad + (i + 1) * lh - 4
        color = (0, 255, 80) if ("Saved!" in line or "Color:" in line) else (200, 200, 200)
        cv2.putText(frame, line, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


class _ClickPicker:
    def __init__(self):
        self.hsv_frame: np.ndarray | None = None
        self.picked: bool = False
        self.lower: list | None = None
        self.upper: list | None = None

    def callback(self, event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv_frame is None:
            return
        px_h, px_s, px_v = self.hsv_frame[y, x]
        margin_h = 15
        margin_sv = 60
        self.lower = [
            max(0,   int(px_h) - margin_h),
            max(0,   int(px_s) - margin_sv),
            max(0,   int(px_v) - margin_sv),
        ]
        self.upper = [
            min(179, int(px_h) + margin_h),
            min(255, int(px_s) + margin_sv),
            min(255, int(px_v) + margin_sv),
        ]
        self.picked = True
        print(f"[PICK] ({x}, {y}) HSV={px_h},{px_s},{px_v} -> lower={self.lower} upper={self.upper}")


def main():
    args = _parse_args()

    try:
        config = ConfigManager(args.config)
    except ConfigError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    all_colors = config.colors
    color_list = list(all_colors.keys())
    if not color_list:
        print("[ERROR] No colors defined in config.")
        sys.exit(1)

    current_color = args.color if args.color in color_list else color_list[0]

    source: str | int = 0
    if args.source is not None:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source

    try:
        cam = Camera(source, config.frame_width, config.frame_height)
        cam.open()
    except CameraError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    cv2.namedWindow(_WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN_CTRL, 400, 250)
    cfg0 = all_colors[current_color]
    _create_trackbars(_WIN_CTRL, cfg0.get("lower", [0, 0, 0]),
                      cfg0.get("upper", [179, 255, 255]))

    cv2.namedWindow(_WIN_RESULT, cv2.WINDOW_NORMAL)
    picker = _ClickPicker()
    cv2.setMouseCallback(_WIN_RESULT, picker.callback)

    saved_flash = 0
    last_frame: np.ndarray | None = None

    print("[INFO] Calibrator started.")
    print(f"[INFO] Current color: {current_color}")
    print("[INFO] Press 'c' to cycle colors, 'w' to save, 'q'/'ESC' to quit.")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord("c"):
            idx = color_list.index(current_color)
            current_color = color_list[(idx + 1) % len(color_list)]
            cfg = all_colors[current_color]
            _set_trackbars(_WIN_CTRL, cfg.get("lower", [0, 0, 0]),
                           cfg.get("upper", [179, 255, 255]))
            picker.picked = False
            print(f"[INFO] Switched to color: {current_color}")

        if key == ord("w"):
            lower, upper = _read_trackbars(_WIN_CTRL)
            config.update_color(current_color, lower, upper)
            all_colors = config.colors
            saved_flash = 60
            print(f"[SAVE] {current_color} lower={lower} upper={upper}")

        frame = cam.read()
        if frame is None:
            if cam.is_image and last_frame is not None:
                frame = last_frame.copy()
            else:
                break

        last_frame = frame.copy()

        if picker.picked:
            _set_trackbars(_WIN_CTRL, picker.lower, picker.upper)
            picker.picked = False

        blurred = cv2.GaussianBlur(frame, (config.blur_kernel_size,
                                           config.blur_kernel_size), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        picker.hsv_frame = hsv

        lower, upper = _read_trackbars(_WIN_CTRL)
        mask, _ = _apply_mask(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview  = np.hstack([frame, mask_3ch, result])

        _overlay_instructions(preview, current_color, color_list,
                               saved=saved_flash > 0)
        if saved_flash > 0:
            saved_flash -= 1

        cv2.imshow(_WIN_RESULT, preview)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()





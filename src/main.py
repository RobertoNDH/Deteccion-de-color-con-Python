import argparse
import os
import sys
import time
import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config_manager import ConfigManager, ConfigError
from src.camera import Camera, CameraError
from src.detector import ColorDetector
from src.tracker import ObjectTracker
from src.visualizer import Visualizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="colortracker",
        description="Real-time color detection and tracking engine.",
    )
    parser.add_argument(
        "--source", default=None,
    )
    parser.add_argument(
        "--config", default="config.json",
    )
    parser.add_argument(
        "--colors", default=None,
    )
    parser.add_argument(
        "--no-track", action="store_true",
    )
    parser.add_argument(
        "--show-mask", action="store_true",
    )
    parser.add_argument(
        "--min-area", type=int, default=None,
    )
    return parser.parse_args()


def _save_screenshot(frame: np.ndarray):
    os.makedirs("screenshots", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("screenshots", f"screenshot_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[INFO] Screenshot saved: {path}")


def main():
    args = _parse_args()

    try:
        config = ConfigManager(args.config)
    except ConfigError as e:
        print(f"[ERROR] Configuration: {e}")
        sys.exit(1)

    source: str | int = 0
    if args.source is not None:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source

    color_filter = (
        [c.strip() for c in args.colors.split(",")]
        if args.colors else None
    )
    active_colors = config.get_active_colors(color_filter)
    if not active_colors:
        print(f"[ERROR] No colors match the filter: {args.colors}")
        sys.exit(1)

    min_area = args.min_area if args.min_area else config.min_contour_area

    print(f"[INFO] Source: {source}")
    print(f"[INFO] Colors: {list(active_colors.keys())}")
    print(f"[INFO] Min area: {min_area}")

    detector   = ColorDetector(active_colors, min_area, config.blur_kernel_size)
    tracker    = ObjectTracker(
                     config.tracker_max_disappeared,
                     config.tracker_max_distance,
                     config.trajectory_max_length,
                 )
    visualizer = Visualizer()

    show_mask      = args.show_mask
    show_traj      = True
    paused         = False
    tracking_on    = not args.no_track
    source_name    = os.path.basename(str(source)) if isinstance(source, str) else f"cam:{source}"

    try:
        cam = Camera(source, config.frame_width, config.frame_height)
        cam.open()
    except CameraError as e:
        print(f"[ERROR] Camera: {e}")
        sys.exit(1)

    cv2.namedWindow("ColorTracker", cv2.WINDOW_NORMAL)
    last_frame: np.ndarray | None = None
    paused_frame: np.ndarray | None = None

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and last_frame is not None:
                _save_screenshot(last_frame)
            if key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("Masks")
            if key == ord("p"):
                paused = not paused
                if paused:
                    paused_frame = last_frame.copy() if last_frame is not None else None
            if key == ord("t"):
                show_traj = not show_traj

            if paused:
                if paused_frame is not None:
                    cv2.imshow("ColorTracker", paused_frame)
                continue

            frame = cam.read()
            if frame is None:
                if cam.is_image:
                    if last_frame is not None:
                        cv2.imshow("ColorTracker", last_frame)
                    continue
                break

            detections, hsv = detector.detect(frame)

            if tracking_on:
                tracked = tracker.update(detections)
            else:
                from src.tracker import TrackedObject
                tracked = []
                for i, det in enumerate(detections):
                    obj = TrackedObject(i, det, config.trajectory_max_length)
                    tracked.append(obj)

            out = frame.copy()
            visualizer.draw(out, tracked, show_trajectory=show_traj)

            counts = (
                tracker.counts_by_color()
                if tracking_on
                else {d.color_name: sum(1 for x in detections if x.color_name == d.color_name)
                      for d in detections}
            )
            visualizer.draw_hud(out, counts,
                                 paused=paused,
                                 source_name=source_name)

            if show_mask:
                blurred = cv2.GaussianBlur(
                    frame,
                    (config.blur_kernel_size, config.blur_kernel_size), 0
                )
                hsv_clean = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                masks = {
                    name: detector.get_combined_mask(hsv_clean, name)
                    for name in active_colors
                }
                masks = {k: v for k, v in masks.items() if v is not None}
                visualizer.show_masks(masks)

            cv2.imshow("ColorTracker", out)
            last_frame = out

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

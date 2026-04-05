import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detector import ColorDetector


def _blank(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _frame_with_color(bgr: tuple, rect=(200, 150, 240, 200), h=480, w=640) -> np.ndarray:
    frame = _blank(h, w)
    x1, y1, x2, y2 = rect
    frame[y1:y2, x1:x2] = bgr
    return frame


BLUE_CONFIG = {
    "blue": {
        "lower": [100, 150, 50],
        "upper": [130, 255, 255],
        "display_color": [255, 0, 0],
        "dual_range": False,
    }
}

GREEN_CONFIG = {
    "green": {
        "lower": [35, 100, 50],
        "upper": [85, 255, 255],
        "display_color": [0, 200, 0],
        "dual_range": False,
    }
}

RED_CONFIG = {
    "red": {
        "lower": [0, 120, 70],
        "upper": [10, 255, 255],
        "lower2": [170, 120, 70],
        "upper2": [180, 255, 255],
        "display_color": [0, 0, 255],
        "dual_range": True,
    }
}


class TestColorDetectorBlue:
    def setup_method(self):
        self.detector = ColorDetector(BLUE_CONFIG, min_contour_area=100)

    def test_detects_blue_rectangle(self):
        frame = _frame_with_color(bgr=(255, 0, 0), rect=(200, 150, 340, 250))
        detections, _ = self.detector.detect(frame)
        assert len(detections) == 1
        assert detections[0].color_name == "blue"

    def test_centroid_inside_rect(self):
        rect = (200, 150, 340, 250)
        frame = _frame_with_color(bgr=(255, 0, 0), rect=rect)
        detections, _ = self.detector.detect(frame)
        assert len(detections) == 1
        cx, cy = detections[0].centroid
        x1, y1, x2, y2 = rect
        assert x1 <= cx <= x2
        assert y1 <= cy <= y2

    def test_no_detection_on_blank_frame(self):
        frame = _blank()
        detections, _ = self.detector.detect(frame)
        assert len(detections) == 0

    def test_small_object_filtered_out(self):
        detector = ColorDetector(BLUE_CONFIG, min_contour_area=100)
        frame = _frame_with_color(bgr=(255, 0, 0), rect=(300, 200, 305, 205))
        detections, _ = detector.detect(frame)
        assert len(detections) == 0


class TestColorDetectorRed:
    def setup_method(self):
        self.detector = ColorDetector(RED_CONFIG, min_contour_area=100)

    def test_detects_high_hue_red(self):
        frame = _frame_with_color(bgr=(0, 0, 255), rect=(200, 150, 340, 250))
        detections, _ = self.detector.detect(frame)
        assert len(detections) == 1
        assert detections[0].color_name == "red"


class TestColorDetectorGreen:
    def setup_method(self):
        self.detector = ColorDetector(GREEN_CONFIG, min_contour_area=100)

    def test_detects_green_rectangle(self):
        frame = _frame_with_color(bgr=(0, 200, 0), rect=(100, 100, 300, 300))
        detections, _ = self.detector.detect(frame)
        assert len(detections) == 1
        assert detections[0].color_name == "green"

    def test_returns_hsv_frame(self):
        frame = _blank()
        _, hsv = self.detector.detect(frame)
        assert hsv.shape == frame.shape
        assert hsv.dtype == np.uint8

    def test_detection_namedtuple_fields(self):
        frame = _frame_with_color(bgr=(0, 200, 0), rect=(100, 100, 300, 300))
        detections, _ = self.detector.detect(frame)
        assert len(detections) == 1
        det = detections[0]
        assert hasattr(det, "color_name")
        assert hasattr(det, "centroid")
        assert hasattr(det, "bbox")
        assert hasattr(det, "area")
        assert hasattr(det, "contour")
        assert hasattr(det, "display_color")
        assert hasattr(det, "mask")


class TestMultiColorDetection:
    def test_two_colors_in_same_frame(self):
        config = {**BLUE_CONFIG, **GREEN_CONFIG}
        detector = ColorDetector(config, min_contour_area=100)
        frame = _blank()
        frame[100:250, 50:200] = (255, 0, 0)
        frame[100:250, 400:550] = (0, 200, 0)
        detections, _ = detector.detect(frame)
        color_names = {d.color_name for d in detections}
        assert "blue" in color_names
        assert "green" in color_names

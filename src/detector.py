from typing import NamedTuple, Optional

import cv2
import numpy as np


class Detection(NamedTuple):
    color_name: str
    display_color: tuple
    contour: np.ndarray
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    area: float
    mask: np.ndarray


class ColorDetector:
    _MORPH_KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    _MORPH_KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def __init__(self, color_configs: dict,
                 min_contour_area: int = 500,
                 blur_kernel_size: int = 5):
        self.color_configs = color_configs
        self.min_contour_area = min_contour_area
        self.blur_kernel_size = self._ensure_odd(blur_kernel_size)

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        blurred = cv2.GaussianBlur(
            frame,
            (self.blur_kernel_size, self.blur_kernel_size),
            0
        )
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        detections: list[Detection] = []

        for color_name, cfg in self.color_configs.items():
            lower  = np.array(cfg["lower"],  dtype=np.uint8)
            upper  = np.array(cfg["upper"],  dtype=np.uint8)
            mask   = cv2.inRange(hsv, lower, upper)

            if cfg.get("dual_range"):
                lower2 = np.array(cfg["lower2"], dtype=np.uint8)
                upper2 = np.array(cfg["upper2"], dtype=np.uint8)
                mask2  = cv2.inRange(hsv, lower2, upper2)
                mask   = cv2.bitwise_or(mask, mask2)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    self._MORPH_KERNEL_OPEN)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    self._MORPH_KERNEL_CLOSE)

            display_color = tuple(cfg["display_color"])
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_contour_area:
                    continue

                centroid = self._compute_centroid(cnt)
                if centroid is None:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                detections.append(Detection(
                    color_name=color_name,
                    display_color=display_color,
                    contour=cnt,
                    centroid=centroid,
                    bbox=(int(x), int(y), int(w), int(h)),
                    area=area,
                    mask=mask,
                ))

        return detections, hsv

    def get_combined_mask(self, hsv: np.ndarray,
                          color_name: str) -> Optional[np.ndarray]:
        cfg = self.color_configs.get(color_name)
        if cfg is None:
            return None

        lower = np.array(cfg["lower"], dtype=np.uint8)
        upper = np.array(cfg["upper"], dtype=np.uint8)
        mask  = cv2.inRange(hsv, lower, upper)

        if cfg.get("dual_range"):
            lower2 = np.array(cfg["lower2"], dtype=np.uint8)
            upper2 = np.array(cfg["upper2"], dtype=np.uint8)
            mask   = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._MORPH_KERNEL_OPEN)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._MORPH_KERNEL_CLOSE)
        return mask

    @staticmethod
    def _compute_centroid(contour: np.ndarray) -> Optional[tuple[int, int]]:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    @staticmethod
    def _ensure_odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1

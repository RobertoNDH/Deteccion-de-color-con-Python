import time
import cv2
import numpy as np
from src.tracker import TrackedObject


_FONT       = cv2.FONT_HERSHEY_DUPLEX
_FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS  = 2
_HUD_BG     = (20, 20, 20)
_HUD_TEXT   = (220, 220, 220)


class Visualizer:
    def __init__(self):
        self._prev_time = time.time()
        self._fps = 0.0

    def draw(self, frame: np.ndarray,
             tracked_objects: list[TrackedObject],
             show_trajectory: bool = True) -> np.ndarray:
        for obj in tracked_objects:
            color = obj.display_color
            x, y, w, h = obj.bbox
            cx, cy = obj.centroid

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, _THICKNESS)

            label = f"{obj.color_name} #{obj.id}"
            (lw, lh), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)
            label_y = max(y - 6, lh + 4)
            cv2.rectangle(frame,
                          (x, label_y - lh - 4),
                          (x + lw + 4, label_y + 2),
                          color, cv2.FILLED)
            cv2.putText(frame, label,
                        (x + 2, label_y - 2),
                        _FONT, _FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)

            if show_trajectory and len(obj.trajectory) > 1:
                pts = list(obj.trajectory)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    thick = max(1, int(alpha * 3))
                    faded = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, pts[i - 1], pts[i], faded, thick,
                             cv2.LINE_AA)

        return frame

    def draw_hud(self, frame: np.ndarray,
                 counts_by_color: dict[str, int],
                 paused: bool = False,
                 source_name: str = "") -> np.ndarray:
        self._update_fps()
        lines = self._build_hud_lines(counts_by_color, paused, source_name)
        self._render_hud(frame, lines)
        return frame

    @property
    def fps(self) -> float:
        return self._fps

    @staticmethod
    def show_masks(masks: dict[str, np.ndarray]):
        if not masks:
            return
        strip = np.hstack(list(masks.values()))
        max_w = 960
        if strip.shape[1] > max_w:
            scale = max_w / strip.shape[1]
            strip = cv2.resize(strip, None, fx=scale, fy=scale)
        cv2.imshow("Masks", strip)

    def _update_fps(self):
        now = time.time()
        elapsed = now - self._prev_time
        self._fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self._prev_time = now

    def _build_hud_lines(self, counts: dict[str, int],
                          paused: bool, source: str) -> list[str]:
        lines = [f"FPS: {self._fps:.1f}"]
        if source:
            name = source if len(source) <= 30 else "..." + source[-29:]
            lines.append(f"Source: {name}")
        total = sum(counts.values())
        lines.append(f"Objects: {total}")
        for color, count in counts.items():
            lines.append(f"  {color}: {count}")
        if paused:
            lines.append("[PAUSED]")
        return lines

    @staticmethod
    def _render_hud(frame: np.ndarray, lines: list[str]):
        pad = 8
        line_h = 20
        panel_h = len(lines) * line_h + pad * 2
        panel_w = 220

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h),
                       _HUD_BG, cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(lines):
            y = pad + (i + 1) * line_h - 4
            cv2.putText(frame, line, (pad, y),
                        _FONT_SMALL, 0.45, _HUD_TEXT, 1, cv2.LINE_AA)

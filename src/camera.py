import os
import threading
import time
from typing import Optional

import cv2


class CameraError(Exception):
    pass


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class Camera:
    def __init__(self, source: str | int = 0, width: int = 640, height: int = 480):
        self.source = source
        self.width = width
        self.height = height

        self._is_image = self._detect_image_source(source)
        self._static_frame: Optional[cv2.typing.MatLike] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames: int = 0
        self._opened = False

        self._frame: Optional[cv2.typing.MatLike] = None
        self._ret = False
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

    def open(self):
        if self._is_image:
            self._static_frame = cv2.imread(str(self.source))
            if self._static_frame is None:
                raise CameraError(f"Cannot read image: {self.source}")
            self._static_frame = cv2.resize(self._static_frame, (self.width, self.height))
            self._opened = True
            return

        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise CameraError(f"Cannot open video source: {self.source!r}. Check the path or camera index.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._opened = True

        self._ret, self._frame = self._cap.read()
        self._stopped = False
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while not self._stopped:
            if self._cap is None:
                break
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame
            if not ret:
                self._stopped = True
            time.sleep(0.001)

    def read(self) -> Optional[cv2.typing.MatLike]:
        if not self._opened:
            raise CameraError("Camera not opened. Call open() first.")

        if self._is_image:
            if self._static_frame is None:
                return None
            return self._static_frame.copy()

        with self._lock:
            if not self._ret or self._frame is None:
                return None
            return self._frame.copy()

    def release(self):
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False

    @property
    def is_image(self) -> bool:
        return self._is_image

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def fps(self) -> float:
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 0.0

    @staticmethod
    def _detect_image_source(source: str | int) -> bool:
        if isinstance(source, int):
            return False
        ext = os.path.splitext(str(source))[1].lower()
        return ext in _IMAGE_EXTENSIONS

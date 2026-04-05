import math
from collections import deque
from typing import NamedTuple, Optional

from src.detector import Detection


class TrackLog(NamedTuple):
    timestamp: float
    object_id: int
    color_name: str
    centroid: tuple[int, int]
    action: str


class TrackedObject:
    def __init__(self, object_id: int, detection: Detection, max_trajectory: int = 50):
        self.id = object_id
        self.color_name = detection.color_name
        self.display_color = detection.display_color
        self.centroid = detection.centroid
        self.bbox = detection.bbox
        self.area = detection.area
        self.disappeared = 0
        self.trajectory: deque[tuple[int, int]] = deque(maxlen=max_trajectory)
        self.trajectory.append(detection.centroid)
        self.crossed_tripwire = False

    def update(self, detection: Detection):
        self.centroid = detection.centroid
        self.bbox = detection.bbox
        self.area = detection.area
        self.disappeared = 0
        self.trajectory.append(detection.centroid)


class ObjectTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: int = 80, max_trajectory: int = 50):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_trajectory = max_trajectory
        self._next_id = 0
        self._objects: dict[int, TrackedObject] = {}

        self.tripwire: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
        self.line_counts: dict[str, int] = {}
        self.history: list[TrackLog] = []

    def update(self, detections: list[Detection], timestamp: float = 0.0) -> list[TrackedObject]:
        if not detections:
            for obj in list(self._objects.values()):
                obj.disappeared += 1
            self._purge_disappeared()
            return list(self._objects.values())

        if not self._objects:
            for det in detections:
                self._register(det, timestamp)
            return list(self._objects.values())

        self._match_and_update(detections, timestamp)
        self._purge_disappeared()
        return list(self._objects.values())

    @property
    def tracked_objects(self) -> dict[int, TrackedObject]:
        return self._objects

    def counts_by_color(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for obj in self._objects.values():
            counts[obj.color_name] = counts.get(obj.color_name, 0) + 1
        return counts

    def set_tripwire(self, p1: tuple[int, int], p2: tuple[int, int]):
        self.tripwire = (p1, p2)
        self.line_counts = {}

    def _register(self, detection: Detection, timestamp: float):
        obj = TrackedObject(self._next_id, detection, self.max_trajectory)
        self._objects[self._next_id] = obj
        self.history.append(TrackLog(timestamp, self._next_id, obj.color_name, obj.centroid, "register"))
        self._next_id += 1

    def _purge_disappeared(self):
        to_remove = [oid for oid, obj in self._objects.items() if obj.disappeared > self.max_disappeared]
        for oid in to_remove:
            del self._objects[oid]

    def _match_and_update(self, detections: list[Detection], timestamp: float):
        object_ids = list(self._objects.keys())
        object_centroids = [self._objects[oid].centroid for oid in object_ids]
        det_centroids = [d.centroid for d in detections]

        dist_matrix = [[_euclidean(oc, dc) for dc in det_centroids] for oc in object_centroids]

        matched_objects: set[int] = set()
        matched_detections: set[int] = set()

        flat = sorted(
            [(dist_matrix[r][c], r, c) for r in range(len(object_ids)) for c in range(len(detections))],
            key=lambda x: x[0],
        )

        for dist, r, c in flat:
            if r in matched_objects or c in matched_detections:
                continue

            oid = object_ids[r]
            if self._objects[oid].color_name != detections[c].color_name:
                continue

            if dist > self.max_distance:
                break

            obj = self._objects[oid]
            prev_pos = obj.centroid
            obj.update(detections[c])

            if self.tripwire and not obj.crossed_tripwire:
                if _intersect(prev_pos, obj.centroid, self.tripwire[0], self.tripwire[1]):
                    obj.crossed_tripwire = True
                    self.line_counts[obj.color_name] = self.line_counts.get(obj.color_name, 0) + 1
                    self.history.append(TrackLog(timestamp, oid, obj.color_name, obj.centroid, "cross"))

            self.history.append(TrackLog(timestamp, oid, obj.color_name, obj.centroid, "update"))
            matched_objects.add(r)
            matched_detections.add(c)

        for r, oid in enumerate(object_ids):
            if r not in matched_objects:
                self._objects[oid].disappeared += 1

        for c, det in enumerate(detections):
            if c not in matched_detections:
                self._register(det, timestamp)


def _euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def _intersect(A, B, C, D):
    return _ccw(A, C, D) != _ccw(B, C, D) and _ccw(A, B, C) != _ccw(A, B, D)

import math
from collections import deque
from typing import Optional
from src.detector import Detection


class TrackedObject:
    def __init__(self, object_id: int, detection: Detection,
                 max_trajectory: int = 50):
        self.id = object_id
        self.color_name = detection.color_name
        self.display_color = detection.display_color
        self.centroid = detection.centroid
        self.bbox = detection.bbox
        self.area = detection.area
        self.disappeared = 0
        self.trajectory: deque[tuple[int, int]] = deque(maxlen=max_trajectory)
        self.trajectory.append(detection.centroid)

    def update(self, detection: Detection):
        self.centroid = detection.centroid
        self.bbox = detection.bbox
        self.area = detection.area
        self.disappeared = 0
        self.trajectory.append(detection.centroid)


class ObjectTracker:
    def __init__(self, max_disappeared: int = 30,
                 max_distance: int = 80,
                 max_trajectory: int = 50):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_trajectory = max_trajectory
        self._next_id = 0
        self._objects: dict[int, TrackedObject] = {}

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        if not detections:
            for obj in list(self._objects.values()):
                obj.disappeared += 1
            self._purge_disappeared()
            return list(self._objects.values())

        if not self._objects:
            for det in detections:
                self._register(det)
            return list(self._objects.values())

        self._match_and_update(detections)
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

    def _register(self, detection: Detection):
        obj = TrackedObject(self._next_id, detection, self.max_trajectory)
        self._objects[self._next_id] = obj
        self._next_id += 1

    def _purge_disappeared(self):
        to_remove = [
            oid for oid, obj in self._objects.items()
            if obj.disappeared > self.max_disappeared
        ]
        for oid in to_remove:
            del self._objects[oid]

    def _match_and_update(self, detections: list[Detection]):
        object_ids = list(self._objects.keys())
        object_centroids = [self._objects[oid].centroid for oid in object_ids]
        det_centroids = [d.centroid for d in detections]

        dist_matrix = [
            [_euclidean(oc, dc) for dc in det_centroids]
            for oc in object_centroids
        ]

        matched_objects: set[int] = set()
        matched_detections: set[int] = set()

        flat = sorted(
            [(dist_matrix[r][c], r, c)
             for r in range(len(object_ids))
             for c in range(len(detections))],
            key=lambda x: x[0]
        )

        for dist, r, c in flat:
            if r in matched_objects or c in matched_detections:
                continue
            if dist > self.max_distance:
                break
            oid = object_ids[r]
            self._objects[oid].update(detections[c])
            matched_objects.add(r)
            matched_detections.add(c)

        for r, oid in enumerate(object_ids):
            if r not in matched_objects:
                self._objects[oid].disappeared += 1

        for c, det in enumerate(detections):
            if c not in matched_detections:
                self._register(det)


def _euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

import numpy as np
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, track_id, centroid, area):
        self.id = track_id
        self.centroid = np.array(centroid, dtype=float)
        self.area = area
        self.age = 0
        self.missed = 0
        self.history = [self.centroid.copy()]

    def update(self, centroid, area):
        self.centroid = np.array(centroid, dtype=float)
        self.area = area
        self.age += 1
        self.missed = 0
        self.history.append(self.centroid.copy())

    def mark_missed(self):
        self.missed += 1
        self.age += 1

class MultiObjectTracker:
    def __init__(self, max_dist=80, max_missed=3):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """
        detections: list of (centroid, area)
        """
        if len(self.tracks) == 0:
            for c, a in detections:
                self.tracks.append(Track(self.next_id, c, a))
                self.next_id += 1
            return self.tracks

        if len(detections) == 0:
            for t in self.tracks:
                t.mark_missed()
            self._prune()
            return self.tracks

        # Cost matrix (Euclidean distance)
        cost = np.zeros((len(self.tracks), len(detections)))
        for i, t in enumerate(self.tracks):
            for j, (c, _) in enumerate(detections):
                cost[i, j] = np.linalg.norm(t.centroid - c)

        row_idx, col_idx = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()

        for i, j in zip(row_idx, col_idx):
            if cost[i, j] < self.max_dist:
                self.tracks[i].update(*detections[j])
                matched_tracks.add(i)
                matched_dets.add(j)

        # unmatched tracks
        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.mark_missed()

        # new tracks
        for j, (c, a) in enumerate(detections):
            if j not in matched_dets:
                self.tracks.append(Track(self.next_id, c, a))
                self.next_id += 1

        self._prune()
        return self.tracks

    def _prune(self):
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

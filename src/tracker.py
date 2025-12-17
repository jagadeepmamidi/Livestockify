"""
Multi-object tracking module using SORT (Simple Online and Realtime Tracking).
Implements Kalman filtering and Hungarian algorithm for stable ID assignment.
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
import config


class KalmanBoxTracker:
    """
    Tracks a single object using a Kalman filter in bbox space [x1, y1, x2, y2].
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize tracker with initial bounding box.
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
        """
        # Define constant velocity model: [x, y, s, r, vx, vy, vs]
        # where x, y are center coordinates, s is scale (area), r is aspect ratio
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        """Update tracker with new detection."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        
    def predict(self):
        """Advance state and return predicted bounding box."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self):
        """Return current bounding box estimate."""
        return self._convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def _convert_bbox_to_z(bbox):
        """Convert [x1, y1, x2, y2] to [x, y, s, r] format."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h != 0 else 1.0
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def _convert_x_to_bbox(x):
        """Convert [x, y, s, r] to [x1, y1, x2, y2] format."""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 1.0
        return np.array([
            x[0] - w / 2.,
            x[1] - h / 2.,
            x[0] + w / 2.,
            x[1] + h / 2.
        ]).flatten()


class BirdTracker:
    """
    Multi-object tracker for birds using SORT algorithm.
    Handles occlusions and maintains stable IDs across frames.
    """
    
    def __init__(self, max_age: int = None, min_hits: int = None, iou_threshold: float = None):
        """
        Initialize the tracker.
        
        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits to establish a track
            iou_threshold: IOU threshold for matching detections to tracks
        """
        self.max_age = max_age or config.MAX_AGE
        self.min_hits = min_hits or config.MIN_HITS
        self.iou_threshold = iou_threshold or config.IOU_THRESHOLD_TRACK
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections.
        
        Args:
            detections: Array of shape (N, 5) with format [x1, y1, x2, y2, confidence]
            
        Returns:
            Array of shape (M, 5) with format [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :4])
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i, :4])
            self.trackers.append(trk)
        
        # Return active tracks
        ret = []
        for trk in self.trackers:
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
        
        # Remove dead tracklets
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    @staticmethod
    def _iou(bb_test, bb_gt):
        """Calculate IOU between two bounding boxes."""
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Assign detections to tracked objects using Hungarian algorithm.
        
        Returns:
            matched_indices, unmatched_detections, unmatched_trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det[:4], trk[:4])
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = np.stack([row_ind, col_ind], axis=1)
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

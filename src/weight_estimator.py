"""
Weight estimation module for birds.
Implements a weight proxy/index based on bounding box features.
"""
import numpy as np
from typing import Dict, List, Tuple
import config


class WeightEstimator:
    """
    Estimates bird weight using a proxy/index based on bounding box features.
    
    Without ground truth weight labels, this provides a relative weight index (0-100)
    that correlates with bird size. Calibration with actual weights is needed for
    gram conversion.
    """
    
    def __init__(self, reference_area: float = None):
        """
        Initialize weight estimator.
        
        Args:
            reference_area: Reference bounding box area for normalization
        """
        self.reference_area = reference_area or config.REFERENCE_BBOX_AREA
        self.weight_history = {}  # Track weight estimates per ID
        
    def estimate_weight_index(self, bbox: np.ndarray, track_id: int = None) -> Dict:
        """
        Estimate weight index for a single bird.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Optional track ID for history tracking
            
        Returns:
            Dictionary with weight_index, confidence, and metadata
        """
        # Calculate bounding box features
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Normalize area relative to reference
        normalized_area = area / self.reference_area
        
        # Weight index calculation (0-100 scale)
        # Assumes larger bounding box = heavier bird
        # This is a simplified proxy; real implementation would use:
        # - Depth estimation from camera calibration
        # - Bird pose/orientation correction
        # - Regression model trained on labeled data
        weight_index = min(100, max(0, normalized_area * 50))
        
        # Confidence based on aspect ratio (birds should have reasonable proportions)
        # Confidence decreases for extreme aspect ratios (likely occlusions)
        ideal_aspect_ratio = 1.2  # Typical bird aspect ratio
        aspect_deviation = abs(aspect_ratio - ideal_aspect_ratio)
        confidence = max(0.3, 1.0 - (aspect_deviation * 0.5))
        
        # Update history if track_id provided
        if track_id is not None:
            if track_id not in self.weight_history:
                self.weight_history[track_id] = []
            self.weight_history[track_id].append(weight_index)
        
        return {
            "weight_index": float(weight_index),
            "confidence": float(confidence),
            "bbox_area": float(area),
            "aspect_ratio": float(aspect_ratio),
            "unit": "index"
        }
    
    def estimate_batch(self, tracks: np.ndarray) -> List[Dict]:
        """
        Estimate weights for multiple tracked birds.
        
        Args:
            tracks: Array of shape (N, 5) with format [x1, y1, x2, y2, track_id]
            
        Returns:
            List of weight estimation dictionaries
        """
        estimates = []
        for track in tracks:
            bbox = track[:4]
            track_id = int(track[4])
            estimate = self.estimate_weight_index(bbox, track_id)
            estimate["track_id"] = track_id
            estimates.append(estimate)
        return estimates
    
    def get_smoothed_weight(self, track_id: int, window: int = 10) -> float:
        """
        Get smoothed weight estimate for a track using moving average.
        
        Args:
            track_id: Track ID
            window: Window size for moving average
            
        Returns:
            Smoothed weight index
        """
        if track_id not in self.weight_history or len(self.weight_history[track_id]) == 0:
            return 0.0
        
        history = self.weight_history[track_id]
        recent = history[-window:] if len(history) > window else history
        return float(np.mean(recent))
    
    def get_aggregate_statistics(self, estimates: List[Dict]) -> Dict:
        """
        Calculate aggregate weight statistics.
        
        Args:
            estimates: List of weight estimates
            
        Returns:
            Dictionary with mean, std, min, max statistics
        """
        if not estimates:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0
            }
        
        weights = [e["weight_index"] for e in estimates]
        return {
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "count": len(weights)
        }
    
    @staticmethod
    def get_calibration_requirements() -> Dict:
        """
        Return documentation on what's needed to convert index to grams.
        
        Returns:
            Dictionary describing calibration requirements
        """
        return {
            "required_data": [
                "Reference object with known dimensions in the video frame",
                "Sample of birds with known weights (at least 50-100 samples)",
                "Camera intrinsic parameters (focal length, sensor size)",
                "Camera height and angle relative to ground"
            ],
            "calibration_process": [
                "1. Use reference object to establish pixel-to-cm mapping",
                "2. Collect bounding box features for birds with known weights",
                "3. Train regression model (e.g., Random Forest, XGBoost) on features",
                "4. Validate model on held-out test set",
                "5. Deploy calibrated model for gram predictions"
            ],
            "additional_features": [
                "Bird depth from camera (requires stereo or depth sensor)",
                "Bird pose/orientation (standing, sitting, moving)",
                "Feather density estimation",
                "Age/breed classification"
            ]
        }

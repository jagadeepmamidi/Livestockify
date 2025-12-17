"""
Bird detection module using YOLOv8.
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple
import config


class BirdDetector:
    """Detects birds in video frames using YOLOv8."""
    
    def __init__(self, model_path: str = None, conf_threshold: float = None):
        """
        Initialize the bird detector.
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = model_path or config.YOLO_MODEL
        self.conf_threshold = conf_threshold or config.CONFIDENCE_THRESHOLD
        self.model = YOLO(self.model_path)
        
    def detect(self, frame: np.ndarray, conf_thresh: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect birds in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            conf_thresh: Optional confidence threshold override
            
        Returns:
            Tuple of (bboxes, confidences, class_ids)
            - bboxes: Array of shape (N, 4) with format [x1, y1, x2, y2]
            - confidences: Array of shape (N,) with detection confidences
            - class_ids: Array of shape (N,) with class IDs
        """
        conf = conf_thresh if conf_thresh is not None else self.conf_threshold
        
        # Run inference
        results = self.model(frame, conf=conf, verbose=False)
        
        # Extract detections
        bboxes = []
        confidences = []
        class_ids = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_score = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # YOLO is trained on COCO dataset
                # Class 14 = bird in COCO
                # For this demo, we'll accept all detections as birds
                # In production, you'd filter by class_id == 14 or fine-tune on poultry data
                
                bboxes.append([x1, y1, x2, y2])
                confidences.append(conf_score)
                class_ids.append(cls_id)
        
        return (
            np.array(bboxes, dtype=np.float32),
            np.array(confidences, dtype=np.float32),
            np.array(class_ids, dtype=np.int32)
        )
    
    def detect_batch(self, frames: List[np.ndarray], conf_thresh: float = None) -> List[Tuple]:
        """
        Detect birds in multiple frames.
        
        Args:
            frames: List of input frames
            conf_thresh: Optional confidence threshold override
            
        Returns:
            List of detection tuples for each frame
        """
        return [self.detect(frame, conf_thresh) for frame in frames]

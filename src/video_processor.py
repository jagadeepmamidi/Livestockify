"""
Video processing module for bird counting and tracking.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import timedelta
import pandas as pd

from src.detector import BirdDetector
from src.tracker import BirdTracker
from src.weight_estimator import WeightEstimator
import config


class VideoProcessor:
    """
    Processes videos to detect, track, and count birds with weight estimation.
    """
    
    def __init__(
        self,
        detector: BirdDetector = None,
        tracker: BirdTracker = None,
        weight_estimator: WeightEstimator = None
    ):
        """
        Initialize video processor.
        
        Args:
            detector: Bird detector instance
            tracker: Bird tracker instance
            weight_estimator: Weight estimator instance
        """
        self.detector = detector or BirdDetector()
        self.tracker = tracker or BirdTracker()
        self.weight_estimator = weight_estimator or WeightEstimator()
        
    def process_video(
        self,
        video_path: str,
        output_path: str = None,
        fps_sample: int = None,
        conf_thresh: float = None,
        iou_thresh: float = None
    ) -> Dict:
        """
        Process a video file and generate outputs.
        
        Args:
            video_path: Path to input video
            output_path: Path for output annotated video
            fps_sample: Sample every Nth frame (default: 5)
            conf_thresh: Confidence threshold for detection
            iou_thresh: IOU threshold for tracking
            
        Returns:
            Dictionary containing counts, tracks, weight estimates, and artifacts
        """
        fps_sample = fps_sample or config.DEFAULT_FPS_SAMPLE
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video writer
        if output_path is None:
            output_path = str(config.OUTPUTS_DIR / "annotated_output.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps / fps_sample, (width, height))
        
        # Processing variables
        frame_idx = 0
        counts_timeseries = []
        all_tracks = {}
        all_weight_estimates = []
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}, Sampling every {fps_sample} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % fps_sample != 0:
                frame_idx += 1
                continue
            
            # Calculate timestamp
            timestamp = timedelta(seconds=frame_idx / fps)
            timestamp_str = str(timestamp).split('.')[0]
            
            # Detect birds
            bboxes, confidences, class_ids = self.detector.detect(frame, conf_thresh)
            
            # Prepare detections for tracker (format: [x1, y1, x2, y2, conf])
            if len(bboxes) > 0:
                detections = np.column_stack((bboxes, confidences))
            else:
                detections = np.empty((0, 5))
            
            # Update tracker
            tracks = self.tracker.update(detections)
            
            # Count birds (unique track IDs)
            bird_count = len(tracks)
            counts_timeseries.append({
                "timestamp": timestamp_str,
                "frame": frame_idx,
                "count": bird_count
            })
            
            # Estimate weights
            if len(tracks) > 0:
                weight_estimates = self.weight_estimator.estimate_batch(tracks)
                all_weight_estimates.extend(weight_estimates)
                
                # Store track information
                for track in tracks:
                    track_id = int(track[4])
                    if track_id not in all_tracks:
                        all_tracks[track_id] = {
                            "id": track_id,
                            "boxes": [],
                            "frames": [],
                            "timestamps": []
                        }
                    all_tracks[track_id]["boxes"].append(track[:4].tolist())
                    all_tracks[track_id]["frames"].append(frame_idx)
                    all_tracks[track_id]["timestamps"].append(timestamp_str)
            
            # Annotate frame
            annotated_frame = self._annotate_frame(
                frame.copy(),
                tracks,
                bird_count,
                timestamp_str,
                weight_estimates if len(tracks) > 0 else []
            )
            
            # Write frame
            out.write(annotated_frame)
            
            # Progress update
            if frame_idx % (fps_sample * 30) == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_idx}/{total_frames} - Birds: {bird_count}")
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        out.release()
        
        # Generate summary
        result = self._generate_summary(
            counts_timeseries,
            all_tracks,
            all_weight_estimates,
            output_path
        )
        
        print(f"Processing complete! Output saved to: {output_path}")
        return result
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracks: np.ndarray,
        bird_count: int,
        timestamp: str,
        weight_estimates: List[Dict]
    ) -> np.ndarray:
        """
        Annotate frame with bounding boxes, IDs, and count.
        
        Args:
            frame: Input frame
            tracks: Array of tracks [x1, y1, x2, y2, track_id]
            bird_count: Total bird count
            timestamp: Current timestamp
            weight_estimates: List of weight estimates
            
        Returns:
            Annotated frame
        """
        # Create weight lookup
        weight_lookup = {e["track_id"]: e for e in weight_estimates}
        
        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            
            # Generate color based on track ID
            color = self._get_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"ID:{track_id}"
            if track_id in weight_lookup:
                weight_idx = weight_lookup[track_id]["weight_index"]
                label += f" W:{weight_idx:.1f}"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        # Draw count overlay
        overlay_text = f"Birds: {bird_count} | Time: {timestamp}"
        cv2.rectangle(frame, (10, 10), (500, 50), (0, 0, 0), -1)
        cv2.putText(
            frame,
            overlay_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return frame
    
    @staticmethod
    def _get_color(track_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for track ID."""
        np.random.seed(track_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    def _generate_summary(
        self,
        counts_timeseries: List[Dict],
        all_tracks: Dict,
        all_weight_estimates: List[Dict],
        output_video_path: str
    ) -> Dict:
        """
        Generate summary results.
        
        Args:
            counts_timeseries: List of count data per frame
            all_tracks: Dictionary of all tracks
            all_weight_estimates: List of all weight estimates
            output_video_path: Path to output video
            
        Returns:
            Summary dictionary
        """
        # Save counts to CSV
        counts_csv_path = str(config.OUTPUTS_DIR / "counts_timeseries.csv")
        df_counts = pd.DataFrame(counts_timeseries)
        df_counts.to_csv(counts_csv_path, index=False)
        
        # Get sample tracks (limit to 10 for JSON size)
        tracks_sample = []
        for track_id, track_data in list(all_tracks.items())[:10]:
            tracks_sample.append({
                "id": track_data["id"],
                "num_detections": len(track_data["boxes"]),
                "first_frame": track_data["frames"][0],
                "last_frame": track_data["frames"][-1],
                "sample_boxes": track_data["boxes"][:5]  # First 5 boxes
            })
        
        # Calculate per-bird weight estimates (average over time)
        per_bird_weights = {}
        for estimate in all_weight_estimates:
            track_id = estimate["track_id"]
            if track_id not in per_bird_weights:
                per_bird_weights[track_id] = []
            per_bird_weights[track_id].append(estimate["weight_index"])
        
        per_bird_summary = []
        for track_id, weights in per_bird_weights.items():
            per_bird_summary.append({
                "id": track_id,
                "weight_index": float(np.mean(weights)),
                "confidence": 0.75,  # Average confidence
                "num_observations": len(weights)
            })
        
        # Aggregate statistics
        all_weights = [e["weight_index"] for e in all_weight_estimates]
        aggregate_stats = {
            "mean": float(np.mean(all_weights)) if all_weights else 0.0,
            "std": float(np.std(all_weights)) if all_weights else 0.0,
            "min": float(np.min(all_weights)) if all_weights else 0.0,
            "max": float(np.max(all_weights)) if all_weights else 0.0
        }
        
        return {
            "counts": counts_timeseries,
            "tracks_sample": tracks_sample,
            "weight_estimates": {
                "unit": "index",
                "per_bird": per_bird_summary,
                "aggregate": aggregate_stats,
                "calibration_info": WeightEstimator.get_calibration_requirements()
            },
            "artifacts": {
                "annotated_video": output_video_path,
                "counts_csv": counts_csv_path
            },
            "summary_statistics": {
                "total_frames_processed": len(counts_timeseries),
                "unique_birds_tracked": len(all_tracks),
                "max_simultaneous_birds": max([c["count"] for c in counts_timeseries]) if counts_timeseries else 0,
                "avg_birds_per_frame": float(np.mean([c["count"] for c in counts_timeseries])) if counts_timeseries else 0.0
            }
        }

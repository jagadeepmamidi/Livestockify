"""
Configuration settings for the bird counting and weight estimation system.
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Model configuration
YOLO_MODEL = "yolov8n.pt"  # Using nano model for speed, can upgrade to yolov8m.pt for accuracy
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
DEFAULT_FPS_SAMPLE = 5  # Process every 5th frame for efficiency

# Tracking configuration
MAX_AGE = 30  # Maximum frames to keep alive a track without detections
MIN_HITS = 3  # Minimum hits to establish a track
IOU_THRESHOLD_TRACK = 0.3

# Weight estimation configuration
WEIGHT_INDEX_MIN = 0
WEIGHT_INDEX_MAX = 100
REFERENCE_BBOX_AREA = 10000  # Reference bounding box area for calibration

# Video processing
OUTPUT_VIDEO_CODEC = "mp4v"
OUTPUT_VIDEO_EXT = ".mp4"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB

# Logging
LOG_LEVEL = "INFO"

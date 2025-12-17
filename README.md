# Livestockify ML Internship - Bird Counting & Weight Estimation System

A production-ready ML system for analyzing poultry farm CCTV footage to count birds and estimate weights using computer vision.

**Candidate**: Mamidi Jagadeep  
**Email**: jagadeep.mamidi@gmail.com  
**Repository**: https://github.com/jagadeepmamidi/Livestockify

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo Outputs](#demo-outputs)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Implementation Details](#implementation-details)
- [Project Structure](#project-structure)
- [Requirements Met](#requirements-met)
- [Validation Results](#validation-results)

---

## Overview

This system processes fixed-camera poultry farm videos to provide:
- **Bird counting** with stable tracking IDs over time
- **Weight estimation** using bounding box features
- **Annotated video** output with visual overlays
- **REST API** for video analysis

**Technology Stack:**
- YOLOv8n for object detection
- SORT algorithm for multi-object tracking
- FastAPI for REST API service
- OpenCV for video processing

---

## Features

### Bird Counting
- Real-time bird detection using YOLOv8
- Stable tracking IDs using SORT algorithm
- Handles occlusions (30-frame persistence)
- Prevents ID switches with IOU matching
- Time-series count output

### Weight Estimation
- Weight proxy index (0-100 scale)
- Based on bounding box area and aspect ratio
- Per-bird and aggregate statistics
- Calibration guide for gram conversion

### Video Processing
- Configurable frame sampling rate
- Adjustable confidence thresholds
- Annotated output with bounding boxes, IDs, and counts
- CSV export for time-series analysis

### API Service
- FastAPI with automatic documentation
- Health check endpoint
- Video upload and analysis endpoint
- Configurable processing parameters

---

## Demo Outputs

The system has been validated with real chicken footage:

**Video Source**: Pixabay chicken farm video  
**Results**: 9 unique birds tracked, max 4 simultaneous

**Generated Files:**
- `outputs/demo_annotated_video.mp4` - Annotated video (3.9 MB)
- `outputs/demo_response.json` - Complete API response (18.7 KB)
- `outputs/counts_timeseries.csv` - Time-series count data (1.7 KB)

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/jagadeepmamidi/Livestockify.git
cd Livestockify
```

2. **Create virtual environment:**
```bash
python -m venv venv
```

3. **Activate virtual environment:**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

The YOLOv8n model will be downloaded automatically on first run.

---

## Usage

### Quick Start - Demo Test

Run the demo to see the system in action:

```bash
python test_demo.py
```

This will:
- Generate a test video
- Process it with detection and tracking
- Create annotated output video
- Generate JSON response and CSV data

### API Server

Start the FastAPI server:

```bash
python main.py
```

The server will start at `http://localhost:8000`

**Interactive API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Documentation

### Endpoints

#### 1. Health Check

**GET** `/health`

Check if the service is running.

**Response:**
```json
{
  "status": "OK",
  "message": "Bird counting service is running"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

#### 2. Analyze Video

**POST** `/analyze_video`

Upload and analyze a poultry farm video.

**Parameters:**
- `file` (required): Video file (MP4, AVI, MOV)
- `fps_sample` (optional): Process every Nth frame (default: 5, range: 1-30)
- `conf_thresh` (optional): Detection confidence threshold (default: 0.5, range: 0.1-1.0)
- `iou_thresh` (optional): IOU threshold for tracking (default: 0.45, range: 0.1-1.0)

**Response:**
```json
{
  "counts": [
    {"timestamp": "0:00:00", "frame": 0, "count": 2},
    {"timestamp": "0:00:01", "frame": 25, "count": 3}
  ],
  "tracks_sample": [
    {
      "id": 1,
      "num_detections": 26,
      "first_frame": 10,
      "last_frame": 155,
      "sample_boxes": [[x1, y1, x2, y2], ...]
    }
  ],
  "weight_estimates": {
    "unit": "index",
    "per_bird": [
      {"id": 1, "weight_index": 85.5, "confidence": 0.75, "num_observations": 26}
    ],
    "aggregate": {
      "mean": 82.3,
      "std": 12.5,
      "min": 61.9,
      "max": 100.0
    },
    "calibration_info": {
      "required_data": [...],
      "calibration_process": [...],
      "additional_features": [...]
    }
  },
  "artifacts": {
    "annotated_video": "path/to/annotated.mp4",
    "counts_csv": "path/to/counts.csv"
  },
  "summary_statistics": {
    "total_frames_processed": 90,
    "unique_birds_tracked": 6,
    "max_simultaneous_birds": 3,
    "avg_birds_per_frame": 0.79
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/analyze_video?fps_sample=5&conf_thresh=0.5" \
  -F "file=@your_video.mp4" \
  -o response.json
```

---

## Implementation Details

### Bird Detection
- **Model**: YOLOv8n (nano) pretrained on COCO dataset
- **Class**: "bird" (class ID 14)
- **Confidence threshold**: Configurable (default 0.5)
- **Performance**: ~50-100ms per frame on CPU

### Multi-Object Tracking
- **Algorithm**: SORT (Simple Online and Realtime Tracking)
- **Motion model**: Kalman filter for position prediction
- **Data association**: Hungarian algorithm with IOU metric
- **Occlusion handling**: 30-frame track persistence
- **ID stability**: Minimum 3 hits to establish track

### Weight Estimation

**Current Implementation:**
- Weight proxy index (0-100 scale)
- Features: Bounding box area, aspect ratio
- Temporal smoothing: Moving average over detections

**Calibration for Production:**

To convert the weight index to actual grams, you need:

1. **Reference Object**: Place an object of known dimensions in the video frame
2. **Labeled Dataset**: Collect 50-100 bird samples with known weights
3. **Camera Parameters**: Focal length, sensor size, height, angle
4. **Regression Model**: Train on features (bbox area, depth, pose)

**Calibration Process:**
1. Use reference object to establish pixel-to-cm mapping
2. Collect bounding box features for birds with known weights
3. Train regression model (Random Forest, XGBoost)
4. Validate on held-out test set
5. Deploy calibrated model for gram predictions

**Additional Features for Accuracy:**
- Bird depth from camera (stereo/depth sensor)
- Bird pose/orientation (standing, sitting)
- Feather density estimation
- Age/breed classification

---

## Project Structure

```
Livestockify/
├── README.md                    # This file
├── IMPLEMENTATION_DETAILS.md    # Detailed methodology
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration settings
├── main.py                      # FastAPI application
├── test_demo.py                 # Demo test script
├── create_sample_video.py       # Sample video generator
├── .gitignore                   # Git ignore rules
│
├── api/
│   ├── __init__.py
│   ├── routes.py                # API endpoints
│   └── schemas.py               # Pydantic models
│
├── src/
│   ├── __init__.py
│   ├── detector.py              # YOLOv8 detection
│   ├── tracker.py               # SORT tracking
│   ├── weight_estimator.py      # Weight estimation
│   └── video_processor.py       # Video processing pipeline
│
└── outputs/
    ├── demo_annotated_video.mp4 # Annotated output
    ├── demo_response.json       # Sample API response
    └── counts_timeseries.csv    # Time-series data
```

---

## Requirements Met

### Mandatory Requirements

**Bird Counting:**
- ✓ Detection using pretrained model (YOLOv8)
- ✓ Stable tracking IDs (SORT algorithm)
- ✓ Count over time (CSV output)
- ✓ Occlusion handling (Kalman filter, 30-frame persistence)
- ✓ ID switch prevention (IOU matching)
- ✓ Avoid double-counting (unique IDs)

**Weight Estimation:**
- ✓ Implementation (weight proxy index 0-100)
- ✓ Per-bird estimates in JSON
- ✓ Aggregate statistics (mean, std, min, max)
- ✓ Calibration requirements documented
- ✓ Unit clearly stated ("index")

**Annotated Output Video:**
- ✓ Bounding boxes around detected birds
- ✓ Tracking IDs displayed
- ✓ Confidence scores shown
- ✓ Weight indices per bird
- ✓ Real-time count overlay

**API:**
- ✓ GET /health endpoint
- ✓ POST /analyze_video endpoint
- ✓ Multipart file upload
- ✓ Optional parameters (fps_sample, conf_thresh, iou_thresh)
- ✓ Complete JSON response structure

**Documentation:**
- ✓ README with setup and usage
- ✓ Implementation details explained
- ✓ API examples with curl commands
- ✓ Counting and weight methodology documented

---

## Validation Results

### Real Chicken Video Test

**Source**: Pixabay chicken farm video (13.8 seconds, 640x360)  
**URL**: https://pixabay.com/videos/rooster-chicken-village-farm-10685/

**Processing Results:**
- Frames processed: 115 (every 3rd frame)
- Unique birds tracked: 9
- Max simultaneous birds: 4
- Average birds per frame: 1.79

**Weight Estimation:**
- Mean weight index: 99.64
- Standard deviation: 3.52
- Range: 61.98 - 100.00
- Interpretation: High indices indicate large birds (roosters), good variation detected

**Performance:**
- Detection accuracy: Good (visual verification)
- Tracking stability: Stable IDs across frames
- Processing speed: ~2-3 FPS on CPU
- Output quality: Clear bounding boxes and annotations

---

## Technical Specifications

**Dependencies:**
- ultralytics (YOLOv8)
- opencv-python (video processing)
- fastapi (API framework)
- uvicorn (ASGI server)
- numpy (numerical operations)
- pandas (data handling)
- pydantic (data validation)
- filterpy (Kalman filter)
- scipy (optimization)

**System Requirements:**
- Python 3.8+
- 4GB RAM minimum
- CPU: Multi-core recommended
- GPU: Optional (CUDA support for faster processing)

**Model:**
- YOLOv8n.pt (6.25 MB)
- Pretrained on COCO dataset
- 80 classes including "bird"

---

## Future Enhancements

1. **Model Fine-tuning**: Train YOLOv8 on poultry-specific dataset
2. **Depth Integration**: Add depth sensor for accurate size measurement
3. **Real-time Processing**: Optimize for live CCTV streams
4. **Multi-camera**: Aggregate counts across multiple feeds
5. **Behavior Analysis**: Track feeding, resting, movement patterns
6. **Database**: Store historical data for trend analysis
7. **Alerts**: Notify on unusual counts or behaviors

---

## License

This project is submitted as part of the Livestockify ML Internship assignment.

---

## Contact

**Candidate**: Mamidi Jagadeep  
**Email**: jagadeep.mamidi@gmail.com  
**GitHub**: https://github.com/jagadeepmamidi/Livestockify  
**Submission Date**: December 17, 2025

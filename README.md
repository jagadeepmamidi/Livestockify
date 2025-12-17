# Bird Counting & Weight Estimation System

A complete ML prototype for poultry farm CCTV video analysis that performs bird detection, tracking, counting, and weight estimation using YOLOv8 and SORT tracking algorithm.

## Features

- **Bird Detection**: YOLOv8-based object detection with configurable confidence thresholds
- **Multi-Object Tracking**: SORT algorithm with Kalman filtering for stable ID assignment
- **Occlusion Handling**: Robust tracking through temporary occlusions
- **Bird Counting**: Accurate count over time using unique tracking IDs
- **Weight Estimation**: Weight proxy/index (0-100 scale) based on bounding box features
- **FastAPI Service**: RESTful API for video analysis
- **Annotated Output**: Visual output with bounding boxes, tracking IDs, and count overlay

## System Requirements

- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- GPU optional (CPU works but slower)

## Installation

### 1. Clone or extract the project

```bash
cd KuppisMart
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first time you run the system, YOLOv8 will automatically download the pretrained model (~6MB).

## Quick Start

### Option 1: Run the API Server

```bash
# Start the FastAPI server
python main.py
```

The server will start on `http://localhost:8000`. You can access:
- API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Option 2: Generate and Test with Sample Video

```bash
# Create a sample test video
python create_sample_video.py

# The sample video will be saved to outputs/sample_test_video.mp4
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "OK",
  "message": "Bird counting service is running"
}
```

### Analyze Video

```bash
# Basic usage
curl -X POST "http://localhost:8000/analyze_video" \
  -F "file=@outputs/sample_test_video.mp4" \
  -o response.json

# With custom parameters
curl -X POST "http://localhost:8000/analyze_video?fps_sample=5&conf_thresh=0.5&iou_thresh=0.45" \
  -F "file=@outputs/sample_test_video.mp4" \
  -o response.json
```

**Parameters:**
- `file` (required): Video file to analyze
- `fps_sample` (optional, default=5): Process every Nth frame (1-30)
- `conf_thresh` (optional, default=0.5): Detection confidence threshold (0.1-1.0)
- `iou_thresh` (optional, default=0.45): IOU threshold for NMS (0.1-1.0)

**Response Structure:**
```json
{
  "counts": [
    {"timestamp": "00:00:00", "frame": 0, "count": 8},
    {"timestamp": "00:00:01", "frame": 30, "count": 8}
  ],
  "tracks_sample": [
    {
      "id": 1,
      "num_detections": 45,
      "first_frame": 0,
      "last_frame": 450,
      "sample_boxes": [[100, 150, 180, 220], ...]
    }
  ],
  "weight_estimates": {
    "unit": "index",
    "per_bird": [
      {
        "id": 1,
        "weight_index": 72.5,
        "confidence": 0.85,
        "num_observations": 45
      }
    ],
    "aggregate": {
      "mean": 68.3,
      "std": 12.1,
      "min": 45.2,
      "max": 89.7
    },
    "calibration_info": { ... }
  },
  "artifacts": {
    "annotated_video": "outputs/annotated_output.mp4",
    "counts_csv": "outputs/counts_timeseries.csv"
  },
  "summary_statistics": {
    "total_frames_processed": 90,
    "unique_birds_tracked": 10,
    "max_simultaneous_birds": 10,
    "avg_birds_per_frame": 9.8
  }
}
```

## Implementation Details

### Bird Counting Method

1. **Detection**: YOLOv8 nano model detects objects in each frame
2. **Tracking**: SORT algorithm assigns stable IDs using:
   - Kalman filter for motion prediction
   - Hungarian algorithm for detection-to-track association
   - IOU-based matching
3. **Counting**: Unique tracking IDs counted per frame
4. **Occlusion Handling**:
   - Kalman filter predicts position during temporary occlusions
   - Tracks maintained for up to 30 frames without detections
   - Minimum 3 hits required to establish a track (prevents false positives)
5. **ID Switch Prevention**:
   - IOU-based matching ensures consistent ID assignment
   - Motion prediction helps maintain IDs during brief occlusions

### Weight Estimation Approach

**Current Implementation (Weight Proxy/Index):**

The system outputs a **weight index (0-100 scale)** based on:
- Bounding box area (primary feature)
- Aspect ratio (confidence adjustment)
- Temporal smoothing (moving average over detections)

**Formula:**
```
normalized_area = bbox_area / reference_area
weight_index = min(100, max(0, normalized_area * 50))
confidence = max(0.3, 1.0 - |aspect_ratio - 1.2| * 0.5)
```

**Limitations:**
- Does not account for camera distance/depth
- Assumes all birds at similar distance from camera
- No breed/age classification
- Outputs relative index, not actual grams

**Converting to Actual Weight (Grams):**

To convert the weight index to actual grams, the following calibration is required:

1. **Reference Object**: Place an object of known dimensions in the video frame to establish pixel-to-cm mapping
2. **Labeled Dataset**: Collect 50-100 bird samples with known weights
3. **Feature Engineering**: Extract features including:
   - Calibrated bounding box area (cm²)
   - Bird depth from camera (requires stereo/depth sensor)
   - Pose/orientation
   - Breed/age classification
4. **Regression Model**: Train Random Forest or XGBoost on labeled data
5. **Validation**: Test on held-out set, iterate until acceptable accuracy

**See `calibration_info` in API response for detailed requirements.**

## Project Structure

```
KuppisMart/
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── create_sample_video.py # Sample video generator
├── api/
│   ├── __init__.py
│   ├── routes.py         # API endpoints
│   └── schemas.py        # Pydantic models
├── src/
│   ├── __init__.py
│   ├── detector.py       # YOLOv8 bird detection
│   ├── tracker.py        # SORT multi-object tracking
│   ├── weight_estimator.py  # Weight proxy calculation
│   └── video_processor.py   # Video processing pipeline
└── outputs/              # Generated outputs (videos, CSVs, JSONs)
```

## Output Files

After processing a video, the following files are generated in the `outputs/` directory:

1. **annotated_output.mp4**: Video with bounding boxes, tracking IDs, and count overlay
2. **counts_timeseries.csv**: Time-series count data (timestamp, frame, count)
3. **latest_analysis.json**: Complete analysis results in JSON format

## Troubleshooting

### Issue: "Could not open video"
- Ensure the video file exists and is a valid format (MP4, AVI, MOV)
- Check file permissions

### Issue: "YOLO model download fails"
- Check internet connection
- Manually download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
- Place in project root directory

### Issue: Low detection accuracy
- Increase `conf_thresh` to reduce false positives
- Decrease `conf_thresh` to detect more birds (may increase false positives)
- Consider fine-tuning YOLOv8 on poultry-specific dataset

### Issue: ID switches during tracking
- Decrease `fps_sample` to process more frames
- Adjust IOU threshold in `config.py`
- Ensure good video quality and lighting

## Performance Optimization

- **FPS Sampling**: Process every 5th frame by default (adjustable via `fps_sample`)
- **Model Size**: Using YOLOv8n (nano) for speed; upgrade to YOLOv8m for better accuracy
- **GPU Acceleration**: Automatically uses GPU if available (CUDA)
- **Batch Processing**: Can process multiple videos sequentially

## Future Enhancements

1. **Fine-tuned Model**: Train YOLOv8 on poultry-specific dataset
2. **Depth Estimation**: Integrate depth sensor for accurate size measurement
3. **Behavior Analysis**: Track movement patterns, feeding behavior
4. **Real-time Processing**: Optimize for live CCTV stream analysis
5. **Multi-camera Support**: Aggregate counts across multiple camera feeds
6. **Database Integration**: Store historical data for trend analysis

## Dataset Information

This system is designed to work with any fixed-camera poultry farm video. For training/evaluation, consider:

- **Kaggle**: Search for "chicken detection" or "poultry farm" datasets
- **Roboflow Universe**: Pre-annotated poultry datasets
- **Custom Data**: Annotate your own farm footage using tools like LabelImg or CVAT

## License

This project is submitted as part of the Livestockify ML Intern assignment.

## Contact

For questions or issues, please contact: jagadeep.mamidi@gmail.com

---



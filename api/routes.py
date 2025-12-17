"""
FastAPI routes for bird counting and weight estimation service.
"""
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from pathlib import Path
import shutil
import json

from api.schemas import VideoAnalysisResponse, HealthResponse
from src.video_processor import VideoProcessor
import config

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with status OK
    """
    return HealthResponse(
        status="OK",
        message="Bird counting service is running"
    )


@router.post("/analyze_video", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(..., description="Video file to analyze"),
    fps_sample: int = Query(
        default=config.DEFAULT_FPS_SAMPLE,
        description="Sample every Nth frame (default: 5)",
        ge=1,
        le=30
    ),
    conf_thresh: float = Query(
        default=config.CONFIDENCE_THRESHOLD,
        description="Confidence threshold for detection (default: 0.5)",
        ge=0.1,
        le=1.0
    ),
    iou_thresh: float = Query(
        default=config.IOU_THRESHOLD,
        description="IOU threshold for NMS (default: 0.45)",
        ge=0.1,
        le=1.0
    )
):
    """
    Analyze video to detect, track, and count birds with weight estimation.
    
    Args:
        file: Uploaded video file
        fps_sample: Sample every Nth frame for processing
        conf_thresh: Confidence threshold for bird detection
        iou_thresh: IOU threshold for non-maximum suppression
        
    Returns:
        VideoAnalysisResponse with counts, tracks, weight estimates, and artifacts
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a video file."
        )
    
    # Save uploaded file
    temp_video_path = config.TEMP_DIR / file.filename
    try:
        with temp_video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )
    
    # Process video
    try:
        processor = VideoProcessor()
        result = processor.process_video(
            video_path=str(temp_video_path),
            fps_sample=fps_sample,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh
        )
        
        # Save result to JSON
        result_json_path = config.OUTPUTS_DIR / "latest_analysis.json"
        with result_json_path.open("w") as f:
            json.dump(result, f, indent=2)
        
        return VideoAnalysisResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )
    finally:
        # Clean up temp file
        if temp_video_path.exists():
            temp_video_path.unlink()

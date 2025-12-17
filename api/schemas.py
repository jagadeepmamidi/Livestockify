"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class CountData(BaseModel):
    """Count data for a single timestamp."""
    timestamp: str
    frame: int
    count: int


class TrackSample(BaseModel):
    """Sample track information."""
    id: int
    num_detections: int
    first_frame: int
    last_frame: int
    sample_boxes: List[List[float]]


class PerBirdWeight(BaseModel):
    """Weight estimate for a single bird."""
    id: int
    weight_index: float
    confidence: float
    num_observations: int


class AggregateStats(BaseModel):
    """Aggregate weight statistics."""
    mean: float
    std: float
    min: float
    max: float


class WeightEstimates(BaseModel):
    """Weight estimation results."""
    unit: str
    per_bird: List[PerBirdWeight]
    aggregate: AggregateStats
    calibration_info: Dict


class Artifacts(BaseModel):
    """Generated artifacts."""
    annotated_video: str
    counts_csv: str


class SummaryStatistics(BaseModel):
    """Summary statistics."""
    total_frames_processed: int
    unique_birds_tracked: int
    max_simultaneous_birds: int
    avg_birds_per_frame: float


class VideoAnalysisResponse(BaseModel):
    """Response from video analysis endpoint."""
    counts: List[CountData]
    tracks_sample: List[TrackSample]
    weight_estimates: WeightEstimates
    artifacts: Artifacts
    summary_statistics: SummaryStatistics


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "OK"
    message: str = "Bird counting service is running"

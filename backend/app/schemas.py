from typing import Any

from pydantic import BaseModel


class ProbabilityItem(BaseModel):
    label: str
    probability: float


class TimelinePoint(BaseModel):
    time: float
    label: str
    confidence: float
    probabilities: list[ProbabilityItem]


class FrameInsight(BaseModel):
    time: float
    label: str
    confidence: float
    image_base64: str
    heatmap_base64: str | None = None


class VideoMetadata(BaseModel):
    filename: str
    duration_seconds: float
    fps: float
    frame_count: int
    sampled_frames: int


class PredictionSummary(BaseModel):
    label: str
    confidence: float


class AnalysisResponse(BaseModel):
    request_id: str
    metadata: VideoMetadata
    prediction: PredictionSummary
    top_predictions: list[ProbabilityItem]
    breakdown: list[ProbabilityItem]
    timeline: list[TimelinePoint]
    key_frames: list[FrameInsight]
    insights: list[str]
    similarity_score: float | None = None
    comparison_summary: str | None = None
    report: dict[str, Any]

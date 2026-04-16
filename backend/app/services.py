from __future__ import annotations

import base64
import io
import math
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from .schemas import AnalysisResponse, FrameInsight, PredictionSummary, ProbabilityItem, TimelinePoint, VideoMetadata

CLASS_NAMES = [
    "Cover Drive",
    "Defense",
    "Flick",
    "Hook",
    "Late Cut",
    "Lofted Drive",
    "Pull Shot",
    "Square Cut",
    "Straight Drive",
    "Sweep",
]
MODEL_FRAME_COUNT = 30
ANALYSIS_FRAME_COUNT = 60
TIMELINE_WINDOWS = 8
KEYFRAME_COUNT = 3
MODEL_SIZE = (224, 224)
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_WEIGHTS_PATH = ROOT_DIR / "model_weights.h5"


def _build_model() -> tf.keras.Model:
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential(
        [
            layers.TimeDistributed(base_model, input_shape=(None, 224, 224, 3)),
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            layers.GRU(256, return_sequences=True),
            layers.GRU(128),
            layers.Dense(1024, activation="relu", name="embedding"),
            layers.Dropout(0.5),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model


@lru_cache(maxsize=1)
def get_model_bundle() -> tuple[tf.keras.Model, tf.keras.Model]:
    model = _build_model()
    feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("embedding").output)
    return model, feature_model


def _safe_fps(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    return fps if fps > 1 else 25.0


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    tensor = tf.image.convert_image_dtype(frame, tf.uint8)
    tensor = tf.image.resize_with_pad(tensor, *MODEL_SIZE)
    return tensor.numpy()


def _encode_image(frame_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(buffer).decode("utf-8")


def _overlay_heatmap(frame_bgr: np.ndarray, saliency_map: np.ndarray) -> str:
    heatmap = np.uint8(np.clip(saliency_map, 0, 1) * 255)
    heatmap = cv2.resize(heatmap, (frame_bgr.shape[1], frame_bgr.shape[0]))
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
    blended = cv2.addWeighted(frame_bgr, 0.55, colored, 0.45, 0)
    return _encode_image(blended)


def _uniform_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        return [0] * sample_count
    if frame_count < sample_count:
        base = list(range(frame_count))
        while len(base) < sample_count:
            base.append(base[-1])
        return base
    return np.linspace(0, frame_count - 1, sample_count, dtype=int).tolist()


def _read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    return frame if ok else None


def extract_video_data(video_path: str | Path, sample_count: int = ANALYSIS_FRAME_COUNT) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Unable to read the uploaded video.")

    fps = _safe_fps(capture)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _uniform_indices(max(frame_count, 1), sample_count if frame_count else MODEL_FRAME_COUNT)

    raw_frames: list[np.ndarray] = []
    processed_frames: list[np.ndarray] = []
    timestamps: list[float] = []

    for index in indices:
        frame = _read_frame(capture, index)
        if frame is None:
            if raw_frames:
                frame = raw_frames[-1].copy()
            else:
                frame = np.zeros((MODEL_SIZE[0], MODEL_SIZE[1], 3), dtype=np.uint8)
        raw_frames.append(frame)
        processed_frames.append(_resize_frame(frame)[..., [2, 1, 0]])
        timestamps.append(index / fps)

    capture.release()

    duration_seconds = (frame_count / fps) if frame_count else (len(raw_frames) / fps)
    return {
        "raw_frames": np.array(raw_frames),
        "processed_frames": np.array(processed_frames, dtype=np.uint8),
        "timestamps": timestamps,
        "frame_count": frame_count if frame_count else len(raw_frames),
        "fps": fps,
        "duration_seconds": duration_seconds,
    }


def _ensure_length(frames: np.ndarray, length: int = MODEL_FRAME_COUNT) -> np.ndarray:
    if len(frames) >= length:
        if len(frames) == length:
            return frames
        indices = np.linspace(0, len(frames) - 1, length, dtype=int)
        return frames[indices]

    pad_frame = frames[-1] if len(frames) else np.zeros((MODEL_SIZE[0], MODEL_SIZE[1], 3), dtype=np.uint8)
    padded = list(frames)
    while len(padded) < length:
        padded.append(pad_frame.copy())
    return np.array(padded)


def _batch_predict(model: tf.keras.Model, sequences: np.ndarray) -> np.ndarray:
    return model.predict(sequences, verbose=0)


def _batch_features(feature_model: tf.keras.Model, sequences: np.ndarray) -> np.ndarray:
    return feature_model.predict(sequences, verbose=0)


def _window_starts(total_frames: int) -> list[int]:
    if total_frames <= MODEL_FRAME_COUNT:
        return [0]
    max_start = total_frames - MODEL_FRAME_COUNT
    count = min(TIMELINE_WINDOWS, max_start + 1)
    return np.linspace(0, max_start, count, dtype=int).tolist()


def _to_probability_items(probabilities: np.ndarray) -> list[ProbabilityItem]:
    ordered = np.argsort(probabilities)[::-1]
    return [
        ProbabilityItem(label=CLASS_NAMES[index], probability=float(probabilities[index] * 100))
        for index in ordered
    ]


def _timeline_points(processed_frames: np.ndarray, timestamps: list[float]) -> tuple[list[TimelinePoint], list[dict[str, int]], np.ndarray]:
    model, _ = get_model_bundle()
    starts = _window_starts(len(processed_frames))
    sequences = np.array([processed_frames[start : start + MODEL_FRAME_COUNT] for start in starts])
    predictions = _batch_predict(model, sequences)

    points: list[TimelinePoint] = []
    window_meta: list[dict[str, int]] = []
    for start, probabilities in zip(starts, predictions):
        center_index = min(start + math.ceil(MODEL_FRAME_COUNT / 2) - 1, len(timestamps) - 1)
        items = _to_probability_items(probabilities)
        top_item = items[0]
        points.append(
            TimelinePoint(
                time=round(float(timestamps[center_index]), 2),
                label=top_item.label,
                confidence=round(top_item.probability, 2),
                probabilities=items[:3],
            )
        )
        window_meta.append({"start": start, "center_index": center_index})

    return points, window_meta, predictions


def _video_summary(processed_frames: np.ndarray) -> tuple[np.ndarray, PredictionSummary, list[ProbabilityItem], list[ProbabilityItem]]:
    model, feature_model = get_model_bundle()
    sequence = np.expand_dims(_ensure_length(processed_frames), axis=0)
    probabilities = _batch_predict(model, sequence)[0]
    summary_items = _to_probability_items(probabilities)
    prediction = PredictionSummary(label=summary_items[0].label, confidence=round(summary_items[0].probability, 2))
    return _batch_features(feature_model, sequence)[0], prediction, summary_items[:3], summary_items


def _saliency_for_window(model: tf.keras.Model, sequence: np.ndarray, class_index: int) -> np.ndarray:
    tensor = tf.convert_to_tensor(sequence[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(tensor)
        predictions = model(tensor, training=False)
        score = predictions[:, class_index]
    gradients = tape.gradient(score, tensor)
    if gradients is None:
        return np.zeros((sequence.shape[0], MODEL_SIZE[0], MODEL_SIZE[1]), dtype=np.float32)
    saliency = tf.reduce_mean(tf.abs(gradients), axis=-1)[0].numpy()
    saliency_min = saliency.min()
    saliency_max = saliency.max()
    if saliency_max - saliency_min < 1e-8:
        return np.zeros_like(saliency)
    return (saliency - saliency_min) / (saliency_max - saliency_min)


def _key_frames(
    raw_frames: np.ndarray,
    processed_frames: np.ndarray,
    timeline: list[TimelinePoint],
    window_meta: list[dict[str, int]],
) -> list[FrameInsight]:
    model, _ = get_model_bundle()
    sorted_indices = sorted(range(len(timeline)), key=lambda idx: timeline[idx].confidence, reverse=True)
    selected: list[int] = []
    for index in sorted_indices:
        center = window_meta[index]["center_index"]
        if all(abs(center - window_meta[chosen]["center_index"]) > 5 for chosen in selected):
            selected.append(index)
        if len(selected) == KEYFRAME_COUNT:
            break
    if not selected:
        selected = [0]

    keyframes: list[FrameInsight] = []
    for timeline_index in selected:
        point = timeline[timeline_index]
        window_start = window_meta[timeline_index]["start"]
        center_index = window_meta[timeline_index]["center_index"]
        sequence = processed_frames[window_start : window_start + MODEL_FRAME_COUNT]
        class_index = CLASS_NAMES.index(point.label)
        saliency = _saliency_for_window(model, sequence, class_index)
        local_center = min(center_index - window_start, len(saliency) - 1)

        frame_bgr = raw_frames[center_index]
        keyframes.append(
            FrameInsight(
                time=point.time,
                label=point.label,
                confidence=point.confidence,
                image_base64=_encode_image(frame_bgr),
                heatmap_base64=_overlay_heatmap(frame_bgr, saliency[local_center]),
            )
        )

    return keyframes


def _consistency_ratio(timeline: list[TimelinePoint], label: str) -> float:
    if not timeline:
        return 0.0
    consistent = sum(1 for point in timeline if point.label == label)
    return consistent / len(timeline)


def _motion_score(raw_frames: np.ndarray) -> float:
    if len(raw_frames) < 2:
        return 0.0
    diffs = []
    for current, nxt in zip(raw_frames[:-1], raw_frames[1:]):
        diffs.append(float(np.mean(cv2.absdiff(current, nxt))))
    return float(np.mean(diffs))


def _build_insights(prediction: PredictionSummary, top_predictions: list[ProbabilityItem], timeline: list[TimelinePoint], raw_frames: np.ndarray) -> list[str]:
    consistency = _consistency_ratio(timeline, prediction.label)
    motion = _motion_score(raw_frames)
    challenger = top_predictions[1] if len(top_predictions) > 1 else top_predictions[0]
    confidence_band = "strong" if prediction.confidence >= 80 else "moderate" if prediction.confidence >= 60 else "uncertain"
    movement = "high-tempo" if motion >= 25 else "controlled"
    return [
        f"The model reads this clip as {prediction.label} with {confidence_band} confidence at {prediction.confidence:.1f}%.",
        f"Timeline consistency is {consistency * 100:.1f}%, which indicates how stable the shot pattern remained across the sampled sequence.",
        f"The main alternative class is {challenger.label} at {challenger.probability:.1f}%, useful for understanding ambiguity in similar bat paths.",
        f"Frame-to-frame motion suggests a {movement} shot profile, based on average visual change across sampled frames.",
    ]


def _similarity_percentage(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    numerator = float(np.dot(vector_a, vector_b))
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0:
        return 0.0
    cosine = max(min(numerator / denominator, 1.0), -1.0)
    return round(((cosine + 1) / 2) * 100, 2)


def analyze_video(video_path: str | Path, filename: str) -> tuple[AnalysisResponse, np.ndarray]:
    video_data = extract_video_data(video_path)
    raw_frames = video_data["raw_frames"]
    processed_frames = video_data["processed_frames"]
    timestamps = video_data["timestamps"]

    embedding, prediction, top_predictions, breakdown = _video_summary(processed_frames)
    timeline, window_meta, _ = _timeline_points(processed_frames, timestamps)
    key_frames = _key_frames(raw_frames, processed_frames, timeline, window_meta)
    insights = _build_insights(prediction, top_predictions, timeline, raw_frames)

    response = AnalysisResponse(
        request_id=str(uuid.uuid4()),
        metadata=VideoMetadata(
            filename=filename,
            duration_seconds=round(float(video_data["duration_seconds"]), 2),
            fps=round(float(video_data["fps"]), 2),
            frame_count=int(video_data["frame_count"]),
            sampled_frames=len(processed_frames),
        ),
        prediction=prediction,
        top_predictions=top_predictions,
        breakdown=breakdown,
        timeline=timeline,
        key_frames=key_frames,
        insights=insights,
        report={
            "title": f"Cricket Shot Analysis - {filename}",
            "summary": prediction.model_dump(),
            "top_predictions": [item.model_dump() for item in top_predictions],
            "insights": insights,
        },
    )
    return response, embedding


def compare_analyses(first: AnalysisResponse, first_embedding: np.ndarray, second: AnalysisResponse, second_embedding: np.ndarray) -> tuple[AnalysisResponse, AnalysisResponse, float, str]:
    similarity = _similarity_percentage(first_embedding, second_embedding)
    pair_summary = (
        "Highly similar batting mechanics detected."
        if similarity >= 85
        else "Noticeable overlap with some variation in timing or follow-through."
        if similarity >= 65
        else "Distinct shot mechanics or shot classes detected."
    )

    first.similarity_score = similarity
    second.similarity_score = similarity
    first.comparison_summary = pair_summary
    second.comparison_summary = pair_summary
    return first, second, similarity, pair_summary


def create_pdf_report(payload: dict) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.rect(0, 0, width, height, fill=1, stroke=0)

    pdf.setFillColor(colors.HexColor("#f8fafc"))
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(18 * mm, height - 24 * mm, "Cricket Shot Recognition Report")

    pdf.setFillColor(colors.HexColor("#94a3b8"))
    pdf.setFont("Helvetica", 11)
    pdf.drawString(18 * mm, height - 31 * mm, payload.get("generatedAt", "Generated locally"))

    pdf.setFillColor(colors.HexColor("#e2e8f0"))
    pdf.roundRect(16 * mm, height - 78 * mm, width - 32 * mm, 34 * mm, 8, fill=0, stroke=1)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(22 * mm, height - 54 * mm, payload.get("title", "Analysis Summary"))

    summary = payload.get("summary", {})
    pdf.setFont("Helvetica", 12)
    pdf.drawString(22 * mm, height - 62 * mm, f"Predicted shot: {summary.get('label', 'N/A')}")
    pdf.drawString(95 * mm, height - 62 * mm, f"Confidence: {summary.get('confidence', 0):.2f}%")

    y = height - 94 * mm
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(18 * mm, y, "Top Predictions")
    y -= 8 * mm

    pdf.setFont("Helvetica", 11)
    for item in payload.get("topPredictions", [])[:5]:
        pdf.drawString(22 * mm, y, f"{item.get('label', 'Unknown')}: {item.get('probability', 0):.2f}%")
        y -= 6 * mm

    y -= 4 * mm
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(18 * mm, y, "Insights")
    y -= 8 * mm
    pdf.setFont("Helvetica", 11)

    for insight in payload.get("insights", [])[:5]:
        wrapped = []
        current = ""
        for word in insight.split():
            candidate = f"{current} {word}".strip()
            if pdf.stringWidth(candidate, "Helvetica", 11) <= width - 42 * mm:
                current = candidate
            else:
                wrapped.append(current)
                current = word
        if current:
            wrapped.append(current)

        for line in wrapped:
            pdf.drawString(22 * mm, y, f"- {line}" if line == wrapped[0] else f"  {line}")
            y -= 5 * mm
            if y < 25 * mm:
                pdf.showPage()
                pdf.setFillColor(colors.HexColor("#0f172a"))
                pdf.rect(0, 0, width, height, fill=1, stroke=0)
                pdf.setFillColor(colors.HexColor("#f8fafc"))
                y = height - 24 * mm

        y -= 2 * mm

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.read()


def save_upload(upload, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(upload.file.read())
        return tmp_file.name

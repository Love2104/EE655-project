from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from torchvision import models, transforms


CLASSES = [
    "cover",
    "defense",
    "flick",
    "hook",
    "late_cut",
    "lofted",
    "pull",
    "square_cut",
    "straight",
    "sweep",
]
FRAME_SIZE = (224, 224)
NUM_FRAMES = 16
NUM_CLASSES = len(CLASSES)
HIDDEN_SIZE = 256
DROPOUT = 0.65
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "results_current" / "checkpoints" / "p1_cnn_only.pth",
    PROJECT_ROOT / "results_current" / "checkpoints" / "p1_gru.pth",
    PROJECT_ROOT / "checkpoints" / "p1_cnn_only.pth",
]


def find_checkpoint() -> Path | None:
    for path in DEFAULT_CHECKPOINT_CANDIDATES:
        if path.exists():
            return path

    candidates = sorted(PROJECT_ROOT.rglob("*.pth"))
    for candidate in candidates:
        if ".venv" not in candidate.parts:
            return candidate
    return None


def infer_architecture(checkpoint_path: str | Path) -> str:
    name = Path(checkpoint_path).name.lower()
    if "gru" in name:
        return "gru"
    if "lstm" in name:
        return "lstm"
    return "cnn_only"


def extract_frames(path: str | Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1:
        fps = 25.0

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, FRAME_SIZE), cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def uniform_sampling(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    if not frames:
        return [np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)] * n
    if len(frames) <= n:
        return frames + [frames[-1]] * (n - len(frames))
    return [frames[i] for i in np.linspace(0, len(frames) - 1, n, dtype=int)]


def compute_motion_scores(frames: list[np.ndarray]) -> np.ndarray:
    scores = [0.0]
    for i in range(1, len(frames)):
        g1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        scores.append(float(np.mean(np.abs(g2 - g1))))
    return np.array(scores)


def motion_sampling(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    if not frames:
        return [np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)] * n
    if len(frames) <= n:
        return frames + [frames[-1]] * (n - len(frames))

    scores = compute_motion_scores(frames)
    top_n = min(n, len(frames))
    top_indices = set(np.argsort(scores)[-top_n:])

    context_indices = set()
    for idx in top_indices:
        for offset in (-3, -2, -1, 1, 2, 3):
            neighbor = idx + offset
            if 0 <= neighbor < len(frames):
                context_indices.add(neighbor)

    all_indices = top_indices | context_indices
    if len(all_indices) < n:
        remaining = n - len(all_indices)
        all_indices |= set(np.linspace(0, len(frames) - 1, remaining, dtype=int))

    result_indices = sorted(all_indices)[:n]
    return [frames[i] for i in result_indices]


def hybrid_sampling(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    if not frames:
        return [np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)] * n
    if len(frames) <= n:
        return frames + [frames[-1]] * (n - len(frames))

    motion_n = int(n * 0.4)
    uniform_n = int(n * 0.4)
    random_n = n - motion_n - uniform_n
    scores = compute_motion_scores(frames)

    motion_indices = set(np.argsort(scores)[-motion_n:])
    context_indices = set()
    for idx in motion_indices:
        for offset in (-2, -1, 1, 2):
            neighbor = idx + offset
            if 0 <= neighbor < len(frames):
                context_indices.add(neighbor)
    motion_indices |= context_indices

    uniform_indices = set(np.linspace(0, len(frames) - 1, uniform_n, dtype=int))
    remaining_indices = [i for i in range(len(frames)) if i not in motion_indices and i not in uniform_indices]
    random_indices: set[int] = set()
    if remaining_indices and random_n > 0:
        random_indices = set(np.random.choice(remaining_indices, min(random_n, len(remaining_indices)), replace=False))

    all_indices = sorted(motion_indices | uniform_indices | random_indices)
    while len(all_indices) < n:
        all_indices.append(all_indices[-1])

    return [frames[i] for i in all_indices[:n]]


def sample_frames(frames: list[np.ndarray], n: int, strategy: str) -> list[np.ndarray]:
    if strategy == "uniform":
        return uniform_sampling(frames, n)
    if strategy == "motion":
        return motion_sampling(frames, n)
    if strategy == "hybrid":
        return hybrid_sampling(frames, n)
    raise ValueError(f"Unknown strategy: {strategy}")


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(FRAME_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class EfficientEncoder(nn.Module):
    def __init__(self, fine_tune_blocks: int = 3, freeze_bn: bool = False):
        super().__init__()
        base = models.efficientnet_b0(weights=None)

        for p in base.parameters():
            p.requires_grad = False

        blocks = list(base.features.children())
        for block in blocks[-fine_tune_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

        if freeze_bn:
            for module in base.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.pool(x).flatten(1)


class CricketLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = EfficientEncoder(fine_tune_blocks=2, freeze_bn=True)
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES),
        )

    def _extract_sequence(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        return self.cnn(x.view(batch * steps, channels, height, width)).view(batch, steps, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._extract_sequence(x)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1])

    def sequence_features(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._extract_sequence(x)
        out, _ = self.lstm(seq)
        return out[:, -1]


class CricketGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = EfficientEncoder(fine_tune_blocks=2, freeze_bn=True)
        self.gru = nn.GRU(
            input_size=1280,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES),
        )

    def _extract_sequence(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        return self.cnn(x.view(batch * steps, channels, height, width)).view(batch, steps, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._extract_sequence(x)
        out, _ = self.gru(seq)
        return self.head(out.mean(dim=1))

    def sequence_features(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._extract_sequence(x)
        out, _ = self.gru(seq)
        return out.mean(dim=1)


class CNNOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = EfficientEncoder(fine_tune_blocks=3)
        self.head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES),
        )

    def _extract_sequence(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        return self.cnn(x.view(batch * steps, channels, height, width)).view(batch, steps, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._extract_sequence(x)
        pooled = seq.mean(dim=1)
        return self.head(pooled)

    def sequence_features(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._extract_sequence(x)
        return seq.mean(dim=1)


def get_model(arch: str) -> nn.Module:
    if arch == "cnn_only":
        return CNNOnly()
    if arch == "gru":
        return CricketGRU()
    if arch == "lstm":
        return CricketLSTM()
    raise ValueError(f"Unsupported architecture: {arch}")


@dataclass
class NotebookModelBundle:
    model: nn.Module
    transform: transforms.Compose
    checkpoint_path: Path
    strategy: str
    architecture: str


def load_bundle(checkpoint_path: str | Path | None = None, strategy: str = "uniform") -> NotebookModelBundle:
    resolved = Path(checkpoint_path) if checkpoint_path else find_checkpoint()
    if resolved is None or not resolved.exists():
        raise FileNotFoundError("No PyTorch checkpoint was found from the notebook results.")

    architecture = infer_architecture(resolved)
    model = get_model(architecture).to(DEVICE)
    state_dict = torch.load(resolved, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return NotebookModelBundle(
        model=model,
        transform=build_transform(),
        checkpoint_path=resolved,
        strategy=strategy,
        architecture=architecture,
    )


def frames_to_tensor(frames: list[np.ndarray], transform: transforms.Compose) -> torch.Tensor:
    return torch.stack([transform(frame) for frame in frames]).unsqueeze(0).to(DEVICE)


def _image_to_base64(frame_rgb: np.ndarray) -> str:
    image = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    value = float(np.dot(a, b) / denom)
    value = max(min(value, 1.0), -1.0)
    return round(((value + 1.0) / 2.0) * 100.0, 2)


def _entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def _window_indices(frame_count: int, n_frames: int, windows: int = 6) -> list[tuple[int, int]]:
    if frame_count <= n_frames:
        return [(0, frame_count)]
    max_start = frame_count - n_frames
    starts = np.linspace(0, max_start, min(windows, max_start + 1), dtype=int).tolist()
    return [(start, start + n_frames) for start in starts]


def _predict_probabilities(bundle: NotebookModelBundle, frames: list[np.ndarray]) -> np.ndarray:
    tensor = frames_to_tensor(frames, bundle.transform)
    return torch.softmax(bundle.model(tensor), dim=1).squeeze(0).cpu().numpy()


def _aggregate_probabilities(
    frames: list[np.ndarray],
    bundle: NotebookModelBundle,
    single_probabilities: np.ndarray,
    voting_strategy: str,
) -> tuple[np.ndarray, dict[str, float]]:
    if voting_strategy == "single":
        return single_probabilities, {"source": "single_clip"}

    if voting_strategy == "majority":
        votes = np.zeros(len(CLASSES), dtype=np.float32)
        for _ in range(5):
            sampled = sample_frames(frames, NUM_FRAMES, bundle.strategy)
            probabilities = _predict_probabilities(bundle, sampled[:NUM_FRAMES])
            votes[int(np.argmax(probabilities))] += 1.0
        total_votes = float(votes.sum())
        if total_votes <= 0:
            return single_probabilities, {"source": "single_clip"}
        return votes / total_votes, {"source": "majority_vote"}

    if voting_strategy == "weighted":
        scores = compute_motion_scores(frames)
        weighted_sum = np.zeros(len(CLASSES), dtype=np.float32)
        total_weight = 0.0
        for _ in range(5):
            top_n = min(len(frames), max(NUM_FRAMES, int(NUM_FRAMES * 1.5)))
            if top_n <= 0:
                continue
            chosen_pool = np.argsort(scores)[-top_n:]
            sample_size = min(NUM_FRAMES, len(chosen_pool))
            if sample_size <= 0:
                continue
            chosen = np.sort(np.random.choice(chosen_pool, size=sample_size, replace=False))
            sampled = [frames[index] for index in chosen]
            while len(sampled) < NUM_FRAMES:
                sampled.append(sampled[-1])
            weight = float(scores[chosen].mean() + 1e-6)
            weighted_sum += weight * _predict_probabilities(bundle, sampled[:NUM_FRAMES])
            total_weight += weight
        if total_weight <= 0:
            return single_probabilities, {"source": "single_clip"}
        return weighted_sum / total_weight, {"source": "weighted_vote"}

    raise ValueError(f"Unknown voting strategy: {voting_strategy}")


@torch.no_grad()
def analyze_video(video_path: str | Path, bundle: NotebookModelBundle, voting_strategy: str = "single") -> dict:
    frames, fps = extract_frames(video_path)
    if not frames:
        raise ValueError("Unable to read frames from the uploaded video.")

    sampled = sample_frames(frames, NUM_FRAMES, bundle.strategy)
    tensor = frames_to_tensor(sampled, bundle.transform)
    logits = bundle.model(tensor)
    single_probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    feature_vector = bundle.model.sequence_features(tensor).squeeze(0).cpu().numpy()

    windows: list[dict[str, float | int | np.ndarray]] = []
    timeline = []
    for start, end in _window_indices(len(frames), NUM_FRAMES):
        clip = sample_frames(frames[start:end], NUM_FRAMES, bundle.strategy)
        clip_tensor = frames_to_tensor(clip, bundle.transform)
        clip_probs = torch.softmax(bundle.model(clip_tensor), dim=1).squeeze(0).cpu().numpy()
        label_index = int(np.argmax(clip_probs))
        entropy = _entropy(clip_probs)
        midpoint = (start + min(end, len(frames)) - 1) / 2.0
        windows.append(
            {
                "label_index": label_index,
                "confidence": float(clip_probs[label_index] * 100),
                "entropy": entropy,
                "probabilities": clip_probs,
            }
        )
        timeline.append(
            {
                "time": round(float(midpoint / fps), 2),
                "label": CLASSES[label_index],
                "confidence": round(float(clip_probs[label_index] * 100), 2),
            }
        )

    final_probabilities, voting_meta = _aggregate_probabilities(frames, bundle, single_probabilities, voting_strategy)
    ordered = np.argsort(final_probabilities)[::-1]
    top_predictions = [
        {"label": CLASSES[index], "probability": round(float(final_probabilities[index] * 100), 2)}
        for index in ordered[:3]
    ]
    breakdown = [
        {"label": CLASSES[index], "probability": round(float(final_probabilities[index] * 100), 2)}
        for index in ordered
    ]

    key_frames = []
    key_indices = np.linspace(0, len(sampled) - 1, min(4, len(sampled)), dtype=int).tolist()
    top_label = CLASSES[int(np.argmax(final_probabilities))]
    top_conf = round(float(np.max(final_probabilities) * 100), 2)
    for idx in key_indices:
        key_frames.append(
            {
                "time": round(float(idx / fps), 2),
                "label": top_label,
                "confidence": top_conf,
                "image_base64": _image_to_base64(sampled[idx]),
            }
        )

    motion_scores = compute_motion_scores(frames)
    duration = round(float(len(frames) / fps), 2)
    insights = [
        f"Notebook pipeline active: EfficientNetB0 + {bundle.architecture.upper()} with {bundle.strategy} sampling.",
        f"Voting strategy in use: {voting_strategy} ({voting_meta['source']}).",
        f"Predicted shot is {top_label} at {top_conf:.2f}% confidence using {NUM_FRAMES} sampled frames.",
        f"Video duration is {duration:.2f}s at {fps:.2f} FPS with mean motion score {float(np.mean(motion_scores)):.2f}.",
        f"Checkpoint in use: {bundle.checkpoint_path.name}.",
    ]

    return {
        "metadata": {
            "filename": Path(video_path).name,
            "frame_count": len(frames),
            "sampled_frames": len(sampled),
            "window_count": len(windows),
            "checkpoint": str(bundle.checkpoint_path),
            "strategy": bundle.strategy,
            "architecture": bundle.architecture,
            "voting_strategy": voting_strategy,
            "fps": round(fps, 2),
            "duration_seconds": duration,
        },
        "prediction": {"label": top_label, "confidence": top_conf},
        "top_predictions": top_predictions,
        "breakdown": breakdown,
        "timeline": timeline,
        "key_frames": key_frames,
        "insights": insights,
        "feature_vector": feature_vector.tolist(),
    }


def compare_analyses(first: dict, second: dict) -> dict:
    score = _cosine_similarity(np.array(first["feature_vector"]), np.array(second["feature_vector"]))
    summary = (
        "Highly similar batting mechanics detected."
        if score >= 85
        else "Noticeable overlap with some variation in execution."
        if score >= 65
        else "Distinct shot mechanics detected."
    )
    return {
        "video_a": first,
        "video_b": second,
        "similarity_score": score,
        "comparison_summary": summary,
    }


def create_pdf_report(payload: dict) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFillColor(colors.HexColor("#102117"))
    pdf.rect(0, 0, width, height, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#f3f7f2"))
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(18 * mm, height - 22 * mm, payload.get("title", "Cricket Shot Report"))

    pdf.setFont("Helvetica", 11)
    pdf.setFillColor(colors.HexColor("#d7e2d8"))
    pdf.drawString(18 * mm, height - 30 * mm, payload.get("generatedAt", "Generated locally"))

    y = height - 44 * mm
    summary = payload.get("summary", {})
    pdf.setFillColor(colors.HexColor("#f3f7f2"))
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(18 * mm, y, "Summary")
    y -= 8 * mm
    pdf.setFont("Helvetica", 11)
    pdf.drawString(22 * mm, y, f"Prediction: {summary.get('label', 'N/A')}")
    y -= 6 * mm
    pdf.drawString(22 * mm, y, f"Confidence: {summary.get('confidence', 0):.2f}%")
    y -= 6 * mm
    if summary.get("architecture"):
        pdf.drawString(22 * mm, y, f"Model: {summary.get('architecture')}")
        y -= 6 * mm
    if summary.get("sampling"):
        pdf.drawString(22 * mm, y, f"Sampling: {summary.get('sampling')}")
        y -= 6 * mm
    if summary.get("voting"):
        pdf.drawString(22 * mm, y, f"Voting: {summary.get('voting')}")
        y -= 8 * mm
    else:
        y -= 4 * mm

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
    for insight in payload.get("insights", [])[:6]:
        words = insight.split()
        current = ""
        lines: list[str] = []
        for word in words:
            trial = f"{current} {word}".strip()
            if pdf.stringWidth(trial, "Helvetica", 11) < width - 44 * mm:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        for line in lines:
            pdf.drawString(22 * mm, y, line)
            y -= 5 * mm
        y -= 2 * mm
        if y < 24 * mm:
            pdf.showPage()
            pdf.setFillColor(colors.HexColor("#102117"))
            pdf.rect(0, 0, width, height, fill=1, stroke=0)
            pdf.setFillColor(colors.HexColor("#f3f7f2"))
            y = height - 24 * mm

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.read()

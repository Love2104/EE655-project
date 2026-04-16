from __future__ import annotations

import base64
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from cricket_notebook_model import analyze_video, compare_analyses, create_pdf_report, find_checkpoint, load_bundle


SUPPORTED_TYPES = ["mp4", "avi", "mov", "mkv"]
SAMPLING_OPTIONS = ["uniform", "motion", "hybrid"]
VOTING_OPTIONS = ["single", "majority", "weighted"]
RESULTS_DIR = Path("results_unzipped") / "results"
CHECKPOINT_DIR = Path("results_unzipped") / "checkpoints"
MODEL_LABELS = {
    "cnn_only": "CNN Only",
    "gru": "GRU",
    "lstm": "LSTM",
}
SAMPLING_LABELS = {
    "uniform": "Uniform",
    "motion": "Motion",
    "hybrid": "Hybrid",
}
VOTING_LABELS = {
    "single": "Single Clip",
    "majority": "Majority Voting",
    "weighted": "Weighted Voting",
}


st.set_page_config(page_title="Cricket Shot Detection", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root {
        --bg: #f5f6f8;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --border: #d8dee6;
        --text: #18212b;
        --muted: #61707d;
        --accent: #1f4e79;
        --sidebar: #1f2430;
        --sidebar-text: #f5f7fb;
    }
    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    .block-container {
        max-width: 1480px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    .hero {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 28px rgba(15, 23, 42, 0.04);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.3rem;
        line-height: 1.1;
        color: #142235;
    }
    .hero p {
        margin: 0.7rem 0 0;
        max-width: 900px;
        color: #586573;
        line-height: 1.65;
        font-size: 1rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1rem 0 0.5rem;
        color: #132235;
    }
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }
    .bar-track {
        width: 100%;
        height: 0.65rem;
        background: #e8edf3;
        border-radius: 999px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        background: #1f4e79;
        border-radius: 999px;
    }
    .prob-row {
        display: grid;
        grid-template-columns: minmax(120px, 1fr) minmax(180px, 2fr) 72px;
        align-items: center;
        gap: 0.75rem;
        margin: 0.55rem 0;
    }
    .report-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 0.9rem;
        margin-top: 0.85rem;
    }
    .report-tile {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }
    .report-label {
        color: #637181;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .report-value {
        color: #142235;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.3rem;
        line-height: 1.25;
    }
    .muted {
        color: #61707d;
        font-size: 0.95rem;
    }
    section[data-testid="stSidebar"] {
        background: var(--sidebar);
    }
    section[data-testid="stSidebar"] * {
        color: var(--sidebar-text);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--sidebar-text) !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="select"] input,
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: var(--text) !important;
        fill: var(--text) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div[data-baseweb="select"] > div:hover {
        border-color: #b8c4d0 !important;
    }
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div,
    div[data-testid="stMetric"] p,
    div[data-testid="stMetricValue"] {
        color: var(--text) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--surface-soft);
        border: 1px solid var(--border);
        border-radius: 10px 10px 0 0;
        color: var(--text);
        padding: 0.55rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent);
        font-weight: 700;
    }
    .stButton > button,
    .stDownloadButton > button {
        background: #ffffff;
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        color: #ffffff;
        border-color: var(--accent);
    }
    .stDataFrame, .stTable {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }
    .stExpander {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    st.session_state.setdefault("single_result", None)
    st.session_state.setdefault("compare_result", None)
    st.session_state.setdefault("history", [])


def save_upload(uploaded_file: Any) -> str:
    suffix = Path(uploaded_file.name or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def convert_video_for_preview(source_path: str) -> bytes | None:
    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        return None

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 12.0

    frames: list[Image.Image] = []
    frame_step = max(int(round(fps / 8.0)), 1)

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image.thumbnail((720, 480))
                frames.append(image)
                if len(frames) >= 80:
                    break
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmp:
        preview_path = Path(tmp.name)

    try:
        duration_ms = max(int(1000 / min(fps, 8.0)), 80)
        frames[0].save(
            preview_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        return preview_path.read_bytes()
    finally:
        if preview_path.exists():
            preview_path.unlink()


def render_video_preview(uploaded_file: Any, label: str) -> None:
    suffix = Path(uploaded_file.name or "").suffix.lower()
    if suffix != ".avi":
        st.video(uploaded_file)
        return

    source_path = save_upload(uploaded_file)
    try:
        preview_bytes = convert_video_for_preview(source_path)
    finally:
        if os.path.exists(source_path):
            os.remove(source_path)

    if preview_bytes:
        st.image(preview_bytes)
        st.caption(f"{label} AVI preview is shown as an animated GIF for browser compatibility.")
    else:
        st.info(
            f"{label} AVI file uploaded successfully. Preview conversion was unavailable, but analysis can still run."
        )


@st.cache_resource(show_spinner=False)
def get_bundle(checkpoint_path: str, strategy: str):
    return load_bundle(checkpoint_path=checkpoint_path, strategy=strategy)


@st.cache_data(show_spinner=False)
def load_phase_summary(csv_name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / csv_name)


@st.cache_data(show_spinner=False)
def discover_checkpoints() -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    if not CHECKPOINT_DIR.exists():
        return catalog

    for checkpoint in sorted(CHECKPOINT_DIR.glob("*.pth")):
        name = checkpoint.stem.lower()
        architecture = "gru" if "gru" in name else "lstm" if "lstm" in name else "cnn_only"
        sampling = "motion" if "motion" in name else "hybrid" if "hybrid" in name else "uniform"
        catalog.append(
            {
                "path": str(checkpoint),
                "name": checkpoint.name,
                "stem": checkpoint.stem,
                "architecture": architecture,
                "sampling": sampling,
            }
        )
    return catalog


def image_bytes(frame: dict[str, Any]) -> bytes:
    return base64.b64decode(frame["image_base64"])


def get_selected_checkpoint(
    catalog: list[dict[str, str]], architecture: str, sampling: str
) -> dict[str, str] | None:
    for item in catalog:
        if item["architecture"] == architecture and item["sampling"] == sampling:
            return item
    return None


def available_samplings(catalog: list[dict[str, str]], architecture: str) -> list[str]:
    values = [item["sampling"] for item in catalog if item["architecture"] == architecture]
    return [value for value in SAMPLING_OPTIONS if value in values] or ["uniform"]


def result_image_path(prefix: str, stem: str) -> Path | None:
    path = RESULTS_DIR / f"{prefix}_{stem}.png"
    return path if path.exists() else None


def percent_bar(value: float) -> None:
    safe = max(0.0, min(float(value), 100.0))
    st.markdown(
        f'<div class="bar-track"><div class="bar-fill" style="width:{safe:.2f}%"></div></div>',
        unsafe_allow_html=True,
    )


def probability_rows(items: list[dict[str, Any]]) -> None:
    for item in items:
        probability = float(item["probability"])
        st.markdown(
            f"""
            <div class="prob-row">
                <strong>{item["label"]}</strong>
                <div class="bar-track"><div class="bar-fill" style="width:{probability:.2f}%"></div></div>
                <span>{probability:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_single_report_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": "Cricket Shot Detection Report",
        "generatedAt": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "summary": {
            **result["prediction"],
            "architecture": result["metadata"]["architecture"],
            "sampling": result["metadata"]["strategy"],
            "voting": result["metadata"]["voting_strategy"],
        },
        "topPredictions": result["top_predictions"],
        "insights": result["insights"],
    }


def build_compare_report_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": "Cricket Shot Comparison Report",
        "generatedAt": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "summary": {
            "label": f"{result['video_a']['prediction']['label']} vs {result['video_b']['prediction']['label']}",
            "confidence": result["similarity_score"],
            "architecture": result["video_a"]["metadata"]["architecture"],
            "sampling": result["video_a"]["metadata"]["strategy"],
            "voting": result["video_a"]["metadata"]["voting_strategy"],
        },
        "topPredictions": result["video_a"]["top_predictions"],
        "insights": [
            result["comparison_summary"],
            *result["video_a"]["insights"][:2],
            *result["video_b"]["insights"][:2],
        ],
    }


def add_history(mode: str, payload: dict[str, Any]) -> None:
    if mode == "compare":
        item = {
            "mode": "Compare",
            "title": f"{payload['video_a']['prediction']['label']} vs {payload['video_b']['prediction']['label']}",
            "subtitle": f"{payload['similarity_score']:.2f}% similarity",
            "data": payload,
        }
    else:
        item = {
            "mode": "Single",
            "title": payload["prediction"]["label"],
            "subtitle": f"{payload['prediction']['confidence']:.2f}% confidence",
            "data": payload,
        }
    st.session_state.history = [item, *st.session_state.history[:5]]


def render_experiment_summary(selected_checkpoint: dict[str, str] | None) -> None:
    phase1 = load_phase_summary("phase1.csv")
    phase2 = load_phase_summary("phase2.csv")
    phase3 = load_phase_summary("phase3.csv")

    with st.expander("Notebook Experiment Results", expanded=False):
        tab_a, tab_b, tab_c, tab_d = st.tabs(
            ["Architecture", "Sampling", "Voting Strategy", "Selected Figures"]
        )
        with tab_a:
            st.dataframe(phase1, hide_index=True, use_container_width=True)
            st.download_button(
                "Download architecture CSV",
                data=phase1.to_csv(index=False).encode("utf-8"),
                file_name="architecture-results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with tab_b:
            st.dataframe(phase2, hide_index=True, use_container_width=True)
            st.download_button(
                "Download sampling CSV",
                data=phase2.to_csv(index=False).encode("utf-8"),
                file_name="sampling-results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with tab_c:
            voting_summary = (
                phase3.groupby("method", as_index=False)
                .agg(
                    accuracy=("correct", "mean"),
                    avg_confidence=("confidence", "mean"),
                    avg_entropy=("entropy", "mean"),
                )
                .assign(
                    accuracy=lambda df: (df["accuracy"] * 100).round(2),
                    avg_confidence=lambda df: (df["avg_confidence"] * 100).round(2),
                    avg_entropy=lambda df: df["avg_entropy"].round(4),
                )
            )
            voting_summary = voting_summary.rename(
                columns={
                    "method": "Voting",
                    "accuracy": "Accuracy %",
                    "avg_confidence": "Avg Confidence %",
                    "avg_entropy": "Avg Entropy",
                }
            )
            st.dataframe(voting_summary, hide_index=True, use_container_width=True)
            st.download_button(
                "Download voting CSV",
                data=phase3.to_csv(index=False).encode("utf-8"),
                file_name="voting-results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with tab_d:
            if not selected_checkpoint:
                st.info("Select a model to view notebook figures.")
            else:
                curves = result_image_path("curves", selected_checkpoint["stem"])
                confusion = result_image_path("cm", selected_checkpoint["stem"])
                left, right = st.columns(2)
                with left:
                    st.markdown("#### Training Curves")
                    if curves:
                        st.image(str(curves), use_column_width=True)
                    else:
                        st.info("Training curve figure not available.")
                with right:
                    st.markdown("#### Confusion Matrix")
                    if confusion:
                        st.image(str(confusion), use_column_width=True)
                    else:
                        st.info("Confusion matrix figure not available.")


def render_report_panel(result: dict[str, Any], mode: str) -> None:
    payload = build_compare_report_payload(result) if mode == "compare" else build_single_report_payload(result)
    summary = payload["summary"]
    st.markdown('<div class="section-title">Frontend Report</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
            <div class="muted">Generated {payload['generatedAt']}</div>
            <div class="report-grid">
                <div class="report-tile">
                    <div class="report-label">Result</div>
                    <div class="report-value">{summary.get('label', 'N/A')}</div>
                </div>
                <div class="report-tile">
                    <div class="report-label">Score</div>
                    <div class="report-value">{float(summary.get('confidence', 0.0)):.2f}%</div>
                </div>
                <div class="report-tile">
                    <div class="report-label">Model</div>
                    <div class="report-value">{MODEL_LABELS.get(summary.get('architecture', ''), str(summary.get('architecture', 'N/A')).upper())}</div>
                </div>
                <div class="report-tile">
                    <div class="report-label">Sampling</div>
                    <div class="report-value">{SAMPLING_LABELS.get(summary.get('sampling', ''), str(summary.get('sampling', 'N/A')).title())}</div>
                </div>
                <div class="report-tile">
                    <div class="report-label">Voting</div>
                    <div class="report-value">{VOTING_LABELS.get(summary.get('voting', ''), str(summary.get('voting', 'N/A')).title())}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Top Predictions")
    st.dataframe(pd.DataFrame(payload["topPredictions"]), hide_index=True, use_container_width=True)
    st.markdown("#### Key Insights")
    for insight in payload["insights"]:
        st.write(f"- {insight}")


def render_analysis(analysis: dict[str, Any], heading: str) -> None:
    st.markdown(f'<div class="section-title">{heading}</div>', unsafe_allow_html=True)
    meta = analysis["metadata"]
    pred = analysis["prediction"]

    a, b = st.columns([2, 1])
    with a:
        st.subheader(pred["label"])
        st.caption(
            f"{meta['filename']} | arch={meta['architecture']} | sampling={meta['strategy']} | "
            f"voting={meta['voting_strategy']} | {meta['duration_seconds']:.2f}s"
        )
    with b:
        st.metric("Confidence", f"{pred['confidence']:.2f}%")
        percent_bar(pred["confidence"])

    cols = st.columns(6)
    cols[0].metric("Frames", meta["frame_count"])
    cols[1].metric("Sampled", meta["sampled_frames"])
    cols[2].metric("Windows", meta["window_count"])
    cols[3].metric("FPS", meta["fps"])
    cols[4].metric("Duration", f"{meta['duration_seconds']:.2f}s")
    cols[5].metric("Checkpoint", Path(meta["checkpoint"]).name)

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Timeline")
        if analysis["timeline"]:
            timeline_df = pd.DataFrame(analysis["timeline"])
            st.line_chart(timeline_df, x="time", y="confidence", height=280)
            st.dataframe(timeline_df, hide_index=True, use_container_width=True)
        else:
            st.info("Timeline not available.")
    with right:
        st.markdown("#### Top Predictions")
        probability_rows(analysis["top_predictions"])
        st.markdown("#### All Classes")
        st.dataframe(pd.DataFrame(analysis["breakdown"]), hide_index=True, use_container_width=True)

    st.markdown("#### Sampled Frames")
    frame_cols = st.columns(max(1, min(4, len(analysis["key_frames"]))))
    for idx, frame in enumerate(analysis["key_frames"]):
        with frame_cols[idx % len(frame_cols)]:
            st.image(image_bytes(frame), use_column_width=True)
            st.caption(f"{frame['label']} at {frame['time']:.2f}s | {frame['confidence']:.2f}%")

    st.markdown("#### Notes")
    for insight in analysis["insights"]:
        st.info(insight)


def render_download_buttons(result: dict[str, Any], mode: str) -> None:
    payload = build_compare_report_payload(result) if mode == "compare" else build_single_report_payload(result)
    pdf = create_pdf_report(payload)
    report_name = "comparison" if mode == "compare" else "analysis"

    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download PDF Report",
            data=pdf,
            file_name=f"cricket-shot-{report_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "Download JSON Report",
            data=json.dumps(payload, indent=2).encode("utf-8"),
            file_name=f"cricket-shot-{report_name}.json",
            mime="application/json",
            use_container_width=True,
        )


def main() -> None:
    init_state()
    discovered_checkpoint = find_checkpoint()
    checkpoint_catalog = discover_checkpoints()

    default_architecture = "cnn_only"
    default_sampling = "uniform"
    if discovered_checkpoint:
        lowered = discovered_checkpoint.stem.lower()
        default_architecture = "gru" if "gru" in lowered else "lstm" if "lstm" in lowered else "cnn_only"
        default_sampling = "motion" if "motion" in lowered else "hybrid" if "hybrid" in lowered else "uniform"

    with st.sidebar:
        st.title("Model Controls")
        st.write("Choose the checkpoint setup you want to run.")

        architecture_options = [
            arch for arch in ["cnn_only", "gru", "lstm"] if any(cp["architecture"] == arch for cp in checkpoint_catalog)
        ]
        architecture = st.selectbox(
            "Model",
            architecture_options,
            index=architecture_options.index(default_architecture) if default_architecture in architecture_options else 0,
            format_func=lambda value: MODEL_LABELS.get(value, value.upper()),
        )

        sampling_options = available_samplings(checkpoint_catalog, architecture)
        default_sampling = default_sampling if default_sampling in sampling_options else sampling_options[0]
        strategy = st.selectbox(
            "Sampling",
            sampling_options,
            index=sampling_options.index(default_sampling),
            format_func=lambda value: SAMPLING_LABELS.get(value, value.title()),
        )
        voting_strategy = st.selectbox(
            "Voting Strategy",
            VOTING_OPTIONS,
            index=0,
            format_func=lambda value: VOTING_LABELS.get(value, value.title()),
        )

        selected_checkpoint = get_selected_checkpoint(checkpoint_catalog, architecture, strategy)
        checkpoint_path = selected_checkpoint["path"] if selected_checkpoint else ""
        if checkpoint_path:
            st.caption(f"Checkpoint: {Path(checkpoint_path).name}")
        else:
            st.warning("Selected combination is not available in the checkpoint folder.")

        st.markdown("#### Best Notebook Result")
        st.success("Best result: CNN Only + Uniform + Single Clip")
        st.markdown("#### Recent Sessions")
        for idx, item in enumerate(st.session_state.history):
            if st.button(f"{item['mode']}: {item['title']}", key=f"history_{idx}", use_container_width=True):
                if item["mode"] == "Compare":
                    st.session_state.compare_result = item["data"]
                    st.session_state.single_result = None
                else:
                    st.session_state.single_result = item["data"]
                    st.session_state.compare_result = None
                st.rerun()

    st.markdown(
        """
        <div class="hero">
            <h1>Cricket Shot Detection</h1>
            <p>
                Upload one video for recognition or two videos for similarity comparison. This version includes
                direct model selection, sampling selection, voting strategy selection, better notebook figures,
                and full report downloads from the frontend.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_a, top_b, top_c, top_d = st.columns(4)
    top_a.metric("Model", MODEL_LABELS.get(architecture, architecture.upper()))
    top_b.metric("Sampling", SAMPLING_LABELS.get(strategy, strategy.title()))
    top_c.metric("Voting", VOTING_LABELS.get(voting_strategy, voting_strategy.title()))
    top_d.metric("Checkpoint", Path(checkpoint_path).name if checkpoint_path else "Unavailable")

    render_experiment_summary(selected_checkpoint if checkpoint_catalog else None)

    if not checkpoint_path:
        st.warning("No checkpoint path found.")
        return

    primary_col, secondary_col = st.columns(2)
    with primary_col:
        primary = st.file_uploader("Primary video", type=SUPPORTED_TYPES, key="primary")
        if primary:
            render_video_preview(primary, "Primary video")
    with secondary_col:
        secondary = st.file_uploader("Secondary video for comparison", type=SUPPORTED_TYPES, key="secondary")
        if secondary:
            render_video_preview(secondary, "Secondary video")

    action = "Run Comparison" if secondary else "Run Analysis"
    if st.button(action, type="primary", use_container_width=True, disabled=primary is None):
        try:
            bundle = get_bundle(checkpoint_path, strategy)
        except Exception as exc:
            st.error(str(exc))
            return

        with st.spinner("Running notebook-based inference..."):
            first_path = save_upload(primary)
            second_path = save_upload(secondary) if secondary else None
            try:
                first = analyze_video(first_path, bundle, voting_strategy=voting_strategy)
                if secondary and second_path:
                    second = analyze_video(second_path, bundle, voting_strategy=voting_strategy)
                    result = compare_analyses(first, second)
                    st.session_state.compare_result = result
                    st.session_state.single_result = None
                    add_history("compare", result)
                else:
                    st.session_state.single_result = first
                    st.session_state.compare_result = None
                    add_history("single", first)
            finally:
                if os.path.exists(first_path):
                    os.remove(first_path)
                if second_path and os.path.exists(second_path):
                    os.remove(second_path)

    if st.session_state.compare_result:
        result = st.session_state.compare_result
        render_report_panel(result, "compare")
        render_download_buttons(result, "compare")
        st.markdown('<div class="section-title">Similarity</div>', unsafe_allow_html=True)
        st.metric("Similarity Score", f"{result['similarity_score']:.2f}%")
        percent_bar(result["similarity_score"])
        st.success(result["comparison_summary"])
        left, right = st.columns(2)
        with left:
            render_analysis(result["video_a"], "Video A")
        with right:
            render_analysis(result["video_b"], "Video B")
    elif st.session_state.single_result:
        render_report_panel(st.session_state.single_result, "single")
        render_download_buttons(st.session_state.single_result, "single")
        render_analysis(st.session_state.single_result, "Recognition Result")
    else:
        st.markdown(
            '<div class="card"><strong>Ready.</strong><br/>Upload one video to classify it or two videos to compare them with the selected checkpoint.</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

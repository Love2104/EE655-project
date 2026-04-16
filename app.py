from __future__ import annotations

import base64
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from backend.app.services import analyze_video, compare_analyses, create_pdf_report, get_model_bundle


SUPPORTED_TYPES = ["mp4", "avi", "mov", "mkv"]


st.set_page_config(
    page_title="Cricket Shot Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --pitch: #0f2418;
        --panel: #f8fbf8;
        --line: #dbe7de;
        --ink: #142116;
        --muted: #66736a;
        --accent: #0e7a4f;
        --amber: #bf7c19;
    }

    .stApp {
        background:
            linear-gradient(90deg, rgba(15, 36, 24, 0.04) 1px, transparent 1px),
            linear-gradient(180deg, rgba(15, 36, 24, 0.04) 1px, transparent 1px),
            #f4f7f2;
        background-size: 28px 28px;
        color: var(--ink);
    }

    .block-container {
        padding-top: 1.6rem;
        max-width: 1500px;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: 0 10px 26px rgba(20, 33, 22, 0.06);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--ink);
    }

    .hero {
        background:
            linear-gradient(135deg, rgba(14, 122, 79, 0.92), rgba(15, 36, 24, 0.94)),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.08) 0, rgba(255,255,255,0.08) 2px, transparent 2px, transparent 44px);
        border-radius: 8px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.16);
        box-shadow: 0 24px 70px rgba(15, 36, 24, 0.22);
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: clamp(2.1rem, 4vw, 4.6rem);
        line-height: 1.03;
        letter-spacing: 0;
    }

    .hero p {
        color: rgba(255,255,255,0.84);
        max-width: 760px;
        margin-top: 0.9rem;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    .section-title {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 700;
        margin: 1rem 0 0.4rem;
    }

    .soft-panel {
        background: rgba(248, 251, 248, 0.9);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 12px 36px rgba(20, 33, 22, 0.06);
    }

    .bar-track {
        width: 100%;
        height: 0.65rem;
        background: #e5ece5;
        border-radius: 999px;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #0e7a4f, #78a22f);
        border-radius: 999px;
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.9rem;
    }

    .prediction-row {
        display: grid;
        grid-template-columns: minmax(110px, 1fr) minmax(140px, 2fr) 64px;
        align-items: center;
        gap: 0.75rem;
        margin: 0.55rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def warm_model() -> bool:
    get_model_bundle()
    return True


def init_state() -> None:
    st.session_state.setdefault("single_result", None)
    st.session_state.setdefault("compare_result", None)
    st.session_state.setdefault("history", [])


def suffix_for(uploaded_file: Any) -> str:
    return Path(uploaded_file.name or "video.mp4").suffix or ".mp4"


def save_uploaded_video(uploaded_file: Any) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_for(uploaded_file)) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return dict(model)


def percent_bar(value: float) -> None:
    safe_value = max(0.0, min(float(value), 100.0))
    st.markdown(
        f"""
        <div class="bar-track">
            <div class="bar-fill" style="width: {safe_value:.2f}%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_rows(items: list[dict[str, Any]]) -> None:
    for item in items:
        probability = float(item["probability"])
        st.markdown(
            f"""
            <div class="prediction-row">
                <strong>{item["label"]}</strong>
                <div class="bar-track"><div class="bar-fill" style="width: {probability:.2f}%"></div></div>
                <span>{probability:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def image_bytes(frame: dict[str, Any], show_heatmap: bool) -> bytes:
    key = "heatmap_base64" if show_heatmap and frame.get("heatmap_base64") else "image_base64"
    return base64.b64decode(frame[key])


def build_single_report_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": "Cricket Shot Recognition Report",
        "generatedAt": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "summary": result["prediction"],
        "topPredictions": result["top_predictions"],
        "insights": result["insights"],
    }


def build_compare_report_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": "Cricket Shot Similarity Report",
        "generatedAt": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "summary": {
            "label": f"{result['video_a']['prediction']['label']} vs {result['video_b']['prediction']['label']}",
            "confidence": result["similarity_score"],
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


def run_single(video_file: Any) -> None:
    temp_path = save_uploaded_video(video_file)
    try:
        warm_model()
        analysis, _ = analyze_video(temp_path, video_file.name)
        payload = to_dict(analysis)
        st.session_state.single_result = payload
        st.session_state.compare_result = None
        add_history("single", payload)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_compare(primary_file: Any, secondary_file: Any) -> None:
    first_path = save_uploaded_video(primary_file)
    second_path = save_uploaded_video(secondary_file)
    try:
        warm_model()
        first, first_embedding = analyze_video(first_path, primary_file.name)
        second, second_embedding = analyze_video(second_path, secondary_file.name)
        first, second, similarity, summary = compare_analyses(first, first_embedding, second, second_embedding)
        payload = {
            "request_id": first.request_id,
            "video_a": to_dict(first),
            "video_b": to_dict(second),
            "similarity_score": similarity,
            "comparison_summary": summary,
        }
        st.session_state.compare_result = payload
        st.session_state.single_result = None
        add_history("compare", payload)
    finally:
        for path in (first_path, second_path):
            if os.path.exists(path):
                os.remove(path)


def render_uploads() -> tuple[Any, Any]:
    st.markdown('<div class="section-title">Upload Clips</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        primary_file = st.file_uploader("Primary video", type=SUPPORTED_TYPES, key="primary_video")
        if primary_file:
            st.video(primary_file)
    with right:
        secondary_file = st.file_uploader("Secondary video for comparison", type=SUPPORTED_TYPES, key="secondary_video")
        if secondary_file:
            st.video(secondary_file)

    action_label = "Run Comparison" if secondary_file else "Run Analysis"
    disabled = primary_file is None
    if st.button(action_label, type="primary", disabled=disabled, use_container_width=True):
        with st.spinner("Loading model and analyzing cricket shot frames..."):
            if secondary_file:
                run_compare(primary_file, secondary_file)
            else:
                run_single(primary_file)
        st.success("Analysis complete.")

    return primary_file, secondary_file


def render_analysis(analysis: dict[str, Any], heading: str) -> None:
    prediction = analysis["prediction"]
    metadata = analysis["metadata"]

    st.markdown(f'<div class="section-title">{heading}</div>', unsafe_allow_html=True)
    top_line, confidence_box = st.columns([2.2, 1])
    with top_line:
        st.subheader(prediction["label"])
        st.caption(metadata["filename"])
    with confidence_box:
        st.metric("Confidence", f"{prediction['confidence']:.2f}%")
        percent_bar(prediction["confidence"])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Duration", f"{metadata['duration_seconds']:.1f}s")
    metric_cols[1].metric("FPS", f"{metadata['fps']:.1f}")
    metric_cols[2].metric("Frames", f"{metadata['frame_count']}")
    metric_cols[3].metric("Sampled", f"{metadata['sampled_frames']}")

    timeline = analysis.get("timeline", [])
    breakdown = analysis.get("breakdown", [])

    chart_col, pred_col = st.columns([1.45, 1])
    with chart_col:
        st.markdown("#### Prediction Timeline")
        if timeline:
            timeline_df = pd.DataFrame(
                {
                    "time_seconds": [point["time"] for point in timeline],
                    "confidence": [point["confidence"] for point in timeline],
                }
            )
            st.line_chart(timeline_df, x="time_seconds", y="confidence", height=285)
            st.dataframe(
                pd.DataFrame(timeline)[["time", "label", "confidence"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No timeline points were produced for this clip.")

    with pred_col:
        st.markdown("#### Top Predictions")
        probability_rows(analysis.get("top_predictions", []))
        st.markdown("#### Probability Breakdown")
        if breakdown:
            st.bar_chart(
                pd.DataFrame(
                    {
                        "label": [item["label"] for item in breakdown],
                        "probability": [item["probability"] for item in breakdown],
                    }
                ),
                x="label",
                y="probability",
                height=285,
            )

    st.markdown("#### Key Frames")
    show_heatmap = st.toggle(f"Show heatmap overlays for {heading}", value=True, key=f"heatmap_{analysis['request_id']}_{heading}")
    frame_cols = st.columns(3)
    for index, frame in enumerate(analysis.get("key_frames", [])):
        with frame_cols[index % 3]:
            st.image(image_bytes(frame, show_heatmap), use_column_width=True)
            st.caption(f"{frame['label']} at {frame['time']:.2f}s, {frame['confidence']:.2f}%")

    st.markdown("#### Analyst Notes")
    for insight in analysis.get("insights", []):
        st.info(insight)


def render_download_button(result: dict[str, Any], mode: str) -> None:
    payload = build_compare_report_payload(result) if mode == "compare" else build_single_report_payload(result)
    pdf_bytes = create_pdf_report(payload)
    st.download_button(
        "Download PDF Report",
        data=pdf_bytes,
        file_name="cricket-shot-report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def render_results() -> None:
    compare_result = st.session_state.compare_result
    single_result = st.session_state.single_result

    if compare_result:
        st.markdown('<div class="section-title">Similarity Result</div>', unsafe_allow_html=True)
        st.metric("Similarity", f"{compare_result['similarity_score']:.2f}%")
        percent_bar(compare_result["similarity_score"])
        st.success(compare_result["comparison_summary"])
        render_download_button(compare_result, "compare")

        first_col, second_col = st.columns(2)
        with first_col:
            render_analysis(compare_result["video_a"], "Video A")
        with second_col:
            render_analysis(compare_result["video_b"], "Video B")
        return

    if single_result:
        render_download_button(single_result, "single")
        render_analysis(single_result, "Recognition Result")
        return

    st.markdown(
        """
        <div class="soft-panel">
            <h3>Results will appear here after analysis.</h3>
            <p class="small-muted">
                Upload one clip for shot recognition, or add a second clip to compare batting mechanics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Cricket Lab")
        st.write("Streamlit version, no React server needed.")
        st.markdown("#### Recent Sessions")
        if not st.session_state.history:
            st.caption("Run an analysis to build this session list.")
        for index, item in enumerate(st.session_state.history):
            if st.button(
                f"{item['mode']}: {item['title']}",
                help=item["subtitle"],
                key=f"history_{index}",
                use_container_width=True,
            ):
                if item["mode"] == "Compare":
                    st.session_state.compare_result = item["data"]
                    st.session_state.single_result = None
                else:
                    st.session_state.single_result = item["data"]
                    st.session_state.compare_result = None
                st.rerun()

        st.markdown("#### Run Command")
        st.code("streamlit run app.py", language="powershell")


def main() -> None:
    init_state()
    render_sidebar()

    st.markdown(
        """
        <div class="hero">
            <h1>Cricket Shot Recognition</h1>
            <p>
                Classify cricket shots, inspect confidence over time, view key-frame heatmaps,
                compare two batting clips, and export the same PDF report without a React build step.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_uploads()
    render_results()


if __name__ == "__main__":
    main()

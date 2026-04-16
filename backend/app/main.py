from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .services import analyze_video, compare_analyses, create_pdf_report, get_model_bundle, save_upload

app = FastAPI(
    title="Cricket Shot Recognition API",
    description="Inference and comparison API for cricket shot classification and similarity analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warm_model() -> None:
    get_model_bundle()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, object]:
    return {
        "message": "Cricket Shot Recognition API is running.",
        "frontend": "Start the React app and open http://127.0.0.1:5173",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "compare": "/compare",
            "docs": "/docs",
        },
    }


def _suffix(upload: UploadFile) -> str:
    return Path(upload.filename or "video.mp4").suffix or ".mp4"


@app.post("/predict")
async def predict(video: UploadFile = File(...)) -> JSONResponse:
    temp_path = save_upload(video, _suffix(video))
    try:
        analysis, _ = analyze_video(temp_path, video.filename or "uploaded-video")
        return JSONResponse(analysis.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/compare")
async def compare(video_a: UploadFile = File(...), video_b: UploadFile = File(...)) -> JSONResponse:
    first_path = save_upload(video_a, _suffix(video_a))
    second_path = save_upload(video_b, _suffix(video_b))
    try:
        first, first_embedding = analyze_video(first_path, video_a.filename or "video-a")
        second, second_embedding = analyze_video(second_path, video_b.filename or "video-b")
        first, second, similarity, summary = compare_analyses(first, first_embedding, second, second_embedding)
        return JSONResponse(
            {
                "request_id": first.request_id,
                "video_a": first.model_dump(),
                "video_b": second.model_dump(),
                "similarity_score": similarity,
                "comparison_summary": summary,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for path in (first_path, second_path):
            if os.path.exists(path):
                os.remove(path)


@app.post("/report")
async def report(payload: dict) -> StreamingResponse:
    pdf = create_pdf_report(payload)
    return StreamingResponse(
        iter([pdf]),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="cricket-shot-report.pdf"'},
    )

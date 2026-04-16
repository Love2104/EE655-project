# Cricket Shot Recognition and Similarity Analysis

A Streamlit cricket analytics application built around the existing `model_weights.h5` model. It classifies cricket shot videos, compares two batting clips, shows key-frame heatmaps, and generates PDF reports without needing a React or Node.js development server.

## Start Here

Run these commands from the project folder:

```text
E:\SEM 6\New folder (2)\Group project
```

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

## Use the App

1. Upload one cricket shot video to classify the shot.
2. Upload two videos to compare batting similarity.
3. Click `Run Analysis` or `Run Comparison`.
4. View confidence, top predictions, timeline, key frames, heatmaps, and similarity score.
5. Click `Download PDF Report` after results appear.

The first run can take some time because TensorFlow loads the EfficientNetB0 + GRU model and reads `model_weights.h5`.

## Features

- Single-video cricket shot classification
- Two-video similarity comparison
- Video preview before processing
- Predicted shot type with confidence
- Top-3 predictions and full probability breakdown
- Frame-wise timeline analysis
- Key frames with optional saliency heatmap overlays
- Similarity percentage for two uploaded videos
- Downloadable PDF report
- Recent-session history in Streamlit session state

## Tech Stack

- Streamlit
- TensorFlow / Keras
- OpenCV
- Pandas
- ReportLab
- FastAPI service modules reused for inference logic

## Folder Structure

```text
.
|-- app.py
|-- backend
|   `-- app
|       |-- __init__.py
|       |-- main.py
|       |-- schemas.py
|       `-- services.py
|-- frontend
|   `-- ...
|-- model_weights.h5
|-- requirements.txt
`-- README.md
```

`app.py` is the main UI now. The `frontend/` React code is still present as reference, but it is no longer required to run the project.

## Optional API Server

The FastAPI backend can still be run separately if needed:

```powershell
python -m uvicorn backend.app.main:app --reload
```

The API starts on:

```text
http://127.0.0.1:8000
```

Useful endpoints:

- `GET /health`
- `POST /predict`
- `POST /compare`
- `POST /report`

## How It Works

1. Save uploaded video temporarily
2. Extract uniformly sampled frames with OpenCV
3. Format frames to `224 x 224`
4. Run the existing EfficientNetB0 + GRU model using `model_weights.h5`
5. Generate shot classification, probability breakdown, timeline inference, key frames, feature embeddings, and saliency heatmaps
6. Display the results in Streamlit and generate a PDF report on demand

## Notes

- The model is not retrained.
- Existing `model_weights.h5` is used directly.
- No Node.js install is needed for the Streamlit workflow.
- Local folders such as `.venv/`, `.vscode/`, `frontend/node_modules/`, and `output/` are ignored by Git.

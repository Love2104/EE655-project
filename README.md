# Cricket Shot Recognition and Similarity Analysis

A full-stack cricket analytics application built on top of the original `model_weights.h5` from the repository. The project now includes:

- A `FastAPI` backend for video inference, comparison, analytics, and PDF report generation
- A `React + Tailwind CSS` frontend with a modern dark dashboard
- Timeline charts, top-3 predictions, key frames, saliency heatmaps, similarity scoring, and local session history

## Start Here

Follow these steps from the project folder:

```text
E:\SEM 6\New folder (2)\Group project
```

### Step 1: Start the Backend

Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload
```

Keep this terminal open. The backend API will run at:

```text
http://127.0.0.1:8000
```

Important: this is only the API server, not the website UI. If you open this URL directly, it shows API information. The actual web app runs from the frontend in Step 2.

You can test it by opening:

```text
http://127.0.0.1:8000/health
```

It should show:

```json
{"status":"ok"}
```

### Step 2: Start the Frontend

Open a second PowerShell terminal in the same project folder and run:

```powershell
cd frontend
npm install
npm run dev
```

Keep this second terminal open. The website will run at:

```text
http://127.0.0.1:5173
```

Open that URL in your browser.

### Step 3: Use the App

1. Upload one cricket shot video to classify the shot.
2. Upload two videos to compare similarity.
3. Click `Run Analysis` or `Run Comparison`.
4. View prediction, confidence, top-3 classes, key frames, timeline graph, heatmaps, and similarity score.
5. Click `Download PDF report` after results appear.

### If Something Goes Wrong

- If `uvicorn` is not recognized, run `pip install -r requirements.txt` again while the virtual environment is active.
- If `npm` is not recognized, install Node.js from `https://nodejs.org/`.
- If the frontend says the backend is not reachable, make sure the backend terminal is still running at `http://127.0.0.1:8000`.
- The first backend start can take some time because TensorFlow loads the model.

## Features

- Upload one video to classify a cricket shot
- Upload two videos to compare batting mechanics
- Preview videos before processing
- Predicted shot type with confidence progress bar
- Top-3 predictions and full probability breakdown
- Frame-wise timeline analysis using sliding-window inference
- Key frames used for prediction
- Saliency-based visual explanation overlays
- Similarity percentage for two uploaded videos
- Downloadable PDF report
- Browser-side history of previous analyses

## Tech Stack

### Backend

- FastAPI
- TensorFlow / Keras
- OpenCV
- ReportLab

### Frontend

- React (Vite)
- Tailwind CSS
- Recharts
- Framer Motion
- React Dropzone

## Folder Structure

```text
.
|-- backend
|   `-- app
|       |-- __init__.py
|       |-- main.py
|       |-- schemas.py
|       `-- services.py
|-- frontend
|   |-- package.json
|   |-- vite.config.js
|   `-- src
|       |-- App.jsx
|       |-- index.css
|       |-- main.jsx
|       |-- components
|       |   |-- AnalysisCard.jsx
|       |   |-- ComparisonView.jsx
|       |   |-- HistoryPanel.jsx
|       |   `-- UploadPanel.jsx
|       `-- lib
|           `-- api.js
|-- model_weights.h5
|-- requirements.txt
`-- README.md
```

## API Endpoints

- `GET /health`
- `POST /predict`
- `POST /compare`
- `POST /report`

## Run Locally

### 1. Backend

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

The backend starts on `http://127.0.0.1:8000`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend starts on `http://127.0.0.1:5173`.

### 3. Optional frontend API override

If the backend runs on a different host or port, create a `.env` file inside `frontend/`:

```bash
VITE_API_URL=http://127.0.0.1:8000
```

## How the Backend Works

1. Save uploaded video temporarily
2. Extract uniformly sampled frames with OpenCV
3. Format frames to `224 x 224`
4. Run the existing EfficientNetB0 + GRU model using `model_weights.h5`
5. Generate:
   - Shot classification
   - Probability breakdown
   - Sliding-window timeline inference
   - Key frames
   - Feature embeddings for similarity
   - Saliency heatmaps
6. Return structured JSON to the frontend

## Notes

- The model is not retrained.
- Existing `model_weights.h5` is used directly.
- The old Streamlit prototype remains in the repository as legacy context, but the new app is the FastAPI + React stack described above.

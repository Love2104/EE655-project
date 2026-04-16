# Cricket Shot Detection

This repository now runs as a Streamlit app for cricket shot recognition and video comparison.

The current app is centered around:

- `app.py` for the Streamlit interface
- `cricket_notebook_model.py` for notebook-style inference and report generation
- `requirements.txt` for Python dependencies

## Features

- Upload one video and classify the cricket shot
- Upload two videos and compare similarity
- Preview common video formats in the portal
- Show AVI uploads with a browser-safe animated preview
- View top predictions, timeline confidence, sampled frames, and report summaries
- Download PDF and JSON reports

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app usually opens at `http://localhost:8501`.

## Model Assets

The Streamlit app expects notebook checkpoint files and result CSVs at runtime, typically under:

- `results_unzipped/checkpoints/`
- `results_unzipped/results/`

Those large generated assets are intentionally not included in this Git push.

## Test Clip

The local smoke test used:

- `test/pull_0001.avi`
- `test/pull_0002.avi`

## Notes

- The current working frontend is Streamlit.
- Large logs and result bundles are ignored in Git to keep the repository lightweight.

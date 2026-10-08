# FraudShield AI V3 — UI refinement

V3 is a design upgrade of the tested V2 project, using the **same trained model**, evaluation report, FastAPI endpoints, investigation workflow, SHAP, simulator, and monitoring.

## What's changed
- Removed verbose platform explanations and prominent sidebar warning.
- Added compact model-ready status, workspace overview, model name, and decision threshold.
- Moved responsible-use disclaimer to a subtle footer.
- Improved Streamlit header, upload panel, and chart label contrast.
- Preserved original test suite and API.

## Windows launch (from the inner project folder)

If `.venv` is in the *outer* folder, as in your screenshots:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Or if `.venv` is in this folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open http://localhost:8501. The original API can still be started separately on port 8081.

For details on datasets, training, and limitations see `README.md` and `README_V2.md`.

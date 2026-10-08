# FraudShield AI — Advanced Financial Fraud Detection

A **working end-to-end portfolio project** for credit-card fraud risk scoring with chronological evaluation, imbalanced ML baselines, XGBoost, a FastAPI inference service, and a Streamlit dashboard. **No fabricated model metrics or synthetic transaction performance claims.**

> Educational research demo only. Do not use for actual financial decisions. Model scores are **not calibrated probabilities**. Anonymised PCA components cannot provide human-readable reasons for real-world fraud.

## Dataset (not included)
Download the public **ULB Credit Card Fraud Detection** `creditcard.csv` from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and save it as `data/raw/creditcard.csv`. The dataset contains 284,807 transactions and 492 fraud labels, with anonymised V1–V28, Time, Amount and Class.

## Windows PowerShell setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.fraudshield.train --csv data/raw/creditcard.csv --out models
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000/docs for API Swagger docs. In a **second** terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open the Streamlit local URL displayed in terminal. For a demo, upload a CSV with columns `Time`, `Amount`, `V1`–`V28`. The `Class` column is optional for scoring. Only upload public anonymised data.

## Project layout

- `src/fraudshield/features.py`: strict 30-feature validation
- `src/fraudshield/train.py`: chronological 60/20/20 split, dummy and logistic baselines, XGBoost, PR-AUC model selection, validation-only threshold, untouched test report
- `src/fraudshield/predict.py`: trusted local artifact inference
- `api/main.py`: `/health`, `/model-info`, `/predict`
- `dashboard/app.py`: upload, risk distribution, results and CSV download
- `tests/`: validation, split and API tests
- `models/`: locally generated artifacts and `evaluation.json` (not committed)

## Evaluation design and limitations

- **Chronological** 60/20/20 split based on Time, with training-only preprocessing. Note that the dataset spans approximately two days; it does not replicate real production data drift.
- Dummy baseline, class-weighted logistic regression and class-weighted XGBoost are compared using **validation PR-AUC**. The winning trained model is saved.
- The threshold is selected using validation labels (max recall subject to a requested precision floor, or validation F1 fallback). Only then are final test metrics computed.
- PR-AUC, ROC-AUC, precision, recall and alert rate are written to `models/evaluation.json`. **Run training before claiming results**.
- Class-weighting can harm calibration. Risk scores are not literal probabilities. Fraud features are anonymised, so SHAP would explain components rather than business semantics.
- This project is **batch-scored and request-scored** (simulated real-time); it does not ingest a true live stream. No authentication, rate limiting, live drift alerts or calibrated decision support is provided.
- Never upload private banking data or commit the model artifact from an untrusted source. `joblib` loads pickle data and is unsafe for untrusted files.

## Deployment

1. Push the source to GitHub (without `data/raw/` or generated model artifacts).
2. Train locally using the licensed public dataset and inspect `models/evaluation.json`.
3. Deploy the Docker API to a Python/Docker host **with a trusted trained model artifact supplied securely** (the default Docker build excludes raw data and models; without the artifact `/health` returns `model_missing`).
4. Deploy the Streamlit app to a Python hosting service with a trusted model artifact. Hosting must support sufficient memory for XGBoost and SHAP if enabled.
5. Add your public GitHub/demo links to the existing Netlify portfolio. **Do not claim a live demo until it works publicly.**

## Future advanced extensions

- SHAP explanations for anonymised features, with caveats
- Explicit alert-budget optimisation and calibrated probabilities
- PaySim dataset and realistic synthetic transaction stream
- Model versioning, monitoring, latency benchmarks, authentication, CI/CD deployment

## Portfolio-ready description (once trained)

“Developed an end-to-end credit-card fraud detection research application using chronological validation, imbalanced learning, XGBoost and a FastAPI risk-scoring service. Built an interactive Streamlit dashboard and automated tests; assessed performance with PR-AUC, recall and precision on held-out transactions.”

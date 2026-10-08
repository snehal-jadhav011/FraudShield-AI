# FraudShield AI V2 — Transaction Risk Command Center

This V2 upgrade uses your existing **trained XGBoost model** and evaluation report.
The original FastAPI service, training pipeline, and automated tests are preserved.

## New capabilities

- Premium dark Streamlit command center with metrics, risk-score distribution and screening tiers
- Transaction investigation table with individual risk scores
- On-demand SHAP explanation for selected XGBoost transactions
- Model comparison and threshold what-if analysis
- Historical transaction replay (simulated, not live banking)
- Data-quality diagnostics and batch distribution comparisons
- Label-aware sample evaluation when CSV contains a `Class` column
- CSV export of scored transactions

## Run on Windows PowerShell

Open the project folder in VS Code, then:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8081
```

In a **second** terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Visit http://localhost:8501 for the dashboard and http://127.0.0.1:8081/docs for the API.

If you downloaded the V2 ZIP without a `.venv` folder, create one first:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dataset

Download `creditcard.csv` from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it at `data/raw/creditcard.csv`. It is **not included in the ZIP**.
The ZIP retains the trained `models/fraud_model.joblib` and `models/evaluation.json`
from the uploaded project; retraining is optional.

## Test file

For a CSV of mixed known fraud/genuine transactions with labels:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_csv('data/raw/creditcard.csv'); x=pd.concat([d[d.Class==1].tail(25),d[d.Class==0].tail(75)]).sample(frac=1,random_state=42); x.to_csv('mixed_with_labels.csv',index=False)"
```

Upload `mixed_with_labels.csv` in the dashboard to view a confusion matrix.
**Important:** This deliberately enriched sample is not representative of
production fraud prevalence; its metrics should not be compared to test metrics.

## Interpretation and limitations

- A flagged transaction is **not** proven fraud.
- Risk scores are **not calibrated probabilities**.
- ULB V1–V28 are anonymised; SHAP explains component influence, not real-world causation.
- SHAP runs on demand and may take time to load.
- The stream simulator replays uploaded data locally; no actual real-time feed exists.
- Monitoring is local batch diagnostics, not deployed drift monitoring.
- Do not upload customer information or deploy this as a real financial decision system.
- Trained joblib models must only be loaded from trusted sources.

## Project structure

The original `README.md` documents training and API endpoints.
The new `src/fraudshield/analytics.py` and `tests/test_analytics.py`
provide unit-tested analytical helpers.

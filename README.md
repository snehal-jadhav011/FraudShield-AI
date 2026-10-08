# FraudShield AI
### Explainable Credit Card Fraud Detection | End-to-End Machine Learning Portfolio Project

**Python · XGBoost · Scikit-learn · SHAP · FastAPI · Streamlit · Plotly · Docker**

FraudShield AI is an end-to-end machine learning research application for identifying potentially fraudulent credit card transactions. It combines imbalanced classification, explainable AI, model evaluation, an inference API, and an interactive investigation dashboard.

**GitHub Repository:** https://github.com/snehal-jadhav011/FraudShield-AI

**Live Demo:** Coming soon — add the verified Streamlit deployment URL after successful testing.

> **Disclaimer:** Educational and portfolio demonstration only. Not intended for real financial decisions. Model scores are not calibrated fraud probabilities.

---

## Key Features

- **Fraud Detection:** XGBoost-based classification of anonymised credit card transactions.
- **Explainable AI:** SHAP-based interpretation of feature contributions.
- **Interactive Dashboard:** Streamlit interface with data exploration and risk visualisations.
- **Transaction Investigation:** Review flagged transactions and their risk scores.
- **Model Lab:** Explore evaluation results and classification performance.
- **Simulator:** Replay historical transactions to demonstrate the scoring workflow.
- **Monitoring:** Inspect uploaded datasets for quality issues and scoring summaries.
- **REST API:** FastAPI endpoints for health checks, model information and predictions.
- **Automated Tests:** Pytest-based validation and API tests.

## System Architecture

```text
            Credit Card Transaction Data
                        |
                        v
             Feature Validation
                        |
                        v
              XGBoost Classifier
                        |
             +----------+----------+
             |                     |
             v                     v
         Risk Scores          SHAP Explanations
             |                     |
             +----------+----------+
                        |
             +----------+----------+
             |                     |
             v                     v
        FastAPI Service     Streamlit Dashboard
```

## Dataset

This project uses the publicly available **ULB Credit Card Fraud Detection Dataset**.

**Dataset:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

| Property | Description |
|---|---|
| Transactions | 284,807 |
| Fraudulent transactions | 492 |
| Features | Time, Amount, V1–V28 |
| Target | Class (0 = legitimate, 1 = fraud) |
| Challenge | Severe class imbalance |

The raw dataset is not included in this repository.

## Machine Learning Methodology

1. Validate the required transaction features.
2. Split data chronologically into training, validation and test sets (60/20/20).
3. Compare dummy, logistic regression and XGBoost baselines.
4. Select the model using validation PR-AUC.
5. Select a classification threshold using validation data.
6. Evaluate the selected model on untouched held-out test data.
7. Save the trusted model artifact and evaluation results.

## Model Performance

Previously recorded held-out evaluation results:

| Metric | Result |
|---|---:|
| PR-AUC | 0.801 |
| Recall | 78.7% |
| Precision | 60.8% |
| Selected threshold | 0.2413 |

These values should be checked against the committed `models/evaluation.json` before treating them as the definitive results for the current model artifact.

PR-AUC is especially useful for this highly imbalanced classification problem. The model's scores should not be interpreted as calibrated probabilities.

## Dashboard

The Streamlit dashboard includes five main areas:

**Overview:** Dataset summaries, visualisations and model information.

**Investigation:** Transaction scoring, fraud alerts and SHAP explanations.

**Model Lab:** Model evaluation and performance exploration.

**Simulator:** Historical transaction replay and alert demonstration.

**Monitoring:** Dataset quality checks and scoring summaries.

### Screenshots

Screenshots will be added after the deployed dashboard has been verified.

Suggested screenshots:

- `docs/images/overview.png`
- `docs/images/investigation.png`
- `docs/images/shap-explanation.png`
- `docs/images/model-lab.png`
- `docs/images/monitoring.png`

## Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Explainability | SHAP |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| Visualisation | Plotly |
| Testing | Pytest |
| Containerisation | Docker |
| Version Control | Git, GitHub |

## Local Installation

**Prerequisite:** Python 3.11

Clone the repository:

```powershell
git clone https://github.com/snehal-jadhav011/FraudShield-AI.git
cd FraudShield-AI
```

Create a virtual environment and install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The repository includes a trained model artifact for demonstration. Only load model artifacts from trusted sources.

To retrain the model, download `creditcard.csv` from the dataset link and place it at `data/raw/creditcard.csv`.

```powershell
.\.venv\Scripts\python.exe -m src.fraudshield.train --csv data/raw/creditcard.csv --out models
```

Run the Streamlit dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Run the FastAPI service in a separate terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8081
```

API documentation: http://127.0.0.1:8081/docs

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Project Structure

```text
FraudShield-AI/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── models/
│   ├── evaluation.json
│   └── fraud_model.joblib
├── src/
│   └── fraudshield/
│       ├── features.py
│       ├── train.py
│       ├── predict.py
│       └── analytics.py
├── tests/
├── .github/workflows/
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── README.md
```

## Limitations

- The dataset covers approximately two days of historical transactions.
- PCA-anonymised features do not provide business-readable fraud reasons.
- SHAP explains model feature contributions, not confirmed causes of fraud.
- Scores are not calibrated probabilities.
- The simulator replays historical data rather than ingesting live banking transactions.
- This is not a production banking platform; authentication, rate limiting and operational monitoring would require further development.
- Model artifacts using `joblib` must only be loaded from trusted sources.

## Future Improvements

- Probability calibration and alert-budget optimisation.
- Model drift detection and versioning.
- More realistic streaming transaction simulation.
- API authentication and deployment hardening.
- Additional model comparisons and automated monitoring.

## Author

**Snehal Jadhav**

MSc Data Science — University of Leicester

**GitHub:** https://github.com/snehal-jadhav011

---

*Built as an applied machine learning and explainable AI portfolio project.*
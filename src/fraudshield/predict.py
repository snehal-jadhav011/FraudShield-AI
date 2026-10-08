"""Read-only model inference. Only load locally trusted model artifacts."""
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from .features import FEATURES,validate

@lru_cache(maxsize=1)
def load_model(path="models/fraud_model.joblib"):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Train the model first: {path}")
    return joblib.load(path)

def score(record, artifact=None):
    artifact=artifact or load_model()
    frame=validate(pd.DataFrame([record]),require_target=False)
    raw=float(artifact["model"].predict_proba(frame[FEATURES])[0,1])
    threshold=float(artifact["threshold"])
    return {"risk_score":round(raw,6),"threshold":round(threshold,6),"flagged":bool(raw>=threshold),"model":artifact["name"],"disclaimer":"Risk score is an uncalibrated model output, not a probability of actual fraud."}

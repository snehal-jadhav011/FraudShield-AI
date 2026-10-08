"""Strict feature contract for the public ULB dataset."""
import numpy as np
import pandas as pd
FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
TARGET = "Class"

def validate(df: pd.DataFrame, require_target=True) -> pd.DataFrame:
    needed = FEATURES + ([TARGET] if require_target else [])
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df[needed].copy()
    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="raise")
    if not np.isfinite(out[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Features must be finite numbers")
    if (out["Amount"] < 0).any() or (out["Time"] < 0).any():
        raise ValueError("Time and Amount must be nonnegative")
    if require_target and not out[TARGET].isin([0, 1]).all():
        raise ValueError("Class must be 0 or 1")
    return out

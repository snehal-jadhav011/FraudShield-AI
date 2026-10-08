"""Pure, testable analytics helpers for FraudShield AI V2."""
import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score

def apply_threshold(scores, threshold):
    arr=np.asarray(scores,dtype=float)
    if arr.ndim!=1 or not np.isfinite(arr).all():
        raise ValueError("Scores must be a finite 1D array")
    if not (0<=threshold<=1):
        raise ValueError("Threshold must be between 0 and 1")
    return arr>=threshold

def sample_metrics(labels, scores, threshold):
    y=np.asarray(labels)
    s=np.asarray(scores,dtype=float)
    if len(y)!=len(s) or len(y)==0 or not np.isin(y,[0,1]).all():
        raise ValueError("Labels must be binary and match scores")
    flags=apply_threshold(s,threshold)
    return {"count":len(y),"flagged":int(flags.sum()),
            "precision":float(precision_score(y,flags,zero_division=0)),
            "recall":float(recall_score(y,flags,zero_division=0)),
            "average_precision":float(average_precision_score(y,s)) if len(set(y))==2 else None}

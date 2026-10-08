"""Train on the real ULB dataset. No synthetic scores are presented as real performance."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from .features import FEATURES, TARGET, validate

def chronological_split(df):
    """Sorted by Time, with a 60/20/20 split to approximate future-data evaluation."""
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    a, b = int(len(df)*0.6), int(len(df)*0.8)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]

def metrics(y, scores, threshold=0.5):
    pred = (np.asarray(scores) >= threshold).astype(int)
    return {"average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)) if len(set(y))==2 else None,
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "alert_rate": float(np.mean(pred)), "threshold": float(threshold)}

def choose_threshold(y, scores, min_precision=0.5):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    candidates = [(float(recall[i]), float(t)) for i,t in enumerate(thresholds) if precision[i] >= min_precision]
    if candidates:
        return max(candidates, key=lambda x:x[0])[1]
    # If constraint is infeasible, maximize F1 on validation data.
    f1 = 2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-12)
    return float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5

def train(csv_path, output_dir, min_precision=0.5):
    df=validate(pd.read_csv(csv_path))
    tr,va,te=chronological_split(df)
    if any(split[TARGET].nunique()<2 for split in [tr,va,te]):
        raise ValueError("Each chronological split needs both classes; cannot evaluate reliably")
    xtr,ytr=tr[FEATURES],tr[TARGET]
    xva,yva=va[FEATURES],va[TARGET]
    xte,yte=te[FEATURES],te[TARGET]
    preprocess=ColumnTransformer([("scale",StandardScaler(),["Time","Amount"])],remainder="passthrough")
    ratio=(ytr==0).sum()/max((ytr==1).sum(),1)
    models={
      "dummy":DummyClassifier(strategy="prior"),
      "logistic":Pipeline([("preprocess",preprocess), ("model",LogisticRegression(class_weight="balanced",max_iter=1500,random_state=42))]),
      "xgboost":Pipeline([("preprocess",ColumnTransformer([("scale",StandardScaler(),["Time","Amount"])],remainder="passthrough")),("model",XGBClassifier(n_estimators=220,max_depth=4,learning_rate=0.06,subsample=0.85,colsample_bytree=0.85,scale_pos_weight=ratio,eval_metric="logloss",n_jobs=2,random_state=42))])}
    comparisons={}
    fitted={}
    for name,model in models.items():
        model.fit(xtr,ytr)
        fitted[name]=model
        comparisons[name]=metrics(yva,model.predict_proba(xva)[:,1])
    winner=max(["logistic","xgboost"],key=lambda n:comparisons[n]["average_precision"])
    model=fitted[winner]
    threshold=choose_threshold(yva,model.predict_proba(xva)[:,1],min_precision)
    test=metrics(yte,model.predict_proba(xte)[:,1],threshold)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":model,"threshold":threshold,"features":FEATURES,"name":winner},output/"fraud_model.joblib")
    report={"dataset":"ULB Credit Card Fraud Detection","split":"chronological 60/20/20", "counts":{"train":len(tr),"validation":len(va),"test":len(te)},"validation_models":comparisons,"selected_model":winner,"threshold_from_validation":threshold,"min_validation_precision_requested":min_precision,"test_metrics":test,"notes":["Dataset Time covers ~2 days only; not representative of real deployment drift.","Scores are not calibrated probabilities.","Threshold chosen on validation only; test assessed once."]}
    (output/"evaluation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--csv",default="data/raw/creditcard.csv");p.add_argument("--out",default="models");p.add_argument("--min-precision",type=float,default=0.5)
    a=p.parse_args();train(a.csv,a.out,a.min_precision)

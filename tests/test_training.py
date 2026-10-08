import numpy as np
import pandas as pd
from src.fraudshield.train import chronological_split,choose_threshold,metrics

def test_chronological_split():
    df=pd.DataFrame({"Time":list(range(100,0,-1))})
    tr,va,te=chronological_split(df)
    assert (len(tr),len(va),len(te))==(60,20,20)
    assert tr.Time.max()<=va.Time.min()<=te.Time.min()
def test_metrics():
    out=metrics([0,1,0,1],[0.1,0.9,0.2,0.8])
    assert out["average_precision"]==1
    assert out["recall"]==1
    assert 0<=choose_threshold(np.array([0,1,0,1]),np.array([0.1,0.9,0.2,0.8]))<=1

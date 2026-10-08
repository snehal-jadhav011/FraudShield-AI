import pytest
from src.fraudshield.analytics import apply_threshold, sample_metrics

def test_threshold_boundary():
    assert apply_threshold([0.1,0.5,0.9],0.5).tolist()==[False,True,True]

def test_threshold_invalid():
    with pytest.raises(ValueError):apply_threshold([float("nan")],0.5)

def test_metrics_with_labels():
    result=sample_metrics([0,1,1,0],[0.1,0.9,0.8,0.2],0.5)
    assert result["precision"]==1
    assert result["recall"]==1
    assert result["flagged"]==2

def test_metrics_invalid_labels():
    with pytest.raises(ValueError):sample_metrics([0,2],[0.1,0.9],0.5)

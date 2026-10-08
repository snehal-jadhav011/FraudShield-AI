import pandas as pd
import pytest
from src.fraudshield.features import FEATURES,validate

def row():return {**{k:0.0 for k in FEATURES},"Class":1}
def test_valid():assert validate(pd.DataFrame([row()])).shape==(1,31)
def test_missing():
    d=row();del d["V1"]
    with pytest.raises(ValueError,match="Missing"):validate(pd.DataFrame([d]))
def test_negative_amount():
    d=row();d["Amount"]=-1
    with pytest.raises(ValueError,match="nonnegative"):validate(pd.DataFrame([d]))
def test_invalid_label():
    d=row();d["Class"]=2
    with pytest.raises(ValueError,match="Class"):validate(pd.DataFrame([d]))

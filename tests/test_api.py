from fastapi.testclient import TestClient
from api.main import app
from src.fraudshield.features import FEATURES

def test_api_schema_rejects_missing_features():
    resp=TestClient(app).post("/predict",json={"Time":0,"Amount":10})
    assert resp.status_code==422

def test_api_schema_rejects_negative_amount():
    row={key:0.0 for key in FEATURES};row["Amount"]=-2
    assert TestClient(app).post("/predict",json=row).status_code==422

def test_health_contract():
    resp=TestClient(app).get("/health")
    assert resp.status_code==200
    assert resp.json()["status"] in {"ready","model_missing"}

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field,ConfigDict
from src.fraudshield.features import FEATURES
from src.fraudshield.predict import load_model,score

class Transaction(BaseModel):
    model_config=ConfigDict(extra="forbid")
    Time:float=Field(ge=0)
    Amount:float=Field(ge=0)
    V1:float;V2:float;V3:float;V4:float;V5:float;V6:float;V7:float
    V8:float;V9:float;V10:float;V11:float;V12:float;V13:float;V14:float
    V15:float;V16:float;V17:float;V18:float;V19:float;V20:float;V21:float
    V22:float;V23:float;V24:float;V25:float;V26:float;V27:float;V28:float

app=FastAPI(title="FraudShield AI",version="1.0.0",description="Demo-only anonymised transaction risk scorer")
@app.get("/health")
def health():
    try:load_model();return {"status":"ready"}
    except FileNotFoundError:return {"status":"model_missing"}
@app.get("/model-info")
def model_info():
    try:a=load_model()
    except FileNotFoundError:raise HTTPException(503,"Model not trained yet")
    return {"model":a["name"],"threshold":a["threshold"],"required_features":FEATURES}
@app.post("/predict")
def predict(transaction:Transaction):
    try:return score(transaction.model_dump())
    except FileNotFoundError:raise HTTPException(503,"Model not trained yet")
    except ValueError as exc:raise HTTPException(422,str(exc))

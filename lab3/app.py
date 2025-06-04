from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle

class PredictRequest(BaseModel):
    t: float

class PredictResponse(BaseModel):
    y_pred: float

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "lab1", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    X = [[req.t]]
    y_pred = model.predict(X)[0]
    return PredictResponse(y_pred=float(y_pred))
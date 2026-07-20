"""
main.py
-------
FastAPI + PyTorch LSTM microservice for Drishyam.

Endpoints match exactly what drishyam-backend/src/services/mlService.js
already calls:

    POST /predict   { region, month }              -> forecast
    POST /scenario  { region, rainfallDelta, tempDelta } -> forecast

If a region hasn't been trained yet, both return 404 — mlService.js
already treats any non-OK response as "service unavailable" and falls
back to its own local trend estimate, so an untrained region never
breaks the app, it just quietly uses the fallback until you train it.

RUN LOCALLY
-----------
    cd ml-service
    pip install -r requirements.txt
    cp .env.example .env
    python -m app.train --region MH     # train at least one region first
    uvicorn app.main:app --reload --port 8000

Then in drishyam-backend/.env:
    ML_SERVICE_URL=http://localhost:8000
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import inference

app = FastAPI(title="Drishyam ML Service", version="0.1.0")


class PredictRequest(BaseModel):
    region: str
    month: str


class ScenarioRequest(BaseModel):
    region: str
    rainfallDelta: float
    tempDelta: float


@app.get("/health")
def health():
    return {"ok": True, "service": "drishyam-ml-service", "trained_regions": inference.available_regions()}


@app.get("/regions")
def regions():
    return {"regions": inference.available_regions()}


@app.post("/predict")
def predict(body: PredictRequest):
    result = inference.predict(body.region, body.month)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for region '{body.region}'. "
                   f"Run: python -m app.train --region {body.region.upper()}",
        )
    return result


@app.post("/scenario")
def scenario(body: ScenarioRequest):
    result = inference.run_scenario(body.region, body.rainfallDelta, body.tempDelta)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for region '{body.region}'. "
                   f"Run: python -m app.train --region {body.region.upper()}",
        )
    return result

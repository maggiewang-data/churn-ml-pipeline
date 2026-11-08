"""REST API for real-time churn scoring.

Run:   uvicorn churn.api:app --app-dir src --reload
Docs:  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import SchemaError
from .predict import load_artifact, score

YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 171.0}}}


class Prediction(BaseModel):
    churn_probability: float
    send_retention_offer: bool
    risk_tier: Literal["low", "medium", "high"]


app = FastAPI(title="Customer Churn Scoring API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    meta = load_artifact()["metadata"]
    return {"status": "ok", "model": meta["model"], "trained_at": meta["trained_at"],
            "threshold": load_artifact()["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict_one(customer: Customer) -> Prediction:
    return predict_batch([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]) -> list[Prediction]:
    if not customers:
        raise HTTPException(status_code=422, detail="Empty request")
    if len(customers) > 1000:
        raise HTTPException(status_code=413, detail="Max 1000 customers per request")
    try:
        out = score(pd.DataFrame([c.model_dump() for c in customers]))
    except SchemaError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return [Prediction(**row) for row in out.to_dict(orient="records")]

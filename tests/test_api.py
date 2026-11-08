"""API contract tests. Requires a trained model (`make train`)."""

import pytest
from fastapi.testclient import TestClient

from churn.api import Customer, app
from churn.config import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `make train` first")
client = TestClient(app)
EXAMPLE = Customer.model_config["json_schema_extra"]["example"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_probability():
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_tier"] in {"low", "medium", "high"}


def test_high_risk_profile_scores_above_low_risk_profile():
    loyal = {**EXAMPLE, "tenure": 70, "Contract": "Two year",
             "PaymentMethod": "Credit card (automatic)", "InternetService": "DSL",
             "OnlineSecurity": "Yes", "TechSupport": "Yes",
             "MonthlyCharges": 60.0, "TotalCharges": 4200.0}
    p_new = client.post("/predict", json=EXAMPLE).json()["churn_probability"]
    p_loyal = client.post("/predict", json=loyal).json()["churn_probability"]
    assert p_new > p_loyal


def test_batch_endpoint():
    r = client.post("/predict/batch", json=[EXAMPLE, EXAMPLE])
    assert r.status_code == 200 and len(r.json()) == 2


def test_rejects_invalid_input():
    bad = {**EXAMPLE, "Contract": "Ten year"}
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "tenure": -1}).status_code == 422

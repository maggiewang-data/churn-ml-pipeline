"""Load the trained artifact and score customers."""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from .config import MODEL_PATH, RAW_FEATURES
from .data import clean, validate


@lru_cache(maxsize=1)
def load_artifact(path: str = str(MODEL_PATH)) -> dict:
    return joblib.load(path)


def risk_tier(p: float) -> str:
    if p >= 0.6:
        return "high"
    if p >= 0.3:
        return "medium"
    return "low"


def score(records: pd.DataFrame) -> pd.DataFrame:
    art = load_artifact()
    df = clean(records)
    validate(df, require_target=False)
    proba = art["model"].predict_proba(df[RAW_FEATURES])[:, 1]
    return pd.DataFrame({
        "churn_probability": proba.round(4),
        "send_retention_offer": proba >= art["threshold"],
        "risk_tier": [risk_tier(p) for p in proba],
    }, index=records.index)

"""Feature engineering, packaged as sklearn transformers so the exact same
logic runs in training and in the API (no train/serve skew)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
ENGINEERED_NUMERIC = ["num_addon_services", "avg_monthly_spend",
                      "charge_increase_ratio", "is_new_customer"]


class ChurnFeatures(BaseEstimator, TransformerMixin):
    """Adds domain-driven features:
    - num_addon_services: more services = higher switching cost
    - avg_monthly_spend: TotalCharges / tenure (lifetime average bill)
    - charge_increase_ratio: current bill vs. lifetime average (price shock)
    - is_new_customer: first 6 months is the highest-risk window
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["num_addon_services"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)
        tenure = X["tenure"].replace(0, np.nan)
        X["avg_monthly_spend"] = (X["TotalCharges"] / tenure).fillna(X["MonthlyCharges"])
        X["charge_increase_ratio"] = X["MonthlyCharges"] / X["avg_monthly_spend"].replace(0, np.nan)
        X["charge_increase_ratio"] = X["charge_increase_ratio"].fillna(1.0)
        X["is_new_customer"] = (X["tenure"] <= 6).astype(int)
        return X


def build_preprocessor() -> Pipeline:
    numeric = NUMERIC_FEATURES + ENGINEERED_NUMERIC + BINARY_FEATURES
    encode = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    return Pipeline([("features", ChurnFeatures()), ("encode", encode)])

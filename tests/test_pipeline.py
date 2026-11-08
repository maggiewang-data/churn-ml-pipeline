"""Tests for data validation, feature engineering, and model behavior."""

import numpy as np
import pandas as pd
import pytest

from churn.config import RAW_FEATURES
from churn.data import SchemaError, clean, load_dataset, validate
from churn.features import ChurnFeatures, build_preprocessor
from churn.train import choose_threshold, expected_profit


@pytest.fixture(scope="module")
def df():
    return load_dataset()


def test_clean_handles_blank_total_charges():
    raw = pd.DataFrame({"TotalCharges": [" ", "10.5"], "Churn": ["Yes", "No"]})
    out = clean(raw)
    assert out["TotalCharges"].tolist() == [0.0, 10.5]
    assert out["Churn"].tolist() == [1, 0]


def test_dataset_is_valid_and_unique(df):
    assert df["customerID"].is_unique
    assert df["Churn"].isin([0, 1]).all()
    assert 0.25 < df["Churn"].mean() < 0.28


def test_validate_rejects_missing_column(df):
    with pytest.raises(SchemaError, match="Missing columns"):
        validate(df.drop(columns=["Contract"]))


def test_validate_rejects_negative_values(df):
    bad = df.copy()
    bad.loc[bad.index[0], "MonthlyCharges"] = -5
    with pytest.raises(SchemaError, match="negative"):
        validate(bad)


def test_engineered_features(df):
    out = ChurnFeatures().transform(df[RAW_FEATURES].head(200))
    assert out["num_addon_services"].between(0, 6).all()
    assert out["is_new_customer"].isin([0, 1]).all()
    assert np.isfinite(out[["avg_monthly_spend", "charge_increase_ratio"]]).all().all()


def test_preprocessor_handles_unseen_category(df):
    prep = build_preprocessor().fit(df[RAW_FEATURES])
    row = df[RAW_FEATURES].head(1).copy()
    row["PaymentMethod"] = "Crypto"          # never seen in training
    assert prep.transform(row).shape[0] == 1


def test_profit_logic():
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.9, 0.8, 0.2, 0.1])
    monthly = np.array([100, 100, 100, 100])
    # threshold 0.5 flags rows 0 and 1: (0.3*12*100 - 40) + (0 - 40) = 280
    assert expected_profit(y, proba, monthly, 0.5) == pytest.approx(280)
    t, curve = choose_threshold(y, proba, monthly)
    assert curve.profit.max() == pytest.approx(expected_profit(y, proba, monthly, t))

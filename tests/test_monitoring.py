"""Tests for drift detection and the retraining policy."""

import numpy as np
import pandas as pd
import pytest

from churn.monitoring import (
    BatchReport,
    categorical_shares,
    numeric_bins,
    numeric_shares,
    psi_from_shares,
)


def test_psi_zero_for_identical_distributions():
    shares = np.array([0.2, 0.3, 0.5])
    assert psi_from_shares(shares, shares) == pytest.approx(0.0)


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 20_000))
    edges = numeric_bins(ref)
    base = numeric_shares(ref, edges)
    small = psi_from_shares(base, numeric_shares(pd.Series(rng.normal(0.1, 1, 20_000)), edges))
    large = psi_from_shares(base, numeric_shares(pd.Series(rng.normal(1.0, 1, 20_000)), edges))
    assert small < 0.10 < 0.25 < large


def test_unseen_category_goes_to_other_bin():
    shares = categorical_shares(pd.Series(["a", "b", "new"]), ["a", "b"])
    assert shares.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_drift_alone_means_investigate_not_retrain():
    rep = BatchReport("m", 100, {"x": 0.4}, 0.05, drift_alerts=["input drift in x"])
    assert rep.action == "INVESTIGATE"


def test_performance_alert_means_retrain():
    rep = BatchReport("m", 100, {"x": 0.01}, 0.01, performance_alerts=["calibration gap"])
    assert rep.action == "RETRAIN"


def test_quiet_batch_is_ok():
    assert BatchReport("m", 100, {"x": 0.01}, 0.01).action == "OK"

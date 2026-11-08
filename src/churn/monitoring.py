"""Post-deployment monitoring: is the model still fit for purpose?

A model trained once degrades silently. Customers change, prices change, and
the relationship between features and churn changes. This module checks three
things for every new batch of scored customers, in the order they become
observable:

1. Data drift (available immediately): have the INPUTS shifted?
   Population Stability Index (PSI) per feature vs. the training data.
2. Prediction drift (available immediately): has the SCORE distribution shifted?
3. Performance (available only once outcomes arrive, about a month later):
   - PR-AUC: can the model still rank churners above non-churners?
   - Calibration gap: does "30% churn risk" still mean 30%? This matters
     because the offer threshold is a dollar calculation on these probabilities.

Two tiers of response, decided in advance rather than by eyeballing a dashboard:
- INVESTIGATE: inputs or scores drifted. Drift is an early warning, not proof of
  harm. A shifted customer mix can still be scored perfectly well.
- RETRAIN: measured performance or calibration has actually degraded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import CATEGORICAL_FEATURES, MODEL_DIR, NUMERIC_FEATURES

REFERENCE_PATH = MODEL_DIR / "reference_profile.json"

# PSI conventions used widely in credit-risk model monitoring
PSI_WATCH = 0.10      # moderate shift: investigate
PSI_ACT = 0.25        # major shift: the model is scoring a population it wasn't trained on

INVESTIGATE_RULES = {
    "max_feature_psi": PSI_ACT,      # any input shifts a lot
    "max_score_psi": PSI_ACT,        # the score distribution itself shifts a lot
}
RETRAIN_RULES = {
    "max_pr_auc_drop": 0.05,         # ranking quality falls > 5 points vs. launch
    "max_calibration_gap": 0.05,     # predicted vs. actual churn rate differ by > 5 points
}
EPS = 1e-4


# --------------------------------------------------------------------------- #
# PSI
# --------------------------------------------------------------------------- #
def psi_from_shares(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI = sum((a - e) * ln(a / e)) over bins. 0 = identical distributions."""
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_bins(reference: pd.Series, n_bins: int = 10) -> list[float]:
    """Quantile bin edges from the reference data (inner edges only)."""
    qs = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, n_bins + 1))[1:-1])
    return qs.tolist()


def numeric_shares(values: pd.Series, edges: list[float]) -> np.ndarray:
    idx = np.searchsorted(edges, values.to_numpy(), side="right")
    return np.bincount(idx, minlength=len(edges) + 1) / len(values)


def categorical_shares(values: pd.Series, levels: list[str]) -> np.ndarray:
    counts = values.astype(str).value_counts()
    shares = np.array([counts.get(lvl, 0) for lvl in levels], float)
    other = len(values) - shares.sum()          # unseen categories land in an "other" bin
    return np.append(shares, other) / len(values)


# --------------------------------------------------------------------------- #
# Reference profile: a snapshot of the training population, saved at train time
# --------------------------------------------------------------------------- #
def build_reference_profile(X_train: pd.DataFrame, train_scores: np.ndarray,
                            launch_pr_auc: float) -> dict:
    prof = {"numeric": {}, "categorical": {}, "launch_pr_auc": launch_pr_auc}
    for col in NUMERIC_FEATURES:
        edges = numeric_bins(X_train[col])
        prof["numeric"][col] = {"edges": edges,
                                "shares": numeric_shares(X_train[col], edges).tolist()}
    for col in CATEGORICAL_FEATURES:
        levels = sorted(X_train[col].astype(str).unique())
        prof["categorical"][col] = {"levels": levels,
                                    "shares": categorical_shares(X_train[col], levels).tolist()}
    s = pd.Series(train_scores)
    edges = numeric_bins(s)
    prof["score"] = {"edges": edges, "shares": numeric_shares(s, edges).tolist()}
    return prof


def save_reference_profile(profile: dict) -> None:
    REFERENCE_PATH.write_text(json.dumps(profile, indent=2))


def load_reference_profile() -> dict:
    return json.loads(REFERENCE_PATH.read_text())


# --------------------------------------------------------------------------- #
# Batch check
# --------------------------------------------------------------------------- #
@dataclass
class BatchReport:
    batch: str
    n: int
    feature_psi: dict[str, float]
    score_psi: float
    pr_auc: float | None = None
    calibration_gap: float | None = None
    drift_alerts: list[str] = field(default_factory=list)
    performance_alerts: list[str] = field(default_factory=list)

    @property
    def action(self) -> str:
        if self.performance_alerts:
            return "RETRAIN"
        if self.drift_alerts:
            return "INVESTIGATE"
        if max(self.feature_psi.values()) > PSI_WATCH or self.score_psi > PSI_WATCH:
            return "WATCH"
        return "OK"

    def summary_row(self) -> dict:
        top = max(self.feature_psi, key=self.feature_psi.get)
        return {"batch": self.batch, "n": self.n,
                "top_drift_feature": top, "top_feature_psi": round(self.feature_psi[top], 3),
                "score_psi": round(self.score_psi, 3),
                "pr_auc": None if self.pr_auc is None else round(self.pr_auc, 3),
                "calibration_gap": None if self.calibration_gap is None
                else round(self.calibration_gap, 3),
                "action": self.action,
                "why": "; ".join(self.performance_alerts + self.drift_alerts)}


def check_batch(batch_name: str, X: pd.DataFrame, scores: np.ndarray,
                profile: dict, y_true: pd.Series | None = None) -> BatchReport:
    feature_psi = {}
    for col, ref in profile["numeric"].items():
        feature_psi[col] = psi_from_shares(ref["shares"], numeric_shares(X[col], ref["edges"]))
    for col, ref in profile["categorical"].items():
        feature_psi[col] = psi_from_shares(ref["shares"], categorical_shares(X[col], ref["levels"]))
    score_psi = psi_from_shares(profile["score"]["shares"],
                                numeric_shares(pd.Series(scores), profile["score"]["edges"]))
    rep = BatchReport(batch_name, len(X), feature_psi, score_psi)

    worst = max(feature_psi, key=feature_psi.get)
    if feature_psi[worst] > INVESTIGATE_RULES["max_feature_psi"]:
        rep.drift_alerts.append(f"input drift in {worst} (PSI {feature_psi[worst]:.2f})")
    if score_psi > INVESTIGATE_RULES["max_score_psi"]:
        rep.drift_alerts.append(f"score drift (PSI {score_psi:.2f})")

    if y_true is not None:
        rep.pr_auc = float(average_precision_score(y_true, scores))
        rep.calibration_gap = float(abs(np.mean(scores) - np.mean(y_true)))
        drop = profile["launch_pr_auc"] - rep.pr_auc
        if drop > RETRAIN_RULES["max_pr_auc_drop"]:
            rep.performance_alerts.append(f"PR-AUC down {drop:.3f} vs. launch")
        if rep.calibration_gap > RETRAIN_RULES["max_calibration_gap"]:
            rep.performance_alerts.append(f"calibration gap {rep.calibration_gap:.3f} "
                              f"(predicted {np.mean(scores):.1%} vs. actual {np.mean(y_true):.1%})")
    return rep

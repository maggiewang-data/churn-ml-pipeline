"""Train, compare, calibrate and save the churn model.

1. Compare 3 candidate models with stratified 5-fold CV (metric: PR-AUC)
2. Refit the winner with probability calibration
3. Pick the decision threshold that maximizes expected retention profit
4. Evaluate ONCE on the held-out test set and save model + metrics

Run:  python -m churn.train      (from the src/ directory, or via `make train`)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import (
    FIG_DIR,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    MONTHS_OF_VALUE_SAVED,
    OFFER_COST,
    OFFER_SUCCESS_RATE,
    RANDOM_STATE,
)
from .data import load_dataset, split
from .features import build_preprocessor


def candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, C=0.5),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=10, n_jobs=-1, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE),
    }


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def expected_profit(y_true, proba, monthly_charges, threshold: float) -> float:
    """Profit of sending an offer to everyone with proba >= threshold.
    TP: 30% of true churners are saved -> 12 months of revenue, minus offer cost.
    FP: non-churners get an offer they didn't need -> offer cost only."""
    flagged = proba >= threshold
    saved_value = (OFFER_SUCCESS_RATE * MONTHS_OF_VALUE_SAVED
                   * np.asarray(monthly_charges) * np.asarray(y_true))
    return float(np.sum(flagged * (saved_value - OFFER_COST)))


def choose_threshold(y_true, proba, monthly_charges) -> tuple[float, pd.DataFrame]:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    curve = pd.DataFrame({
        "threshold": grid,
        "profit": [expected_profit(y_true, proba, monthly_charges, t) for t in grid],
    })
    return float(curve.loc[curve.profit.idxmax(), "threshold"]), curve


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    X_train, X_test, y_train, y_test = split(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- 1. Model comparison --------------------------------------------- #
    cv_results = {}
    for name, model in candidate_models().items():
        scores = cross_val_score(make_pipeline(model), X_train, y_train,
                                 cv=cv, scoring="average_precision", n_jobs=-1)
        cv_results[name] = {"pr_auc_mean": float(scores.mean()), "pr_auc_std": float(scores.std())}
        print(f"{name:20s} PR-AUC {scores.mean():.4f} +/- {scores.std():.4f}")
    best_name = max(cv_results, key=lambda k: cv_results[k]["pr_auc_mean"])
    print(f"Selected: {best_name}")

    # ---- 2. Calibrated final model --------------------------------------- #
    final = CalibratedClassifierCV(make_pipeline(candidate_models()[best_name]),
                                   method="isotonic", cv=5)

    # ---- 3. Threshold from out-of-fold predictions on TRAIN only --------- #
    oof = cross_val_predict(final, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold, profit_curve = choose_threshold(y_train, oof, X_train["MonthlyCharges"])
    print(f"Profit-maximizing threshold: {threshold:.2f}")

    final.fit(X_train, y_train)

    # ---- 4. One-time evaluation on the held-out test set ---------------- #
    proba = final.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    base_rate = float(y_test.mean())
    test_profit = expected_profit(y_test, proba, X_test["MonthlyCharges"], threshold)
    target_all = expected_profit(y_test, proba, X_test["MonthlyCharges"], 0.0)
    # Simple business-rule baseline: offer to every month-to-month customer
    rule = (X_test["Contract"] == "Month-to-month").astype(float).values
    rule_profit = expected_profit(y_test, rule, X_test["MonthlyCharges"], 0.5)
    metrics = {
        "model": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "test_churn_rate": base_rate,
        "cv_comparison": cv_results,
        "threshold": threshold,
        "test": {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
            "brier": float(brier_score_loss(y_test, proba)),
            "precision_at_threshold": float(precision_score(y_test, pred)),
            "recall_at_threshold": float(recall_score(y_test, pred)),
            "share_flagged": float(pred.mean()),
            "expected_profit_usd": test_profit,
            "profit_if_offer_to_everyone_usd": target_all,
            "profit_rule_month_to_month_usd": rule_profit,
        },
        "business_assumptions": {
            "offer_cost": OFFER_COST, "offer_success_rate": OFFER_SUCCESS_RATE,
            "months_of_value_saved": MONTHS_OF_VALUE_SAVED,
        },
    }
    joblib.dump({"model": final, "threshold": threshold, "metadata": metrics}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["test"], indent=2))

    # ---- Figures ---------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(profit_curve.threshold, profit_curve.profit / 1000, lw=2)
    ax.axvline(threshold, color="crimson", ls="--", label=f"Chosen threshold = {threshold:.2f}")
    ax.set_xlabel("Decision threshold (churn probability)")
    ax.set_ylabel("Expected profit, $K (training OOF)")
    ax.set_title("Choosing the threshold by business value")
    ax.legend(frameon=False)
    ax = axes[1]
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], color="grey", ls=":", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", lw=2, label="Model (test set)")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration")
    ax.legend(frameon=False)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_and_calibration.png", dpi=150)
    plt.close(fig)

    imp = permutation_importance(final, X_test, y_test, scoring="average_precision",
                                 n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    top = (pd.Series(imp.importances_mean, index=X_test.columns)
             .sort_values().tail(10))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top.index, top.values)
    ax.set_xlabel("Drop in PR-AUC when feature is shuffled")
    ax.set_title("Top 10 drivers of churn predictions (permutation importance)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

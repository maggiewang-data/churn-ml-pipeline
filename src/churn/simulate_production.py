"""Simulate six months of production traffic and run the monitor on each month.

The held-out test customers stand in for "new customers arriving after launch".
Each month is a resample of them with a realistic change layered on top, so the
monitor's response to each kind of change can be checked:

  Month 1-2  business as usual                         -> should stay quiet
  Month 3    marketing push brings in more month-to-month customers
             (inputs shift, but the churn relationship holds)
  Month 4    fiber price increase: bills go up AND fiber customers become
             more likely to leave than the model learned (concept drift)
  Month 5    a new payment method launches that the model has never seen
  Month 6    back to normal mix

Run:  python -m churn.simulate_production    (after training)
Writes: reports/monitoring_report.md, figures/monitoring.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR, RAW_FEATURES, ROOT
from .data import load_dataset, split
from .monitoring import (
    INVESTIGATE_RULES,
    PSI_ACT,
    PSI_WATCH,
    RETRAIN_RULES,
    check_batch,
    load_reference_profile,
)
from .predict import load_artifact

BATCH_SIZE = 1500
REPORT_DIR = ROOT / "reports"


def make_months(X: pd.DataFrame, y: pd.Series,
                seed: int = 7) -> list[tuple[str, pd.DataFrame, pd.Series]]:
    rng = np.random.default_rng(seed)
    base = X.assign(_y=y.values).reset_index(drop=True)

    def sample(weights=None):
        p = None if weights is None else weights / weights.sum()
        idx = rng.choice(len(base), BATCH_SIZE, replace=True, p=p)
        return base.iloc[idx].reset_index(drop=True).copy()

    months = []
    for name in ["M1 normal", "M2 normal"]:
        months.append((name, sample()))

    w = np.where(base.Contract == "Month-to-month", 3.0, 1.0)
    months.append(("M3 m2m promo", sample(w)))

    m4 = sample()
    fiber = m4.InternetService == "Fiber optic"
    m4.loc[fiber, "MonthlyCharges"] *= 1.15
    flip = fiber & (m4._y == 0) & (rng.random(len(m4)) < 0.30)
    m4.loc[flip, "_y"] = 1
    months.append(("M4 fiber price +15%", m4))

    m5 = sample()
    swap = (m5.PaymentMethod == "Electronic check") & (rng.random(len(m5)) < 0.7)
    m5.loc[swap, "PaymentMethod"] = "Digital wallet"
    months.append(("M5 new payment method", m5))

    months.append(("M6 normal", sample()))
    return [(n, d[RAW_FEATURES], d["_y"]) for n, d in months]


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    art = load_artifact()
    profile = load_reference_profile()
    _, X_test, _, y_test = split(load_dataset())

    reports = []
    for name, Xb, yb in make_months(X_test, y_test):
        scores = art["model"].predict_proba(Xb)[:, 1]
        reports.append(check_batch(name, Xb, scores, profile, y_true=yb))
    table = pd.DataFrame([r.summary_row() for r in reports])

    md = ["# Monitoring report: 6 simulated months after launch\n",
          f"Launch PR-AUC: {profile['launch_pr_auc']:.3f}. "
          f"PSI thresholds: watch > {PSI_WATCH}, act > {PSI_ACT}. "
          f"Investigate rules: {INVESTIGATE_RULES}. Retrain rules: {RETRAIN_RULES}\n",
          table.to_markdown(index=False), ""]
    (REPORT_DIR / "monitoring_report.md").write_text("\n".join(md))
    print("\n".join(md))

    # ---- figure ---------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(reports))
    labels = [r.batch.split(" ", 1)[0] for r in reports]
    ax = axes[0]
    ax.plot(x, [max(r.feature_psi.values()) for r in reports], marker="o", lw=2,
            label="Largest input-feature PSI")
    ax.plot(x, [r.score_psi for r in reports], marker="s", lw=2, label="Score PSI")
    ax.axhline(PSI_WATCH, color="orange", ls="--", lw=1, label="Watch")
    ax.axhline(PSI_ACT, color="crimson", ls="--", lw=1, label="Act")
    ax.set_title("Drift you can see immediately")
    ax.set_ylabel("Population Stability Index")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.plot(x, [r.pr_auc for r in reports], marker="o", lw=2, label="PR-AUC")
    ax.axhline(profile["launch_pr_auc"] - RETRAIN_RULES["max_pr_auc_drop"], color="crimson",
               ls="--", lw=1, label="PR-AUC retrain line")
    ax2 = ax.twinx()
    ax2.bar(x, [r.calibration_gap for r in reports], alpha=0.25, color="grey",
            label="Calibration gap")
    ax2.axhline(RETRAIN_RULES["max_calibration_gap"], color="grey", ls=":", lw=1)
    ax2.set_ylabel("|predicted - actual churn rate|")
    ax.set_title("Performance once outcomes arrive")
    ax.set_ylabel("PR-AUC")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(labels)
        a.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monitoring.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

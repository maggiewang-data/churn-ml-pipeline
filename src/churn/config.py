"""Central configuration: paths, feature lists, and business assumptions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FIG_DIR = ROOT / "figures"

TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
RAW_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# ---- Business assumptions for choosing the decision threshold ------------ #
# A customer flagged as "at risk" receives a retention offer.
OFFER_COST = 40.0            # $ per offer (discount + outreach)
OFFER_SUCCESS_RATE = 0.30    # share of true churners the offer keeps
MONTHS_OF_VALUE_SAVED = 12   # value of a saved customer = 12 x MonthlyCharges

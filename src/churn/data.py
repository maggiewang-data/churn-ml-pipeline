"""Loading, validation and splitting."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    ID_COL,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RAW_FEATURES,
    TARGET,
)


class SchemaError(ValueError):
    """Raised when input data does not match the expected schema."""


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues in the raw Telco file."""
    df = df.copy()
    # TotalCharges is stored as text; 11 brand-new customers (tenure = 0)
    # have a blank value. They have not been billed yet, so 0 is correct.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip()
                                       .replace("", "0"), errors="coerce").fillna(0.0)
    if TARGET in df:
        df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    return df


def validate(df: pd.DataFrame, require_target: bool = True) -> None:
    """Fail loudly if columns are missing or numerics are invalid."""
    needed = RAW_FEATURES + ([TARGET] if require_target else [])
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise SchemaError(f"Missing columns: {missing}")
    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaError(f"Column {col} must be numeric")
        if (df[col] < 0).any():
            raise SchemaError(f"Column {col} has negative values")
    for col in CATEGORICAL_FEATURES:
        if df[col].isna().any():
            raise SchemaError(f"Column {col} has missing values")


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    df = clean(pd.read_csv(path))
    validate(df)
    return df.drop_duplicates(subset=ID_COL)


def split(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified train/test split so both sets keep the ~26.5% churn rate."""
    X, y = df[RAW_FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=RANDOM_STATE)

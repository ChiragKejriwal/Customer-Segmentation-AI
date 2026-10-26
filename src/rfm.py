"""
RFM (Recency, Frequency, Monetary) Module.
Computes customer-level RFM behavioral metrics and quintile scores.
"""

import datetime
from typing import Optional
import pandas as pd


def calculate_rfm(
    df: pd.DataFrame,
    reference_date: Optional[datetime.datetime] = None
) -> pd.DataFrame:
    """
    Aggregates transactional logs to customer-level RFM metrics:
    - Recency: Days between customer's last purchase and reference date.
    - Frequency: Count of unique invoices per customer.
    - Monetary: Total lifetime spend per customer.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["CustomerID", "Recency", "Frequency", "Monetary"])

    cleaned = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(cleaned["InvoiceDate"]):
        cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"])

    if reference_date is None:
        reference_date = cleaned["InvoiceDate"].max() + datetime.timedelta(days=1)
    elif isinstance(reference_date, str):
        reference_date = pd.to_datetime(reference_date)

    rfm = cleaned.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum")
    ).reset_index()

    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm


# Backward compatibility alias
compute_rfm = calculate_rfm


def score_rfm(rfm_df: pd.DataFrame, num_bins: int = 5) -> pd.DataFrame:
    """
    Assigns quintile (1-5) scores to Recency, Frequency, and Monetary metrics.
    Note: Recency is reverse-scored (lower recency = higher score).
    """
    if rfm_df is None or rfm_df.empty:
        return pd.DataFrame() if rfm_df is None else rfm_df.copy()

    df = rfm_df.copy()
    df["R_Score"] = pd.qcut(df["Recency"], q=num_bins, labels=range(num_bins, 0, -1), duplicates="drop").astype(int)
    df["F_Score"] = pd.qcut(df["Frequency"].rank(method="first"), q=num_bins, labels=range(1, num_bins + 1)).astype(int)
    df["M_Score"] = pd.qcut(df["Monetary"].rank(method="first"), q=num_bins, labels=range(1, num_bins + 1)).astype(int)

    df["RFM_Segment"] = df["R_Score"].astype(str) + df["F_Score"].astype(str) + df["M_Score"].astype(str)
    df["RFM_Score"] = df[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    return df

"""
RFM (Recency, Frequency, Monetary) Analysis Module.
Calculates RFM metrics and scores for customer segmentation.
"""

import datetime
from typing import Optional
import numpy as np
import pandas as pd


def calculate_rfm(
    df: pd.DataFrame,
    reference_date: Optional[datetime.datetime] = None
) -> pd.DataFrame:
    """
    Aggregates cleaned transaction data to the CustomerID level:
    - Recency: Days between the customer's last purchase and reference date (max date in dataset + 1 day).
    - Frequency: Count of unique InvoiceNo per customer.
    - Monetary: Sum of TotalAmount per customer.

    Args:
        df: Cleaned transaction DataFrame with CustomerID, InvoiceDate, InvoiceNo, TotalAmount.
        reference_date: Optional reference date to calculate recency (defaults to max date + 1 day).

    Returns:
        pd.DataFrame: RFM DataFrame with CustomerID, Recency, Frequency, and Monetary.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["CustomerID", "Recency", "Frequency", "Monetary"])

    # Ensure InvoiceDate is datetime
    cleaned_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(cleaned_df["InvoiceDate"]):
        cleaned_df["InvoiceDate"] = pd.to_datetime(cleaned_df["InvoiceDate"])

    # Determine reference date
    if reference_date is None:
        reference_date = cleaned_df["InvoiceDate"].max() + datetime.timedelta(days=1)
    elif isinstance(reference_date, str):
        reference_date = pd.to_datetime(reference_date)

    # Aggregate by CustomerID
    rfm = cleaned_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum")
    ).reset_index()

    # Ensure monetary is rounded
    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm


# Alias for backward compatibility
compute_rfm = calculate_rfm


def score_rfm(rfm_df: pd.DataFrame, num_bins: int = 5) -> pd.DataFrame:
    """
    Assigns quintile (1-5) scores for R, F, and M.
    Note: Recency is reverse-scored (lower recency days = higher score).

    Args:
        rfm_df: DataFrame with Recency, Frequency, Monetary columns.
        num_bins: Number of bins / quintiles (default 5).

    Returns:
        pd.DataFrame: RFM DataFrame with R_Score, F_Score, M_Score, and RFM_Score.
    """
    if rfm_df is None or rfm_df.empty:
        return pd.DataFrame() if rfm_df is None else rfm_df.copy()

    df = rfm_df.copy()

    # R Score: lower recency is better -> higher score
    df["R_Score"] = pd.qcut(df["Recency"], q=num_bins, labels=range(num_bins, 0, -1), duplicates="drop").astype(int)

    # F & M Scores: higher frequency/monetary is better -> higher score
    df["F_Score"] = pd.qcut(df["Frequency"].rank(method="first"), q=num_bins, labels=range(1, num_bins + 1)).astype(int)
    df["M_Score"] = pd.qcut(df["Monetary"].rank(method="first"), q=num_bins, labels=range(1, num_bins + 1)).astype(int)

    # Combined RFM Score string
    df["RFM_Segment"] = df["R_Score"].astype(str) + df["F_Score"].astype(str) + df["M_Score"].astype(str)
    df["RFM_Score"] = df[["R_Score", "F_Score", "M_Score"]].sum(axis=1)

    return df

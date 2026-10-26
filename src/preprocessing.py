"""
Data Preprocessing Module for Customer Segmentation.
Cleans raw transactional logs and computes derived line-item metrics.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw transactional records:
    1. Removes records with missing CustomerID.
    2. Drops cancelled orders (InvoiceNo starting with 'C') and negative quantities.
    3. Filters out non-positive unit prices.
    4. Computes line-item TotalAmount (Quantity * UnitPrice).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cleaned = df.copy()

    if "CustomerID" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["CustomerID"])
        try:
            cleaned["CustomerID"] = cleaned["CustomerID"].astype(int).astype(str)
        except (ValueError, TypeError):
            cleaned["CustomerID"] = cleaned["CustomerID"].astype(str)

    if "InvoiceNo" in cleaned.columns:
        invoice_str = cleaned["InvoiceNo"].astype(str).str.strip().str.upper()
        cleaned = cleaned[~invoice_str.str.startswith("C")]

    if "Quantity" in cleaned.columns:
        cleaned = cleaned[cleaned["Quantity"] > 0]

    if "UnitPrice" in cleaned.columns:
        cleaned = cleaned[cleaned["UnitPrice"] > 0]

    if "Quantity" in cleaned.columns and "UnitPrice" in cleaned.columns:
        cleaned["TotalAmount"] = (cleaned["Quantity"] * cleaned["UnitPrice"]).round(2)

    return cleaned.reset_index(drop=True)

"""
Data Preprocessing Module for Customer Segmentation Platform.
Cleans raw transactional data and creates essential derived features.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw transactional data according to standard retail analysis rules:
    1. Drops rows where `CustomerID` is null / NaN.
    2. Filters out cancelled orders (where `InvoiceNo` starts with 'C' or `Quantity` < 0).
    3. Filters out rows where `UnitPrice` <= 0.
    4. Creates `TotalAmount` column (`Quantity * UnitPrice`).

    Args:
        df: Raw transactional DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for RFM calculation and modeling.
    """
    if df is None or df.empty:
        logger.warning("Empty or None DataFrame passed to clean_data.")
        return pd.DataFrame()

    initial_count = len(df)
    logger.info(f"Starting data cleaning on {initial_count:,} records...")

    # Work on a copy to avoid SettingWithCopyWarning
    cleaned_df = df.copy()

    # 1. Drop rows where CustomerID is null
    if "CustomerID" in cleaned_df.columns:
        cleaned_df = cleaned_df.dropna(subset=["CustomerID"])
        # Format CustomerID as integer string or int if appropriate
        try:
            cleaned_df["CustomerID"] = cleaned_df["CustomerID"].astype(int).astype(str)
        except (ValueError, TypeError):
            cleaned_df["CustomerID"] = cleaned_df["CustomerID"].astype(str)
    else:
        logger.warning("Column 'CustomerID' not found in DataFrame.")

    # 2. Filter out cancelled orders: InvoiceNo starts with 'C' or Quantity < 0
    if "InvoiceNo" in cleaned_df.columns:
        invoice_str = cleaned_df["InvoiceNo"].astype(str).str.strip().str.upper()
        non_cancelled_invoice = ~invoice_str.str.startswith("C")
        cleaned_df = cleaned_df[non_cancelled_invoice]

    if "Quantity" in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df["Quantity"] > 0]

    # 3. Filter out rows where UnitPrice <= 0
    if "UnitPrice" in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df["UnitPrice"] > 0]

    # 4. Create TotalAmount column (Quantity * UnitPrice)
    if "Quantity" in cleaned_df.columns and "UnitPrice" in cleaned_df.columns:
        cleaned_df["TotalAmount"] = (cleaned_df["Quantity"] * cleaned_df["UnitPrice"]).round(2)

    final_count = len(cleaned_df)
    dropped_count = initial_count - final_count
    logger.info(
        f"Data cleaning complete. Retained {final_count:,} records ({dropped_count:,} dropped)."
    )

    return cleaned_df.reset_index(drop=True)

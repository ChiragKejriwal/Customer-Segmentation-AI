"""
End-to-end integration and behavioral test suite for customer segmentation pipeline.
"""

import datetime
import numpy as np
import pandas as pd
import pytest
from src.preprocessing import clean_data
from src.rfm import calculate_rfm


@pytest.fixture
def sample_raw_transactions():
    """Generates synthetic raw retail transactions including anomalies and negative quantities."""
    return pd.DataFrame({
        "InvoiceNo": ["536365", "536366", "C536367", "536368", "536369"],
        "StockCode": ["85123A", "71053", "84406B", "84029G", "22752"],
        "Description": ["Product A", "Product B", "Product C (Cancelled)", "Product D (Returned)", "Product E"],
        "Quantity": [10, 5, -2, -10, 8],
        "InvoiceDate": [
            datetime.datetime(2023, 1, 10, 10, 0),
            datetime.datetime(2023, 1, 15, 12, 30),
            datetime.datetime(2023, 1, 16, 14, 0),
            datetime.datetime(2023, 1, 18, 9, 15),
            datetime.datetime(2023, 1, 20, 16, 45)
        ],
        "UnitPrice": [2.50, 4.00, 15.00, 5.00, 12.50],
        "CustomerID": ["10001", "10001", "10001", "10002", "10002"],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "Germany", "Germany"]
    })


def test_clean_data_removes_negative_quantities(sample_raw_transactions):
    """Ensure clean_data correctly filters out cancelled orders and negative quantities."""
    cleaned = clean_data(sample_raw_transactions)

    # 1. No negative or zero quantities should exist
    assert (cleaned["Quantity"] <= 0).sum() == 0
    assert all(cleaned["Quantity"] > 0)

    # 2. No cancelled invoices (starting with 'C') should exist
    assert not any(cleaned["InvoiceNo"].astype(str).str.startswith("C"))

    # 3. Expected valid rows: Invoice 536365 (qty 10), 536366 (qty 5), 536369 (qty 8)
    assert len(cleaned) == 3
    assert set(cleaned["InvoiceNo"]) == {"536365", "536366", "536369"}


def test_calculate_rfm_recency_and_monetary(sample_raw_transactions):
    """Ensure calculate_rfm correctly calculates positive Recency (> 0) and accurate Monetary values."""
    cleaned = clean_data(sample_raw_transactions)
    rfm = calculate_rfm(cleaned)

    assert not rfm.empty
    assert set(rfm.columns) == {"CustomerID", "Recency", "Frequency", "Monetary"}

    # Recency should strictly be greater than 0 since reference_date = max_date + 1 day
    assert (rfm["Recency"] > 0).all()

    # Verify Customer 10001:
    # Purchases:
    # 536365: 10 * 2.50 = 25.00
    # 536366: 5 * 4.00 = 20.00
    # Total Monetary = 45.00
    # Last purchase date: 2023-01-15 -> Reference date: 2023-01-21 -> Recency: 6 days
    # Frequency: 2 distinct invoices
    c1 = rfm[rfm["CustomerID"] == "10001"].iloc[0]
    assert c1["Recency"] == 6
    assert c1["Recency"] > 0
    assert c1["Frequency"] == 2
    assert c1["Monetary"] == 45.00

    # Verify Customer 10002:
    # Purchases:
    # 536369: 8 * 12.50 = 100.00
    # Last purchase date: 2023-01-20 -> Reference date: 2023-01-21 -> Recency: 1 day
    # Frequency: 1 invoice
    c2 = rfm[rfm["CustomerID"] == "10002"].iloc[0]
    assert c2["Recency"] == 1
    assert c2["Recency"] > 0
    assert c2["Frequency"] == 1
    assert c2["Monetary"] == 100.00

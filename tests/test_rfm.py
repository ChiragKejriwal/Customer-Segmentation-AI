"""
Unit tests for RFM calculation module.
"""

import datetime
import pandas as pd
import numpy as np
import pytest
from src.rfm import calculate_rfm, score_rfm


@pytest.fixture
def sample_transactions():
    """Generates synthetic cleaned transactional dataset."""
    return pd.DataFrame({
        "InvoiceNo": ["1001", "1002", "1003", "1004", "1005", "1006"],
        "StockCode": ["A", "B", "C", "D", "E", "F"],
        "Description": ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6"],
        "Quantity": [2, 1, 5, 3, 1, 2],
        "InvoiceDate": [
            datetime.datetime(2023, 1, 1),
            datetime.datetime(2023, 1, 10),
            datetime.datetime(2023, 1, 15),
            datetime.datetime(2023, 1, 1),
            datetime.datetime(2023, 1, 20),
            datetime.datetime(2023, 1, 20)
        ],
        "UnitPrice": [10.0, 20.0, 5.0, 15.0, 50.0, 25.0],
        "CustomerID": ["C1", "C1", "C2", "C3", "C3", "C3"],
        "TotalAmount": [20.0, 20.0, 25.0, 45.0, 50.0, 50.0]
    })


def test_calculate_rfm(sample_transactions):
    # Max date in dataset is 2023-01-20 -> Reference date is 2023-01-21
    rfm = calculate_rfm(sample_transactions)

    assert set(rfm.columns) == {"CustomerID", "Recency", "Frequency", "Monetary"}
    assert len(rfm) == 3

    # Check C1:
    # Last purchase: 2023-01-10 -> Recency: 2023-01-21 - 2023-01-10 = 11 days
    # Frequency: Invoices 1001, 1002 = 2
    # Monetary: 20.0 + 20.0 = 40.0
    c1 = rfm[rfm["CustomerID"] == "C1"].iloc[0]
    assert c1["Recency"] == 11
    assert c1["Frequency"] == 2
    assert c1["Monetary"] == 40.0

    # Check C2:
    # Last purchase: 2023-01-15 -> Recency: 2023-01-21 - 2023-01-15 = 6 days
    # Frequency: Invoice 1003 = 1
    # Monetary: 25.0
    c2 = rfm[rfm["CustomerID"] == "C2"].iloc[0]
    assert c2["Recency"] == 6
    assert c2["Frequency"] == 1
    assert c2["Monetary"] == 25.0

    # Check C3:
    # Last purchase: 2023-01-20 -> Recency: 2023-01-21 - 2023-01-20 = 1 day
    # Frequency: Invoices 1004, 1005, 1006 = 3
    # Monetary: 45 + 50 + 50 = 145.0
    c3 = rfm[rfm["CustomerID"] == "C3"].iloc[0]
    assert c3["Recency"] == 1
    assert c3["Frequency"] == 3
    assert c3["Monetary"] == 145.0


def test_calculate_rfm_empty():
    empty_df = pd.DataFrame()
    rfm = calculate_rfm(empty_df)
    assert rfm.empty
    assert "CustomerID" in rfm.columns
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns


def test_score_rfm(sample_transactions):
    rfm = calculate_rfm(sample_transactions)
    scored = score_rfm(rfm, num_bins=3)
    assert "R_Score" in scored.columns
    assert "F_Score" in scored.columns
    assert "M_Score" in scored.columns
    assert "RFM_Score" in scored.columns
    assert "RFM_Segment" in scored.columns

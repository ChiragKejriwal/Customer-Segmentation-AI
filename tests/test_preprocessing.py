"""
Unit tests for data_loader and preprocessing modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_data
from src.preprocessing import clean_data


@pytest.fixture
def sample_raw_data(tmp_path):
    """Creates a temporary CSV file mimicking UCI Online Retail dataset."""
    csv_file = tmp_path / "test_retail.csv"
    data = """InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,12/1/2010 8:26,2.55,17850,United Kingdom
536365,71053,WHITE METAL LANTERN,6,12/1/2010 8:26,3.39,17850,United Kingdom
C536379,D,Discount,-1,12/1/2010 9:41,27.50,14527,United Kingdom
536381,21523,DOORMAT FANCY FONT,1,12/1/2010 9:41,7.95,,United Kingdom
536382,21730,GLASS STAR FROSTED T-LIGHT HOLDER,0,12/1/2010 9:45,4.25,13047,United Kingdom
536383,21731,GLASS STAR FROSTED T-LIGHT HOLDER,2,12/1/2010 9:45,-1.50,13047,United Kingdom
536384,22492,MINI PAINT SET VINTAGE,10,12/1/2010 9:50,0.65,13047,United Kingdom
"""
    csv_file.write_text(data)
    return csv_file


def test_load_data_success(sample_raw_data):
    df = load_data(sample_raw_data)
    assert not df.empty
    assert "InvoiceDate" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"])


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file_xyz_123.csv")


def test_clean_data(sample_raw_data):
    raw_df = load_data(sample_raw_data)
    cleaned = clean_data(raw_df)

    # 1. Null CustomerID dropped (row with missing CustomerID)
    assert cleaned["CustomerID"].isnull().sum() == 0

    # 2. Cancelled orders (InvoiceNo startswith 'C' or Quantity <= 0) dropped
    assert not any(cleaned["InvoiceNo"].astype(str).str.startswith("C"))
    assert (cleaned["Quantity"] <= 0).sum() == 0

    # 3. UnitPrice <= 0 dropped
    assert (cleaned["UnitPrice"] <= 0).sum() == 0

    # 4. TotalAmount column calculated correctly
    assert "TotalAmount" in cleaned.columns
    expected_totals = (cleaned["Quantity"] * cleaned["UnitPrice"]).round(2)
    pd.testing.assert_series_equal(cleaned["TotalAmount"], expected_totals, check_names=False)

    # From 7 rows:
    # row 1: valid (6 * 2.55 = 15.30)
    # row 2: valid (6 * 3.39 = 20.34)
    # row 3: C536379 -> dropped (cancelled)
    # row 4: missing CustomerID -> dropped
    # row 5: Quantity=0 -> dropped
    # row 6: UnitPrice=-1.50 -> dropped
    # row 7: valid (10 * 0.65 = 6.50)
    assert len(cleaned) == 3

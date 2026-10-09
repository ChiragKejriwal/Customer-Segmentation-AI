"""
Unit tests for CSV Schema Inspection and Template Generation in Customer Segmentation.
"""

import pandas as pd
from src.data_loader import inspect_uploaded_df, get_sample_template_df, MANDATORY_COLUMNS


def test_get_sample_template_df():
    sample_df = get_sample_template_df()
    assert isinstance(sample_df, pd.DataFrame)
    assert not sample_df.empty
    for col in MANDATORY_COLUMNS:
        assert col in sample_df.columns


def test_inspect_uploaded_df_valid():
    df_valid = pd.DataFrame({
        "InvoiceNo": ["536365", "536366"],
        "StockCode": ["85123A", "71053"],
        "Description": ["Product A", "Product B"],
        "Quantity": [6, 12],
        "InvoiceDate": ["2023-12-01 08:26:00", "2023-12-01 08:28:00"],
        "UnitPrice": [2.55, 1.85],
        "CustomerID": ["17850", "13047"],
        "Country": ["United Kingdom", "United Kingdom"]
    })
    res = inspect_uploaded_df(df_valid)
    assert res["is_valid"] is True
    assert len(res["missing_mandatory"]) == 0
    assert len(res["errors"]) == 0


def test_inspect_uploaded_df_synonyms():
    df_synonyms = pd.DataFrame({
        "invoice_no": ["536365"],
        "product_id": ["85123A"],
        "product_name": ["Product A"],
        "qty": [6],
        "transaction_date": ["2023-12-01"],
        "price": [2.55],
        "customer_id": ["17850"],
    })
    res = inspect_uploaded_df(df_synonyms)
    assert res["is_valid"] is True
    assert len(res["missing_mandatory"]) == 0


def test_inspect_uploaded_df_missing_mandatory():
    df_invalid = pd.DataFrame({
        "InvoiceNo": ["536365"],
        "Description": ["Product A"],
        "Quantity": [6]
    })
    res = inspect_uploaded_df(df_invalid)
    assert res["is_valid"] is False
    assert "UnitPrice" in res["missing_mandatory"]
    assert "CustomerID" in res["missing_mandatory"]
    assert len(res["errors"]) > 0

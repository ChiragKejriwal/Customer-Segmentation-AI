"""
Data Loader Module for Customer Segmentation Platform.
Handles data ingestion, schema validation, column synonym resolution, and cached loading.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import logging
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MANDATORY_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
]

OPTIONAL_COLUMNS = [
    "Country"
]

COLUMN_SYNONYMS = {
    "InvoiceNo": ["invoiceno", "invoice_no", "invoicenumber", "invoice_number", "order_id", "orderid", "transaction_id"],
    "StockCode": ["stockcode", "stock_code", "product_id", "productid", "item_code", "item_id", "sku"],
    "Description": ["description", "product_name", "item_description", "item_name", "product_description", "title"],
    "Quantity": ["quantity", "qty", "units", "units_sold", "volume", "item_count"],
    "InvoiceDate": ["invoicedate", "invoice_date", "date", "transaction_date", "timestamp", "order_date"],
    "UnitPrice": ["unitprice", "unit_price", "price", "current_price", "item_price", "rate", "cost_per_unit"],
    "CustomerID": ["customerid", "customer_id", "client_id", "user_id", "account_id", "buyer_id"],
    "Country": ["country", "region", "location", "nation", "market", "state"],
}


def get_sample_template_df() -> pd.DataFrame:
    """Returns sample retail transactional records matching the required schema."""
    return pd.DataFrame([
        {
            "InvoiceNo": "536365",
            "StockCode": "85123A",
            "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
            "Quantity": 6,
            "InvoiceDate": "2023-12-01 08:26:00",
            "UnitPrice": 2.55,
            "CustomerID": "17850",
            "Country": "United Kingdom"
        },
        {
            "InvoiceNo": "536365",
            "StockCode": "71053",
            "Description": "WHITE METAL LANTERN",
            "Quantity": 6,
            "InvoiceDate": "2023-12-01 08:26:00",
            "UnitPrice": 3.39,
            "CustomerID": "17850",
            "Country": "United Kingdom"
        },
        {
            "InvoiceNo": "536366",
            "StockCode": "22633",
            "Description": "HAND WARMER UNION JACK",
            "Quantity": 12,
            "InvoiceDate": "2023-12-01 08:28:00",
            "UnitPrice": 1.85,
            "CustomerID": "13047",
            "Country": "United Kingdom"
        },
        {
            "InvoiceNo": "536367",
            "StockCode": "84879",
            "Description": "ASSORTED COLOUR BIRD ORNAMENT",
            "Quantity": 32,
            "InvoiceDate": "2023-12-01 08:34:00",
            "UnitPrice": 1.69,
            "CustomerID": "13047",
            "Country": "United Kingdom"
        },
        {
            "InvoiceNo": "536368",
            "StockCode": "22960",
            "Description": "JAM MAKING SET WITH JARS",
            "Quantity": 8,
            "InvoiceDate": "2023-12-01 08:45:00",
            "UnitPrice": 4.25,
            "CustomerID": "12583",
            "Country": "France"
        },
    ])


def inspect_uploaded_df(df_raw: pd.DataFrame) -> dict:
    """
    Validates uploaded DataFrame schema and maps column synonyms to canonical names.
    """
    raw_cols = list(df_raw.columns)
    col_lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df_raw.columns}

    def _match_target_cols(targets):
        matched, missing = {}, []
        for col in targets:
            syns = [col.lower(), col.lower().replace(" ", "_")] + COLUMN_SYNONYMS.get(col, [])
            found = next((col_lookup[s] for s in syns if s in col_lookup), None)
            if found is not None:
                matched[col] = found
            else:
                missing.append(col)
        return matched, missing

    matched_mandatory, missing_mandatory = _match_target_cols(MANDATORY_COLUMNS)
    matched_optional, missing_optional = _match_target_cols(OPTIONAL_COLUMNS)

    accounted_cols = set(matched_mandatory.values()).union(set(matched_optional.values()))
    extra_columns = [c for c in raw_cols if c not in accounted_cols]

    is_valid = len(missing_mandatory) == 0
    errors = [f"Missing mandatory fields: {missing_mandatory}"] if not is_valid else []

    return {
        "raw_columns": raw_cols,
        "matched_mandatory": matched_mandatory,
        "missing_mandatory": missing_mandatory,
        "matched_optional": matched_optional,
        "missing_optional": missing_optional,
        "extra_columns": extra_columns,
        "is_valid": is_valid,
        "errors": errors,
    }


def get_default_data_path() -> Optional[Path]:
    """Resolves default customer dataset path across different runtime environments."""
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "customer_data.csv",
        Path("data/customer_data.csv").resolve(),
        Path("Customer Segmentation/data/customer_data.csv").resolve(),
        Path(__file__).resolve().parent.parent / "data" / "Online Retail.xlsx",
        Path("data/Online Retail.xlsx").resolve(),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_cached_raw_data(filepath_str: str) -> pd.DataFrame:
    """Cached loader for raw dataset to optimize startup latency."""
    return load_data(filepath_str)


def load_data(filepath: Union[str, Path, object]) -> pd.DataFrame:
    """
    Loads raw transaction logs from a CSV, Excel file, or file-like buffer into a DataFrame.
    """
    try:
        if hasattr(filepath, "read") or hasattr(filepath, "seek"):
            df = pd.read_csv(filepath)
        else:
            path = Path(filepath)
            if not path.is_file():
                error_msg = f"Dataset file not found: {path.resolve()}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            if path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
            else:
                try:
                    df = pd.read_csv(path, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(path, encoding="ISO-8859-1")

        if df.empty:
            raise ValueError("The provided dataset is empty.")

        inspection = inspect_uploaded_df(df)
        rename_map = {orig: target for target, orig in inspection["matched_mandatory"].items() if orig != target}
        if rename_map:
            df = df.rename(columns=rename_map)

        if "InvoiceDate" in df.columns:
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

        return df

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise ValueError(f"Error reading file: {str(e)}") from e


def initialize_app_data(force_reload: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Initializes session state with default cleaned and scored RFM data.
    """
    from src.preprocessing import clean_data
    from src.rfm import calculate_rfm, score_rfm

    if not force_reload and "clean_df" in st.session_state and "rfm_df" in st.session_state:
        return st.session_state["clean_df"], st.session_state["rfm_df"]

    default_path = get_default_data_path()
    if default_path and default_path.is_file():
        try:
            raw_df = load_cached_raw_data(str(default_path.resolve()))
            clean_df = clean_data(raw_df)
            rfm_df = score_rfm(calculate_rfm(clean_df))

            st.session_state["raw_df"] = raw_df
            st.session_state["clean_df"] = clean_df
            st.session_state["rfm_df"] = rfm_df
            st.session_state["data_source_name"] = "UCI Online Retail Catalog"
            return clean_df, rfm_df
        except Exception as e:
            logger.error(f"Failed to initialize default dataset: {e}")

    return None, None

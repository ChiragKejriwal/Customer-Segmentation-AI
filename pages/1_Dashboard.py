"""
Page 1: Executive Dashboard & Data Ingestion
"""

from pathlib import Path
import streamlit as st
import pandas as pd
from src.data_loader import initialize_app_data, inspect_uploaded_df, load_data
from src.preprocessing import clean_data
from src.rfm import calculate_rfm, score_rfm
from src.ui_components import render_sidebar_header, show_schema_modal, ensure_page_data_loaded

st.set_page_config(
    page_title="Dashboard | Customer Segmentation Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure data is properly loaded before rendering
clean_df, rfm_df = ensure_page_data_loaded(page_title="Dashboard")

# Render Modern Sidebar Header & Telemetry
render_sidebar_header(active_page_title="Dashboard")

st.title("📊 Executive Dashboard & Data Ingestion")
st.markdown("Inspect preloaded transactional records, monitor top-level revenue KPIs, or upload custom retail datasets.")

# Sidebar Upload Section
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data Source Configuration")

if st.sidebar.button("📥 Download Sample CSV Template", use_container_width=True):
    show_schema_modal()

uploaded_file = st.sidebar.file_uploader(
    "Upload Transactional CSV",
    type=["csv"],
    help="Upload your transactional retail records matching the required schema."
)

prev_file = st.session_state.get("_last_uploaded_filename", None)
curr_file = uploaded_file.name if uploaded_file is not None else None

if uploaded_file is not None:
    if prev_file != curr_file:
        st.session_state["_last_uploaded_filename"] = curr_file
        try:
            with st.spinner(f"Ingesting and validating `{uploaded_file.name}`..."):
                raw_df = pd.read_csv(uploaded_file)
                inspection = inspect_uploaded_df(raw_df)

                if not inspection["is_valid"]:
                    st.session_state["upload_error_info"] = {
                        "msg": f"Missing mandatory columns: {inspection['missing_mandatory']}",
                        "inspection": inspection,
                        "name": uploaded_file.name
                    }
                    show_schema_modal(inspection=inspection, error_msg=st.session_state["upload_error_info"]["msg"])
                else:
                    st.session_state.pop("upload_error_info", None)
                    clean_df = clean_data(raw_df)
                    rfm_df = score_rfm(calculate_rfm(clean_df))

                    st.session_state["raw_df"] = raw_df
                    st.session_state["clean_df"] = clean_df
                    st.session_state["rfm_df"] = rfm_df
                    st.session_state["data_source_name"] = f"Custom: {uploaded_file.name}"
                    st.sidebar.success(f"Loaded: `{uploaded_file.name}` ({len(clean_df):,} records)")
                    st.rerun()
        except Exception as e:
            st.session_state["upload_error_info"] = {
                "msg": str(e),
                "inspection": None,
                "name": uploaded_file.name
            }
            st.sidebar.error(f"Error parsing file: {e}")
    else:
        if "upload_error_info" in st.session_state:
            err_info = st.session_state["upload_error_info"]
            st.sidebar.error("❌ Schema Validation Alert")
            st.sidebar.caption(f"Mandatory fields missing in `{err_info['name']}`.")
            if st.sidebar.button("🔍 Open Schema & Fix Guide", type="primary", use_container_width=True):
                show_schema_modal(inspection=err_info.get("inspection"), error_msg=err_info.get("msg"))
        else:
            st.sidebar.success(f"Active: `{uploaded_file.name}`")
elif "clean_df" not in st.session_state:
    initialize_app_data(force_reload=True)

# Main Dashboard Content
if "clean_df" in st.session_state and not st.session_state["clean_df"].empty:
    df = st.session_state["clean_df"]
    raw_total = len(st.session_state.get("raw_df", df))

    total_customers = df["CustomerID"].nunique()
    total_revenue = df["TotalAmount"].sum()
    total_transactions = df["InvoiceNo"].nunique()
    total_records = len(df)
    avg_order_val = df["TotalAmount"].mean()

    # Top-Level Business Metric Cards
    st.subheader("📈 Top-Level Business Metrics")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">👥 Unique Customers</div>
            <div class="kpi-val">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💰 Total Revenue</div>
            <div class="kpi-val">${total_revenue:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🧾 Total Invoices</div>
            <div class="kpi-val">{total_transactions:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📦 Cleaned Records</div>
            <div class="kpi-val">{total_records:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔍 Cleaned Transaction Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

    with st.expander("ℹ️ Data Cleaning & Pipeline Validation Summary", expanded=False):
        dropped = raw_total - total_records
        st.write(f"- **Raw Ingested Records:** `{raw_total:,}`")
        st.write(f"- **Clean Valid Records:** `{total_records:,}` ({dropped:,} cancellations / null customer records dropped)")
        st.write(f"- **Average Line Item Amount:** `${avg_order_val:.2f}`")
        if "InvoiceDate" in df.columns and pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
            st.write(f"- **Transaction Date Span:** `{df['InvoiceDate'].min().strftime('%Y-%m-%d')}` to `{df['InvoiceDate'].max().strftime('%Y-%m-%d')}`")
else:
    st.info("💡 Please upload a transaction CSV file from the sidebar or reload the default dataset.")

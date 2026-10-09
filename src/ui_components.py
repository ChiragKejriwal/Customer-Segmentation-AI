"""
Shared UI Components & Styling Module for Customer Segmentation Platform.
Provides rich sidebar branding, dataset telemetry cards, custom CSS, and CSV schema modals.
"""

from typing import Any, Dict, Optional
import streamlit as st
import pandas as pd
from src.data_loader import get_sample_template_df, MANDATORY_COLUMNS


def apply_custom_css():
    """Injects high-contrast, modern glassmorphism CSS across the application."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0B0F19 0%, #0F172A 50%, #0B0F19 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        .sidebar-brand-box {
            background: linear-gradient(135deg, rgba(30, 27, 75, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
            padding: 1.15rem 1.25rem;
            border-radius: 14px;
            border: 1px solid rgba(99, 102, 241, 0.35);
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.2);
        }

        .sidebar-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34D399;
            margin-bottom: 0.6rem;
        }

        .sidebar-metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
        }

        .sidebar-mini-metric {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 0.5rem 0.65rem;
            text-align: center;
        }

        .sidebar-mini-metric-label {
            font-size: 0.68rem;
            color: #94A3B8;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .sidebar-mini-metric-val {
            font-size: 0.95rem;
            color: #F8FAFC;
            font-weight: 700;
            margin-top: 2px;
        }

        /* Glass Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }

        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.18);
            transform: translateY(-2px);
        }

        .kpi-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
            transition: border-color 0.2s ease;
        }

        .kpi-card:hover {
            border-color: rgba(56, 189, 248, 0.4);
        }

        .kpi-title {
            color: #94A3B8;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }

        .kpi-val {
            color: #F8FAFC;
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        /* Chips & Badges */
        .card-badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .badge-green { background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }
        .badge-blue { background: rgba(56, 189, 248, 0.18); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); }
        .badge-indigo { background: rgba(99, 102, 241, 0.18); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.4); }
        .badge-rose { background: rgba(244, 63, 94, 0.18); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.4); }
        .badge-amber { background: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }

        /* Performance Metric Glass Cards */
        .perf-metric-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.25rem 1.25rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
            transition: all 0.25s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .perf-metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 14px 30px -5px rgba(0, 0, 0, 0.5);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .perf-card-cyan { border-top: 3px solid #38BDF8; }
        .perf-card-indigo { border-top: 3px solid #818CF8; }
        .perf-card-emerald { border-top: 3px solid #34D399; }
        .perf-card-amber { border-top: 3px solid #F59E0B; }

        .perf-metric-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.75rem;
        }

        .perf-metric-title {
            color: #94A3B8;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .perf-metric-val {
            font-size: 1.65rem;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            line-height: 1.2;
            margin-bottom: 0.4rem;
        }

        .perf-metric-sub {
            color: #64748B;
            font-size: 0.75rem;
            font-weight: 500;
            line-height: 1.3;
        }

        /* Animated Loader Styles */
        @keyframes spinRing {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes pulseGlow {
            0%, 100% { opacity: 0.85; box-shadow: 0 10px 30px rgba(99, 102, 241, 0.35); }
            50% { opacity: 1; box-shadow: 0 15px 50px rgba(56, 189, 248, 0.55); }
        }

        .loader-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.9) 50%, rgba(15, 23, 42, 0.98) 100%);
            border: 1px solid rgba(99, 102, 241, 0.45);
            border-radius: 20px;
            padding: 3rem 2.5rem;
            text-align: center;
            max-width: 680px;
            margin: 3.5rem auto;
            animation: pulseGlow 3s infinite ease-in-out;
        }

        .loader-spinner-ring {
            display: inline-block;
            width: 56px;
            height: 56px;
            border: 4px solid rgba(99, 102, 241, 0.2);
            border-top: 4px solid #38BDF8;
            border-right: 4px solid #818CF8;
            border-radius: 50%;
            animation: spinRing 0.9s linear infinite;
            margin-bottom: 1.25rem;
        }

        .loader-step-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 0.4rem 1rem;
            border-radius: 9999px;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.35);
            color: #C7D2FE;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


def ensure_page_data_loaded(page_title: str = "") -> tuple:
    """
    Ensures that cleaned and RFM datasets are completely and properly loaded before rendering the page.
    Displays an elegant animated glassmorphic loader with progress feedback if loading from disk.

    Returns:
        tuple: (clean_df, rfm_df) ready for computation and visualization.
    """
    apply_custom_css()

    # If data is already fully populated in session state, return immediately with zero delay
    if (
        "clean_df" in st.session_state 
        and "rfm_df" in st.session_state 
        and st.session_state["clean_df"] is not None
        and not st.session_state["clean_df"].empty
    ):
        return st.session_state["clean_df"], st.session_state["rfm_df"]

    # Render dedicated full-page loading card
    loader_placeholder = st.empty()
    progress_placeholder = st.empty()

    with loader_placeholder.container():
        st.markdown(f"""
        <div class="loader-card">
            <div class="loader-spinner-ring"></div>
            <div style="font-size: 1.45rem; font-weight: 800; background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 50%, #818CF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">
                🔄 Initializing Customer Intelligence Engine
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8; line-height: 1.5; max-width: 520px; margin: 0 auto;">
                Loading and validating transactional retail records, cleaning anomalies, and scoring behavioral RFM matrices...
            </div>
            <div class="loader-step-pill">
                <span>⚡</span> Preparing {page_title if page_title else 'Workspace'} View...
            </div>
        </div>
        """, unsafe_allow_html=True)

    from src.data_loader import initialize_app_data

    clean_df, rfm_df = initialize_app_data(force_reload=True)

    # Clear loading placeholders once data is fully ready
    loader_placeholder.empty()
    progress_placeholder.empty()

    if clean_df is None or clean_df.empty:
        st.error("❌ **Data Loading Alert:** Could not locate or initialize default retail dataset. Please upload a dataset in Dashboard.")
        st.stop()

    return clean_df, rfm_df


def render_sidebar_header(active_page_title: str = "Overview"):
    """Renders the styled header brand card and active telemetry indicators in the sidebar."""
    apply_custom_css()

    st.sidebar.markdown("""
    <div class="sidebar-brand-box">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="font-size: 1.6rem; line-height: 1;">🎯</div>
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 50%, #818CF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.01em;">
                    SegmentAI Hub
                </div>
                <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 500;">
                    Behavioral RFM & Unsupervised ML
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active dataset status card
    data_source_name = st.session_state.get("data_source_name", "UCI Online Retail Catalog")
    total_customers = len(st.session_state.get("rfm_df", [])) if "rfm_df" in st.session_state else 0
    total_records = len(st.session_state.get("clean_df", [])) if "clean_df" in st.session_state else 0

    st.sidebar.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 0.85rem 1rem; margin-bottom: 1rem;">
        <div class="sidebar-status-pill">
            <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #34D399; box-shadow: 0 0 8px #34D399;"></span>
            {data_source_name}
        </div>
        <div class="sidebar-metric-grid">
            <div class="sidebar-mini-metric">
                <div class="sidebar-mini-metric-label">👥 Customers</div>
                <div class="sidebar-mini-metric-val">{total_customers:,}</div>
            </div>
            <div class="sidebar-mini-metric">
                <div class="sidebar-mini-metric-label">📦 Clean Logs</div>
                <div class="sidebar-mini-metric-val">{total_records:,}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


@st.dialog("📋 Required CSV Schema & Template Guide", width="large")
def show_schema_modal(inspection: Optional[Dict[str, Any]] = None, error_msg: Optional[str] = None):
    """
    Renders an interactive modal explaining the mandatory transactional CSV schema,
    showing validation errors/warnings, a sample preview, and a download template button.
    """
    if error_msg or (inspection and not inspection.get("is_valid", True)):
        st.markdown(
            """
            <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 1.15rem 1.35rem; margin-bottom: 1.25rem;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #F87171; margin-bottom: 0.35rem;">
                    ❌ Schema Validation Issue Detected
                </div>
                <div style="font-size: 0.92rem; color: #E2E8F0; line-height: 1.5;">
                    The uploaded CSV cannot be processed because one or more mandatory transactional fields are missing or improperly formatted.
                    Please match the required column schema below or download the sample template.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if inspection and inspection.get("missing_mandatory"):
            missing_html = " ".join([f"<span class='card-badge badge-rose'>Missing: {c}</span>" for c in inspection["missing_mandatory"]])
            st.markdown(f"**Missing Mandatory Columns:** {missing_html}", unsafe_allow_html=True)

        if inspection and inspection.get("raw_columns"):
            detected_html = " ".join([f"<span class='card-badge badge-blue'>{c}</span>" for c in inspection["raw_columns"]])
            st.markdown(f"**Detected Columns in Upload:** {detected_html}", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(56, 189, 248, 0.1) 100%); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 12px; padding: 1.15rem 1.35rem; margin-bottom: 1.25rem;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #818CF8; margin-bottom: 0.35rem;">
                    📥 Pre-Formatted Transactional Dataset Template
                </div>
                <div style="font-size: 0.92rem; color: #E2E8F0; line-height: 1.5;">
                    Upload standard e-commerce transactional logs. The pipeline requires: 
                    <code>InvoiceNo</code>, <code>StockCode</code>, <code>Description</code>, <code>Quantity</code>, <code>InvoiceDate</code>, <code>UnitPrice</code>, and <code>CustomerID</code>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 📋 Required Schema Columns")
    col_desc_df = pd.DataFrame([
        {"Column": "InvoiceNo", "Type": "String / Integer", "Description": "Unique invoice identifier. 'C' prefix indicates cancellations.", "Required": "Yes"},
        {"Column": "StockCode", "Type": "String", "Description": "Product / Item SKU identifier code.", "Required": "Yes"},
        {"Column": "Description", "Type": "String", "Description": "Product title or item description.", "Required": "Yes"},
        {"Column": "Quantity", "Type": "Integer (> 0)", "Description": "Number of units purchased per line item.", "Required": "Yes"},
        {"Column": "InvoiceDate", "Type": "Datetime (YYYY-MM-DD)", "Description": "Date and timestamp when the transaction occurred.", "Required": "Yes"},
        {"Column": "UnitPrice", "Type": "Float (> 0)", "Description": "Unit price in dollars or currency per item.", "Required": "Yes"},
        {"Column": "CustomerID", "Type": "String / Integer", "Description": "Unique customer account identifier for RFM tracking.", "Required": "Yes"},
        {"Column": "Country", "Type": "String", "Description": "Customer geographic country location.", "Required": "Optional"},
    ])
    st.dataframe(col_desc_df, use_container_width=True, hide_index=True)

    st.markdown("#### 📄 Sample Template Preview")
    sample_df = get_sample_template_df()
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

    csv_buffer = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Sample CSV Template (.csv)",
        data=csv_buffer,
        file_name="sample_retail_segmentation_template.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )


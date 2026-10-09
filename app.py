"""
Customer Segmentation Platform - Main Application Landing Page.
Streamlit multi-page application for unsupervised customer intelligence and behavioral persona generation.
"""

import streamlit as st
from src.ui_components import render_sidebar_header, show_schema_modal, ensure_page_data_loaded

st.set_page_config(
    page_title="Customer Segmentation Platform | AI Behavioral Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure data is completely loaded before rendering page components
clean_df, rfm_df = ensure_page_data_loaded(page_title="Platform Overview")

# Render Modern Sidebar Header & Telemetry
render_sidebar_header(active_page_title="Platform Overview")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Data Schema & Templates")
if st.sidebar.button("📥 Download Sample CSV Template", use_container_width=True):
    show_schema_modal()

# Hero Section
st.markdown("""
<div class="main-hero">
    <h1 class="hero-title">🎯 Customer Segmentation Platform</h1>
    <p class="hero-subtitle">
        An enterprise-grade customer intelligence system that transforms raw e-commerce transaction logs 
        into mathematical behavioral clusters and actionable marketing personas using unsupervised machine learning.
    </p>
</div>
""", unsafe_allow_html=True)

# Status Pill
if "clean_df" in st.session_state and "rfm_df" in st.session_state:
    num_cust = len(st.session_state["rfm_df"])
    num_trans = len(st.session_state["clean_df"])
    source_name = st.session_state.get("data_source_name", "UCI Online Retail")
    st.success(f"⚡ **Pipeline Ready ({source_name}):** Preloaded `{num_cust:,}` customer accounts across `{num_trans:,}` clean transactions ready for analysis.")

st.subheader("⚡ Core Architecture & Workflow")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="glass-card">
        <h4>📊 1. Data Ingestion & Quality Control</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Ingests UCI Online Retail logs, drops anomalies and cancellations, enforces strict typing, 
            and computes derived line-item revenue.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div class="glass-card">
        <h4>📐 2. RFM Behavioral Modeling</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Aggregates customer transactions into <b>Recency</b> (days since purchase), 
            <b>Frequency</b> (total unique orders), and <b>Monetary</b> (lifetime spend) metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <h4>🔬 3. Unsupervised Clustering Engine</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Applies log-transformation (`np.log1p`), 99.5th percentile outlier clipping, and StandardScaler 
            before K-Means/DBSCAN clustering with 2D/3D PCA projection.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div class="glass-card">
        <h4>👥 4. Targeted Marketing Personas</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Dynamically maps mathematical clusters into unique, non-overlapping marketing personas 
            with tailored retention playbooks, interactive Treemaps, and CSV export.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("🧭 Platform Navigation")
st.info("👈 **Select any page in the left sidebar** to explore the data pipeline, train clustering algorithms, or inspect marketing personas.")


"""
Customer Segmentation Platform - Main Application Entry
"""

import streamlit as st
from src.ui_components import render_sidebar_header, ensure_page_data_loaded

st.set_page_config(
    page_title="Customer Segmentation Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure data is ready before rendering
clean_df, rfm_df = ensure_page_data_loaded(page_title="Platform Overview")
render_sidebar_header(active_page_title="Platform Overview")

# Hero Section
st.markdown("""
<div class="main-hero">
    <h1 class="hero-title">🎯 Customer Segmentation Platform</h1>
    <p class="hero-subtitle">
        An enterprise-grade customer intelligence system that transforms raw transactional logs 
        into mathematical behavioral clusters and actionable marketing personas using unsupervised machine learning.
    </p>
</div>
""", unsafe_allow_html=True)

if "clean_df" in st.session_state and "rfm_df" in st.session_state:
    num_cust = len(st.session_state["rfm_df"])
    num_trans = len(st.session_state["clean_df"])
    source_name = st.session_state.get("data_source_name", "UCI Online Retail Catalog")
    st.success(f"⚡ **Pipeline Ready ({source_name}):** `{num_cust:,}` customer accounts across `{num_trans:,}` clean transactions.")

st.subheader("⚡ Core Architecture & Workflow")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="glass-card">
        <h4>📊 1. Data Ingestion & Quality Control</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Ingests transactional logs, drops anomalies and cancellations, enforces strict typing, 
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
            <b>Frequency</b> (unique orders), and <b>Monetary</b> (lifetime spend) metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <h4>🔬 3. Unsupervised Clustering Engine</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Applies log1p transformation, 99.5th percentile outlier clipping, and StandardScaler 
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

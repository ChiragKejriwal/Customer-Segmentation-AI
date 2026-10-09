"""
Page 2: RFM Analysis
Calculates customer-level Recency, Frequency, and Monetary metrics and displays behavioral distributions.
"""

import streamlit as st
import pandas as pd
from src.data_loader import initialize_app_data
from src.rfm import calculate_rfm, score_rfm
from src.visualization import plot_rfm_distributions
from src.ui_components import render_sidebar_header, ensure_page_data_loaded

st.set_page_config(
    page_title="RFM Analysis | Customer Segmentation",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure data is properly loaded before rendering
clean_df, rfm_df = ensure_page_data_loaded(page_title="RFM Analysis")

# Render Modern Sidebar Header & Telemetry
render_sidebar_header(active_page_title="RFM Analysis")

st.title("📐 Recency, Frequency & Monetary (RFM) Analysis")
st.markdown("""
RFM is a behavioral segmentation framework modeling customer purchasing velocity:
- **Recency ($R$):** Days since the customer's last purchase (lower = more active).
- **Frequency ($F$):** Total unique orders placed (higher = more loyal).
- **Monetary ($M$):** Gross lifetime customer revenue (higher = more valuable).
""")

scored_rfm = rfm_df

# Top-level RFM Summary Cards
st.subheader("📊 Customer Benchmark Metrics")
c1, c2, c3, c4 = st.columns(4)

total_cust = len(scored_rfm)
avg_r = scored_rfm["Recency"].mean()
avg_f = scored_rfm["Frequency"].mean()
avg_m = scored_rfm["Monetary"].mean()

with c1:
    st.markdown(f"""
    <div class="rfm-card">
        <div class="rfm-title">👥 Analyzed Customers</div>
        <div class="rfm-val">{total_cust:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="rfm-card">
        <div class="rfm-title">⏱️ Average Recency</div>
        <div class="rfm-val">{avg_r:.1f} days</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="rfm-card">
        <div class="rfm-title">🔄 Average Frequency</div>
        <div class="rfm-val">{avg_f:.1f} orders</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="rfm-card">
        <div class="rfm-title">💳 Average Spend (LTV)</div>
        <div class="rfm-val">${avg_m:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("📈 RFM Metric Distributions")
fig_dist = plot_rfm_distributions(scored_rfm)
st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")
st.subheader("📋 Customer RFM Table Preview")
st.dataframe(scored_rfm.head(100), use_container_width=True)

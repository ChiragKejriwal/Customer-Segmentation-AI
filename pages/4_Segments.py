"""
Page 4: Personas and Targeted Marketing Strategies
"""

import streamlit as st
import pandas as pd
from src.data_loader import initialize_app_data
from src.personas import generate_personas, assign_personas_by_rfm, PERSONA_MAP
from src.visualization import plot_persona_treemap, plot_persona_pie, plot_persona_revenue_comparison
from src.ui_components import render_sidebar_header, ensure_page_data_loaded

st.set_page_config(
    page_title="Segments & Personas | Customer Segmentation Platform",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure data is properly loaded before rendering
clean_df, rfm_df = ensure_page_data_loaded(page_title="Customer Personas")

# Render Modern Sidebar Header & Telemetry
render_sidebar_header(active_page_title="Personas")

st.title("👥 Customer Personas & Marketing Playbooks")
st.markdown("""
Translate algorithmic clusters into distinct, non-overlapping business personas. 
Inspect revenue contributions via interactive Treemaps, review prescriptive marketing strategies, and export tagged customer lists.
""")

# Use ML cluster labels if available, otherwise fallback to rule-based personas
if "cluster_labels" in st.session_state:
    cluster_labels = st.session_state["cluster_labels"]
    persona_df = generate_personas(rfm_df, cluster_labels)
elif "clustered_df" in st.session_state and "Cluster" in st.session_state["clustered_df"].columns:
    persona_df = generate_personas(st.session_state["clustered_df"], st.session_state["clustered_df"]["Cluster"])
else:
    st.info("💡 ML Clustering has not been run yet. Using quantile-based RFM persona rules. Visit the **Clustering** page for unsupervised ML models.")
    persona_df = assign_personas_by_rfm(rfm_df)

st.session_state["persona_df"] = persona_df

# Persona Summary Aggregation
summary = persona_df.groupby("Persona").agg(
    Count=("CustomerID", "count"),
    Avg_Monetary=("Monetary", "mean"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean")
).reset_index()
summary["% Share"] = ((summary["Count"] / len(persona_df)) * 100).round(1).astype(str) + "%"

# 1. Visualizations Section
st.subheader("📊 Customer Persona Composition & Segment Economics")

col1, col2 = st.columns([1.15, 0.85])

with col1:
    st.plotly_chart(plot_persona_treemap(summary), use_container_width=True)

with col2:
    st.plotly_chart(plot_persona_pie(summary), use_container_width=True)

with st.expander("📊 View Total Revenue Breakdown by Persona Segment", expanded=False):
    st.plotly_chart(plot_persona_revenue_comparison(summary), use_container_width=True)

st.markdown("---")

# 2. Actionable Marketing Recommendations Playbook
st.subheader("🎯 Actionable Marketing Playbooks by Persona")

for persona_name, details in PERSONA_MAP.items():
    if persona_name in summary["Persona"].values:
        matching_stats = summary[summary["Persona"] == persona_name].iloc[0]
        with st.expander(f"📌 **{persona_name}** ({matching_stats['Count']:,} Customers — {matching_stats['% Share']})", expanded=True):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(f"""
                - **Avg Recency:** `{matching_stats['Avg_Recency']:.1f} days`
                - **Avg Frequency:** `{matching_stats['Avg_Frequency']:.1f} orders`
                - **Avg Spend (LTV):** `${matching_stats['Avg_Monetary']:,.2f}`
                """)
            with col_b:
                st.markdown(f"**Customer Behavioral Profile:**\n{details['description']}")
                st.info(f"💡 **Prescriptive Strategy:** {details['action']}")

st.markdown("---")

# 3. Export CSV with Download Button
st.subheader("📥 Export Tagged Customer List")
st.markdown("Download customer records mapped with their behavioral segment persona tags for CRM integration (e.g., Klaviyo, HubSpot, Salesforce).")

export_cols = [c for c in ["CustomerID", "Recency", "Frequency", "Monetary", "Cluster", "Persona", "RFM_Segment"] if c in persona_df.columns]
export_df = persona_df[export_cols]

csv_data = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Customer Personas CSV",
    data=csv_data,
    file_name="customer_personas_export.csv",
    mime="text/csv",
    type="primary"
)

st.markdown("---")

# 4. Customer Lookup
st.subheader("🔍 Single Customer Lookup")
search_id = st.text_input("Enter CustomerID (e.g., 17850):")
if search_id:
    match = persona_df[persona_df["CustomerID"].astype(str) == search_id.strip()]
    if not match.empty:
        st.success(f"Customer `{search_id.strip()}` found!")
        st.dataframe(match[export_cols], use_container_width=True)
    else:
        st.warning(f"No records found for CustomerID `{search_id.strip()}`.")

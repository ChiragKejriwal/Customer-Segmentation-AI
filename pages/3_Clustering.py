"""
Page 3: Machine Learning Clustering & 2D/3D PCA Projection
"""

import streamlit as st
import pandas as pd
from src.data_loader import initialize_app_data
from src.clustering import (
    preprocess_for_clustering,
    apply_kmeans,
    apply_dbscan,
    compute_pca_projections
)
from src.visualization import plot_2d_clusters, plot_3d_clusters
from src.ui_components import render_sidebar_header, ensure_page_data_loaded

st.set_page_config(
    page_title="Clustering | Customer Segmentation Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure data is properly loaded before rendering
clean_df, rfm_df = ensure_page_data_loaded(page_title="ML Clustering")

# Render Modern Sidebar Header & Telemetry
render_sidebar_header(active_page_title="Clustering")

st.title("🔬 Unsupervised ML Customer Clustering")
st.markdown("""
Train unsupervised clustering models on log-transformed, standard-scaled RFM features. 
Evaluate cluster compactness via the **Silhouette Score** and inspect latent customer relationships in **2D or 3D PCA space**.
""")

st.sidebar.header("⚙️ Clustering Configuration")
algo = st.sidebar.selectbox("Clustering Algorithm", ["K-Means", "DBSCAN"], index=0)

@st.cache_data(show_spinner=False)
def _get_scaled_data(df: pd.DataFrame):
    return preprocess_for_clustering(df)

@st.cache_data(show_spinner=False)
def _get_pca(scaled):
    return compute_pca_projections(scaled)

@st.cache_data(show_spinner=False)
def _get_kmeans(scaled, n_clusters: int):
    return apply_kmeans(scaled, n_clusters=n_clusters)

@st.cache_data(show_spinner=False)
def _get_dbscan(scaled, eps: float, min_samples: int):
    return apply_dbscan(scaled, eps=eps, min_samples=min_samples)

# Preprocess features: log1p + 99.5th percentile capping + StandardScaler
scaled_data, scaler = _get_scaled_data(rfm_df)

if algo == "K-Means":
    k = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=8, value=4, step=1)
    labels, sil_score = _get_kmeans(scaled_data, n_clusters=k)
else:
    eps = st.sidebar.slider("DBSCAN Epsilon (eps)", min_value=0.1, max_value=3.0, value=0.6, step=0.1)
    min_samples = st.sidebar.slider("Min Samples", min_value=2, max_value=20, value=5, step=1)
    labels, sil_score = _get_dbscan(scaled_data, eps=eps, min_samples=min_samples)

# Sidebar 2D vs 3D Projection Toggle
st.sidebar.markdown("---")
st.sidebar.header("🌐 Projection View Mode")
view_mode = st.sidebar.radio(
    "Select Dimensionality Mode:",
    ["2D PCA Scatter", "3D PCA Space"],
    index=0
)

# Compute PCA Projections (2D and 3D)
pca_dict = _get_pca(scaled_data)
pca_2d = pca_dict["pca_2d"]
pca_3d = pca_dict["pca_3d"]
var_2d = pca_dict["var_ratio_2d"]
var_3d = pca_dict["var_ratio_3d"]

clustered_df = rfm_df.copy()
clustered_df["Cluster"] = labels.astype(str)
clustered_df["PCA1"] = pca_3d["PCA1"]
clustered_df["PCA2"] = pca_3d["PCA2"]
clustered_df["PCA3"] = pca_3d["PCA3"]

st.session_state["cluster_labels"] = labels
st.session_state["clustered_df"] = clustered_df

# Top Metrics Row
st.markdown("### 🎯 Model Performance & Variance Metrics")
c1, c2, c3, c4 = st.columns(4)

unique_clusters = sorted(list(set(labels)))
num_clusters = len([c for c in unique_clusters if c != -1])
noise_count = int((labels == -1).sum()) if -1 in unique_clusters else 0

# Silhouette score quality assessment
if sil_score >= 0.50:
    sil_badge_class = "badge-green"
    sil_badge_text = "✦ Strong Structure"
elif sil_score >= 0.30:
    sil_badge_class = "badge-blue"
    sil_badge_text = "✦ Good Separation"
elif sil_score >= 0.20:
    sil_badge_class = "badge-amber"
    sil_badge_text = "✦ Moderate Overlap"
else:
    sil_badge_class = "badge-rose"
    sil_badge_text = "✦ Low Distinctness"

with c1:
    st.markdown(f"""
    <div class="perf-metric-card perf-card-cyan">
        <div class="perf-metric-header">
            <span class="perf-metric-title">🎯 Algorithm</span>
            <span class="card-badge badge-blue">Unsupervised</span>
        </div>
        <div class="perf-metric-val">{algo}</div>
        <div class="perf-metric-sub">{'Centroid Partitioning' if algo == 'K-Means' else 'Density-Based Spatial'}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="perf-metric-card perf-card-indigo">
        <div class="perf-metric-header">
            <span class="perf-metric-title">🔢 Formed Clusters</span>
            <span class="card-badge badge-indigo">{num_clusters} Segments</span>
        </div>
        <div class="perf-metric-val">{num_clusters} <span style="font-size: 0.95rem; font-weight: 500; color: #94A3B8;">groups</span></div>
        <div class="perf-metric-sub">{f"{noise_count:,} noise points" if noise_count > 0 else "0 unclassified noise records"}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="perf-metric-card perf-card-emerald">
        <div class="perf-metric-header">
            <span class="perf-metric-title">✨ Silhouette Score</span>
            <span class="card-badge {sil_badge_class}">{sil_badge_text}</span>
        </div>
        <div class="perf-metric-val" style="color: #34D399;">{sil_score:.4f}</div>
        <div class="perf-metric-sub">Cluster cohesion & distance index</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="perf-metric-card perf-card-amber">
        <div class="perf-metric-header">
            <span class="perf-metric-title">📈 PCA Variance</span>
            <span class="card-badge badge-amber">{var_2d * 100:.1f}% in 2D</span>
        </div>
        <div class="perf-metric-val">{var_2d * 100:.1f}% <span style="font-size: 0.95rem; font-weight: 500; color: #94A3B8;">/ {var_3d * 100:.1f}%</span></div>
        <div class="perf-metric-sub">2D / 3D latent variance captured</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Render Selected Visualization
if view_mode == "2D PCA Scatter":
    st.subheader(f"📊 2D PCA Latent Space (Retains {var_2d * 100:.1f}% Total Variance)")
    fig_2d = plot_2d_clusters(pca_2d, labels=labels, customer_ids=clustered_df.get("CustomerID"))
    st.plotly_chart(fig_2d, use_container_width=True)
else:
    st.subheader(f"🌐 3D PCA Interactive Space (Retains {var_3d * 100:.1f}% Total Variance)")
    fig_3d = plot_3d_clusters(pca_3d, labels=labels, customer_ids=clustered_df.get("CustomerID"))
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")
st.subheader("📊 Cluster Statistical Profiles (Means)")
profile = clustered_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
profile["Customer Count"] = clustered_df.groupby("Cluster")["CustomerID"].count()
profile["% of Base"] = ((profile["Customer Count"] / len(clustered_df)) * 100).round(1).astype(str) + "%"
st.dataframe(profile, use_container_width=True)

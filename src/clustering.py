"""
Clustering Module for Customer Segmentation.
Applies log-transformation, outlier percentile clipping, StandardScaler,
K-Means, DBSCAN, and 2D/3D PCA dimensionality reduction.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def preprocess_for_clustering(
    rfm_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    outlier_percentile: float = 99.5
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Preprocesses RFM features for clustering:
    1. Applies np.log1p transformation to reduce skewness.
    2. Clips extreme outliers above specified percentile (default 99.5%).
    3. Normalizes features using StandardScaler.
    """
    if feature_cols is None:
        feature_cols = ["Recency", "Frequency", "Monetary"]

    if rfm_df is None or rfm_df.empty:
        return np.empty((0, len(feature_cols))), StandardScaler()

    features = rfm_df[feature_cols].copy().astype(float)
    non_negative = np.maximum(features.values, 0.0)
    log_transformed = np.log1p(non_negative)

    capped = np.zeros_like(log_transformed)
    for col_idx in range(log_transformed.shape[1]):
        col_vals = log_transformed[:, col_idx]
        cap_val = np.percentile(col_vals, outlier_percentile)
        capped[:, col_idx] = np.clip(col_vals, a_min=None, a_max=cap_val)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(capped)
    return scaled_data, scaler


scale_features = preprocess_for_clustering


def apply_kmeans(
    scaled_data: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Fits K-Means algorithm and calculates silhouette score with adaptive sampling.
    """
    if len(scaled_data) < n_clusters or len(scaled_data) <= 1:
        return np.zeros(len(scaled_data), dtype=int), 0.0

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(scaled_data)

    unique_labels = np.unique(labels)
    if len(unique_labels) > 1 and len(scaled_data) > len(unique_labels):
        sample_size = min(len(scaled_data), 2500) if len(scaled_data) > 2500 else None
        sil_score = float(silhouette_score(scaled_data, labels, sample_size=sample_size, random_state=random_state))
    else:
        sil_score = 0.0

    return labels, sil_score


def run_kmeans(
    scaled_data: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[np.ndarray, KMeans, float]:
    """
    Fits K-Means and returns labels, model object, and silhouette score.
    """
    if len(scaled_data) < n_clusters or len(scaled_data) <= 1:
        kmeans = KMeans(n_clusters=max(1, min(n_clusters, len(scaled_data))), random_state=random_state, n_init=10)
        return np.zeros(len(scaled_data), dtype=int), kmeans, 0.0

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    unique_labels = np.unique(labels)
    sample_size = min(len(scaled_data), 2500) if len(scaled_data) > 2500 else None
    sil_score = float(silhouette_score(scaled_data, labels, sample_size=sample_size, random_state=random_state)) if len(unique_labels) > 1 else 0.0
    return labels, kmeans, sil_score


def apply_dbscan(
    scaled_data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Fits DBSCAN density-based clustering algorithm with adaptive silhouette scoring.
    """
    if len(scaled_data) == 0:
        return np.array([], dtype=int), 0.0

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_data)

    unique_labels = set(labels)
    non_noise = [l for l in unique_labels if l != -1]

    if len(non_noise) > 1 and len(scaled_data) > len(unique_labels):
        try:
            sample_size = min(len(scaled_data), 2500) if len(scaled_data) > 2500 else None
            sil_score = float(silhouette_score(scaled_data, labels, sample_size=sample_size, random_state=random_state))
        except Exception:
            sil_score = 0.0
    else:
        sil_score = 0.0

    return labels, sil_score


def apply_pca(
    scaled_data: np.ndarray,
    n_components: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, PCA]:
    """
    Applies Principal Component Analysis (PCA) to reduce feature dimensionality.
    """
    if len(scaled_data) == 0:
        return np.empty((0, n_components)), PCA(n_components=n_components, random_state=random_state)

    components = min(n_components, scaled_data.shape[0], scaled_data.shape[1])
    pca = PCA(n_components=components, random_state=random_state)
    pca_data = pca.fit_transform(scaled_data)
    return pca_data, pca


run_pca = apply_pca


def compute_pca_projections(
    scaled_data: np.ndarray,
    random_state: int = 42
) -> Dict[str, Union[pd.DataFrame, float, PCA]]:
    """
    Computes 2D and 3D PCA coordinates and explained variance ratios.
    """
    pca_2d_arr, model_2d = apply_pca(scaled_data, n_components=2, random_state=random_state)
    pca_3d_arr, model_3d = apply_pca(scaled_data, n_components=3, random_state=random_state)

    cols_2d = [f"PCA{i+1}" for i in range(pca_2d_arr.shape[1])]
    cols_3d = [f"PCA{i+1}" for i in range(pca_3d_arr.shape[1])]

    var_2d = float(np.sum(model_2d.explained_variance_ratio_)) if hasattr(model_2d, "explained_variance_ratio_") else 0.0
    var_3d = float(np.sum(model_3d.explained_variance_ratio_)) if hasattr(model_3d, "explained_variance_ratio_") else 0.0

    return {
        "pca_2d": pd.DataFrame(pca_2d_arr, columns=cols_2d),
        "pca_3d": pd.DataFrame(pca_3d_arr, columns=cols_3d),
        "var_ratio_2d": var_2d,
        "var_ratio_3d": var_3d,
        "pca_model_2d": model_2d,
        "pca_model_3d": model_3d
    }

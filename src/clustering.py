"""
Clustering Module for Customer Segmentation.
Applies log-transformation, outlier capping, StandardScaler preprocessing,
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
    Preprocesses RFM features for robust clustering and PCA visualization:
    1. Applies log-transformation (np.log1p) to neutralize power-law skewness.
    2. Caps extreme outliers above the specified percentile (default: 99.5th).
    3. Standardizes features with StandardScaler (zero mean, unit variance).

    Args:
        rfm_df: DataFrame containing Recency, Frequency, and Monetary features.
        feature_cols: Target feature column names (defaults to ["Recency", "Frequency", "Monetary"]).
        outlier_percentile: Upper percentile threshold to clip extreme values.

    Returns:
        Tuple[np.ndarray, StandardScaler]: Scaled feature array and fitted StandardScaler instance.
    """
    if feature_cols is None:
        feature_cols = ["Recency", "Frequency", "Monetary"]

    if rfm_df is None or rfm_df.empty:
        scaler = StandardScaler()
        return np.empty((0, len(feature_cols))), scaler

    features = rfm_df[feature_cols].copy().astype(float)

    # 1. Apply log-transformation to handle exponential/skewed retail distributions
    non_negative = np.maximum(features.values, 0.0)
    log_transformed = np.log1p(non_negative)

    # 2. Cap extreme outliers above the 99.5th percentile to prevent PCA squishing
    capped_features = np.zeros_like(log_transformed)
    for col_idx in range(log_transformed.shape[1]):
        col_vals = log_transformed[:, col_idx]
        cap_val = np.percentile(col_vals, outlier_percentile)
        capped_features[:, col_idx] = np.clip(col_vals, a_min=None, a_max=cap_val)

    # 3. Apply StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(capped_features)

    return scaled_data, scaler


# Alias for backward compatibility
scale_features = preprocess_for_clustering


def apply_kmeans(
    scaled_data: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Fits K-Means clustering algorithm on scaled data with fast silhouette scoring.

    Returns:
        Tuple[np.ndarray, float]: Cluster labels and the Silhouette Score.
    """
    if len(scaled_data) < n_clusters or len(scaled_data) <= 1:
        labels = np.zeros(len(scaled_data), dtype=int)
        return labels, 0.0

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
    Fits K-Means and returns labels, fitted model, and silhouette score.
    """
    if len(scaled_data) < n_clusters or len(scaled_data) <= 1:
        kmeans = KMeans(n_clusters=max(1, min(n_clusters, len(scaled_data))), random_state=random_state, n_init=10)
        labels = np.zeros(len(scaled_data), dtype=int)
        return labels, kmeans, 0.0

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
    Fits DBSCAN clustering algorithm on scaled data as a density-based alternative with fast silhouette scoring.
    """
    if len(scaled_data) == 0:
        return np.array([], dtype=int), 0.0

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_data)

    unique_labels = set(labels)
    non_noise_labels = [l for l in unique_labels if l != -1]

    if len(non_noise_labels) > 1 and len(scaled_data) > len(unique_labels):
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
    Performs Principal Component Analysis (PCA) to reduce dimensions.

    Args:
        scaled_data: 2D numpy array of scaled features.
        n_components: Target dimensionality (2 or 3).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, PCA]: Transformed PCA coordinates and fitted PCA model.
    """
    if len(scaled_data) == 0:
        pca = PCA(n_components=n_components, random_state=random_state)
        return np.empty((0, n_components)), pca

    max_components = min(scaled_data.shape[0], scaled_data.shape[1])
    components = min(n_components, max_components)

    pca = PCA(n_components=components, random_state=random_state)
    pca_data = pca.fit_transform(scaled_data)
    return pca_data, pca


def compute_pca_projections(
    scaled_data: np.ndarray,
    random_state: int = 42
) -> Dict[str, Union[pd.DataFrame, float, PCA]]:
    """
    Computes both 2D and 3D PCA representations along with explained variance ratios.

    Returns:
        dict: {
            "pca_2d": pd.DataFrame with ['PCA1', 'PCA2'],
            "pca_3d": pd.DataFrame with ['PCA1', 'PCA2', 'PCA3'],
            "var_ratio_2d": total explained variance ratio for 2D,
            "var_ratio_3d": total explained variance ratio for 3D,
            "pca_model_2d": fitted 2D PCA object,
            "pca_model_3d": fitted 3D PCA object
        }
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


run_pca = apply_pca

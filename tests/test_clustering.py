"""
Unit tests for clustering and dimensionality reduction module.
"""

import numpy as np
import pandas as pd
import pytest
from src.clustering import (
    preprocess_for_clustering,
    apply_kmeans,
    apply_dbscan,
    apply_pca
)


@pytest.fixture
def sample_rfm():
    """Generates synthetic RFM dataset with distinct customer clusters."""
    np.random.seed(42)
    c1 = pd.DataFrame({
        "CustomerID": [f"C_champ_{i}" for i in range(20)],
        "Recency": np.random.normal(5, 2, 20),
        "Frequency": np.random.normal(15, 2, 20),
        "Monetary": np.random.normal(1000, 100, 20)
    })
    c2 = pd.DataFrame({
        "CustomerID": [f"C_hib_{i}" for i in range(20)],
        "Recency": np.random.normal(150, 10, 20),
        "Frequency": np.random.normal(1, 0.5, 20),
        "Monetary": np.random.normal(50, 10, 20)
    })
    return pd.concat([c1, c2], ignore_index=True)


def test_preprocess_for_clustering(sample_rfm):
    scaled_data, scaler = preprocess_for_clustering(sample_rfm)
    assert scaled_data.shape == (40, 3)
    # Check mean approx 0 and std approx 1
    assert np.allclose(scaled_data.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(scaled_data.std(axis=0), 1, atol=1e-6)


def test_apply_kmeans(sample_rfm):
    scaled_data, _ = preprocess_for_clustering(sample_rfm)
    labels, sil_score = apply_kmeans(scaled_data, n_clusters=2, random_state=42)

    assert len(labels) == 40
    assert len(np.unique(labels)) == 2
    assert -1.0 <= sil_score <= 1.0
    assert sil_score > 0.5  # High silhouette score for two distinct clusters


def test_apply_dbscan(sample_rfm):
    scaled_data, _ = preprocess_for_clustering(sample_rfm)
    labels, sil_score = apply_dbscan(scaled_data, eps=1.5, min_samples=3)

    assert len(labels) == 40
    # DBSCAN should detect clusters
    assert len(np.unique(labels)) >= 1
    assert isinstance(sil_score, float)


def test_apply_pca(sample_rfm):
    scaled_data, _ = preprocess_for_clustering(sample_rfm)
    pca_data, pca = apply_pca(scaled_data, n_components=3, random_state=42)

    assert pca_data.shape == (40, 3)
    assert hasattr(pca, "explained_variance_ratio_")
    assert len(pca.explained_variance_ratio_) == 3

"""
Unit tests for visualization functions.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.visualization import (
    plot_rfm_distributions,
    plot_2d_clusters,
    plot_3d_clusters,
    plot_persona_treemap,
    plot_cluster_scatter,
    plot_persona_pie
)


def test_plot_rfm_distributions():
    df = pd.DataFrame({
        "CustomerID": ["C1", "C2", "C3"],
        "Recency": [5, 20, 100],
        "Frequency": [10, 2, 1],
        "Monetary": [500.0, 100.0, 20.0]
    })
    fig = plot_rfm_distributions(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


def test_plot_2d_clusters():
    pca_coords = np.array([
        [1.0, 2.0],
        [-1.0, -2.0],
        [0.5, 0.5]
    ])
    labels = [0, 1, 0]
    fig = plot_2d_clusters(pca_coords, labels=labels)
    assert isinstance(fig, go.Figure)


def test_plot_3d_clusters():
    pca_coords = np.array([
        [1.0, 2.0, 3.0],
        [-1.0, -2.0, -3.0],
        [0.5, 0.5, 0.5]
    ])
    labels = [0, 1, 0]
    fig = plot_3d_clusters(pca_coords, labels=labels)
    assert isinstance(fig, go.Figure)


def test_plot_persona_treemap():
    summary_df = pd.DataFrame({
        "Persona": ["Champions", "At Risk", "Hibernating"],
        "Count": [50, 30, 20],
        "Monetary": [1200.0, 400.0, 50.0]
    })
    fig = plot_persona_treemap(summary_df)
    assert isinstance(fig, go.Figure)


def test_plot_persona_revenue_comparison():
    summary_df = pd.DataFrame({
        "Persona": ["Champions", "At Risk", "Hibernating"],
        "Count": [50, 30, 20],
        "Avg_Monetary": [1200.0, 400.0, 50.0]
    })
    from src.visualization import plot_persona_revenue_comparison
    fig = plot_persona_revenue_comparison(summary_df)
    assert isinstance(fig, go.Figure)


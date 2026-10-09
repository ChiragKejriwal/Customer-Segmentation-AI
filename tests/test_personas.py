"""
Unit tests for persona generation module.
"""

import pandas as pd
import numpy as np
import pytest
from src.personas import generate_personas, assign_personas_by_rfm, PERSONA_MAP


@pytest.fixture
def segmented_rfm_data():
    """Synthetic dataset with 3 distinct clusters."""
    # Cluster 0: Champions (low R, high F, high M)
    c0 = pd.DataFrame({
        "CustomerID": ["C0_1", "C0_2", "C0_3"],
        "Recency": [2, 5, 4],
        "Frequency": [20, 25, 30],
        "Monetary": [2000.0, 3000.0, 2500.0],
        "Cluster": [0, 0, 0]
    })
    # Cluster 1: At Risk (high R, high F, high M)
    c1 = pd.DataFrame({
        "CustomerID": ["C1_1", "C1_2", "C1_3"],
        "Recency": [120, 150, 130],
        "Frequency": [15, 18, 12],
        "Monetary": [1500.0, 1800.0, 1200.0],
        "Cluster": [1, 1, 1]
    })
    # Cluster 2: Hibernating (high R, low F, low M)
    c2 = pd.DataFrame({
        "CustomerID": ["C2_1", "C2_2", "C2_3"],
        "Recency": [180, 200, 220],
        "Frequency": [1, 1, 2],
        "Monetary": [30.0, 40.0, 50.0],
        "Cluster": [2, 2, 2]
    })
    return pd.concat([c0, c1, c2], ignore_index=True)


def test_generate_personas(segmented_rfm_data):
    cluster_labels = segmented_rfm_data["Cluster"].values
    rfm_df = segmented_rfm_data[["CustomerID", "Recency", "Frequency", "Monetary"]]

    result = generate_personas(rfm_df, cluster_labels)

    assert "Persona" in result.columns
    assert "Cluster" in result.columns
    assert len(result) == 9

    # Verify that every regular cluster gets a unique persona name (no duplicates)
    unique_personas = result["Persona"].unique()
    assert len(unique_personas) == 3

    # Check that high value cluster receives Whale VIPs/Champions persona
    c0_persona = result[result["Cluster"] == 0]["Persona"].iloc[0]
    c1_persona = result[result["Cluster"] == 1]["Persona"].iloc[0]
    c2_persona = result[result["Cluster"] == 2]["Persona"].iloc[0]

    assert "VIP" in c0_persona or "Champion" in c0_persona
    assert "Risk" in c1_persona or "Lapsed" in c1_persona
    assert "Hibernating" in c2_persona or "Churn" in c2_persona


def test_generate_personas_with_outliers():
    df = pd.DataFrame({
        "CustomerID": ["O1", "O2"],
        "Recency": [10, 20],
        "Frequency": [2, 3],
        "Monetary": [100.0, 150.0]
    })
    labels = [-1, -1]
    result = generate_personas(df, labels)
    assert all("Noise" in p or "Outlier" in p or "Anomal" in p for p in result["Persona"])


def test_generate_personas_empty():
    empty_df = pd.DataFrame()
    result = generate_personas(empty_df, [])
    assert result.empty
    assert "Persona" in result.columns

"""
Interactive Visualization Module using Plotly.
Provides modern, high-contrast charts for RFM distributions, 2D/3D cluster projections,
centroid overlays, and persona composition treemaps.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


HIGH_CONTRAST_PALETTE = [
    "#6366F1",  # Electric Indigo
    "#10B981",  # Emerald Green
    "#F59E0B",  # Amber / Gold
    "#EC4899",  # Vivid Pink
    "#3B82F6",  # Bright Blue
    "#8B5CF6",  # Deep Violet
    "#14B8A6",  # Teal
    "#F43F5E",  # Rose Red
    "#84CC16",  # Lime Green
    "#06B6D4",  # Cyan
]


def _get_trimmed_axis_range(series: pd.Series, padding_factor: float = 0.1) -> list:
    """Calculates robust axis bounds using 1st-99th percentiles and IQR to avoid squishing."""
    q01 = series.quantile(0.01)
    q99 = series.quantile(0.99)
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25

    lower = max(series.min(), min(q01, q25 - 1.5 * iqr))
    upper = min(series.max(), max(q99, q75 + 1.5 * iqr))
    span = (upper - lower) or 1.0

    return [float(lower - span * padding_factor), float(upper + span * padding_factor)]


def plot_rfm_distributions(rfm_df: pd.DataFrame) -> go.Figure:
    """
    Returns a multi-panel distribution figure for Recency, Frequency, and Monetary metrics.
    Applies logarithmic/quantile-aware binning for Monetary and Frequency to prevent single-bar collapse.
    """
    if rfm_df is None or rfm_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No RFM Data Available", template="plotly_dark")
        return fig

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "<b>Recency Distribution</b> (Days)",
            "<b>Frequency Distribution</b> (Orders)",
            "<b>Monetary Distribution</b> (Spend $)"
        ),
        horizontal_spacing=0.08
    )

    # 1. Recency Histogram
    fig.add_trace(
        go.Histogram(
            x=rfm_df["Recency"],
            name="Recency",
            marker_color="#6366F1",
            opacity=0.85,
            nbinsx=35
        ),
        row=1, col=1
    )

    # 2. Frequency Histogram (using log-scale on Y to show full distribution)
    freq_cap = rfm_df["Frequency"].quantile(0.99)
    fig.add_trace(
        go.Histogram(
            x=rfm_df["Frequency"],
            name="Frequency",
            marker_color="#10B981",
            opacity=0.85,
            nbinsx=30
        ),
        row=1, col=2
    )

    # 3. Monetary Histogram (log scale on Y to display both mid and whale spenders)
    fig.add_trace(
        go.Histogram(
            x=rfm_df["Monetary"],
            name="Monetary",
            marker_color="#EC4899",
            opacity=0.85,
            nbinsx=35
        ),
        row=1, col=3
    )

    fig.update_yaxes(type="log", row=1, col=2, title_text="Count (Log Scale)")
    fig.update_yaxes(type="log", row=1, col=3, title_text="Count (Log Scale)")
    fig.update_yaxes(row=1, col=1, title_text="Customer Count")

    fig.update_layout(
        title="<b>Customer RFM Metric Distributions</b> (Log-Scaled Y-Axes for Power-Law Features)",
        title_font=dict(size=16, color="#F8FAFC"),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        showlegend=False,
        bargap=0.08,
        margin=dict(l=40, r=30, t=70, b=40),
        font=dict(family="Inter, Roboto, sans-serif", color="#E2E8F0")
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")

    return fig


def plot_2d_clusters(
    pca_df: Union[pd.DataFrame, np.ndarray],
    labels: Optional[Union[np.ndarray, pd.Series, list]] = None,
    customer_ids: Optional[Union[np.ndarray, pd.Series, list]] = None
) -> go.Figure:
    """
    Renders a crisp 2D PCA cluster scatter plot with prominent centroid markers.
    """
    if isinstance(pca_df, np.ndarray):
        cols = [f"PCA{i+1}" for i in range(pca_df.shape[1])]
        df = pd.DataFrame(pca_df, columns=cols)
    else:
        df = pca_df.copy()

    if labels is not None:
        df["Cluster"] = [f"Cluster {l}" if str(l) != "-1" else "Outliers" for l in labels]
    elif "Cluster" not in df.columns:
        df["Cluster"] = "Cluster 0"

    if customer_ids is not None:
        df["CustomerID"] = [str(cid) for cid in customer_ids]

    pca_cols = [c for c in df.columns if c.startswith("PCA")]
    x_col = pca_cols[0] if len(pca_cols) > 0 else df.columns[0]
    y_col = pca_cols[1] if len(pca_cols) > 1 else df.columns[1]

    fig = go.Figure()

    # Plot customer data points per cluster
    unique_clusters = df["Cluster"].unique()
    for idx, cluster_name in enumerate(sorted(unique_clusters)):
        cluster_mask = df["Cluster"] == cluster_name
        cluster_data = df[cluster_mask]
        color = "#6B7280" if cluster_name == "Outliers" else HIGH_CONTRAST_PALETTE[idx % len(HIGH_CONTRAST_PALETTE)]

        hover_text = [
            f"<b>{cluster_name}</b><br>ID: {cid}<br>{x_col}: {x:.2f}<br>{y_col}: {y:.2f}"
            for cid, x, y in zip(cluster_data.get("CustomerID", ["N/A"] * len(cluster_data)), cluster_data[x_col], cluster_data[y_col])
        ]

        fig.add_trace(
            go.Scatter(
                x=cluster_data[x_col],
                y=cluster_data[y_col],
                mode="markers",
                name=str(cluster_name),
                text=hover_text,
                hoverinfo="text",
                marker=dict(
                    size=6,
                    color=color,
                    opacity=0.75,
                    line=dict(width=0.5, color="white")
                )
            )
        )

        # Plot cluster centroid if not outlier
        if cluster_name != "Outliers" and len(cluster_data) > 0:
            cx = cluster_data[x_col].mean()
            cy = cluster_data[y_col].mean()
            fig.add_trace(
                go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="markers+text",
                    name=f"{cluster_name} Centroid",
                    text=[f"★ {cluster_name}"],
                    textposition="top center",
                    textfont=dict(size=11, color="#FFFFFF", family="Inter, sans-serif"),
                    showlegend=False,
                    marker=dict(
                        symbol="diamond",
                        size=14,
                        color=color,
                        line=dict(width=2, color="#FFFFFF")
                    )
                )
            )

    # Trim axis range using robust percentile bounds
    x_range = _get_trimmed_axis_range(df[x_col])
    y_range = _get_trimmed_axis_range(df[y_col])

    fig.update_layout(
        title="<b>2D PCA Latent Space Cluster Projection</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(family="Inter, Roboto, sans-serif", color="#E2E8F0"),
        xaxis=dict(title=x_col, range=x_range, showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)"),
        yaxis=dict(title=y_col, range=y_range, showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def plot_3d_clusters(
    pca_df: Union[pd.DataFrame, np.ndarray],
    labels: Optional[Union[np.ndarray, pd.Series, list]] = None,
    customer_ids: Optional[Union[np.ndarray, pd.Series, list]] = None
) -> go.Figure:
    """
    Renders a high-contrast, unsquished 3D scatter plot of PCA components with centroid overlays.
    """
    if isinstance(pca_df, np.ndarray):
        cols = [f"PCA{i+1}" for i in range(pca_df.shape[1])]
        df = pd.DataFrame(pca_df, columns=cols)
    else:
        df = pca_df.copy()

    if labels is not None:
        df["Cluster"] = [f"Cluster {l}" if str(l) != "-1" else "Outliers" for l in labels]
    elif "Cluster" not in df.columns:
        df["Cluster"] = "Cluster 0"

    if customer_ids is not None:
        df["CustomerID"] = [str(cid) for cid in customer_ids]

    pca_cols = [c for c in df.columns if c.startswith("PCA")]
    x_col = pca_cols[0] if len(pca_cols) > 0 else df.columns[0]
    y_col = pca_cols[1] if len(pca_cols) > 1 else df.columns[1]
    z_col = pca_cols[2] if len(pca_cols) > 2 else df.columns[2]

    fig = go.Figure()

    unique_clusters = df["Cluster"].unique()
    for idx, cluster_name in enumerate(sorted(unique_clusters)):
        cluster_mask = df["Cluster"] == cluster_name
        cluster_data = df[cluster_mask]
        color = "#6B7280" if cluster_name == "Outliers" else HIGH_CONTRAST_PALETTE[idx % len(HIGH_CONTRAST_PALETTE)]

        hover_text = [
            f"<b>{cluster_name}</b><br>ID: {cid}<br>{x_col}: {x:.2f}<br>{y_col}: {y:.2f}<br>{z_col}: {z:.2f}"
            for cid, x, y, z in zip(
                cluster_data.get("CustomerID", ["N/A"] * len(cluster_data)),
                cluster_data[x_col],
                cluster_data[y_col],
                cluster_data[z_col]
            )
        ]

        fig.add_trace(
            go.Scatter3d(
                x=cluster_data[x_col],
                y=cluster_data[y_col],
                z=cluster_data[z_col],
                mode="markers",
                name=str(cluster_name),
                text=hover_text,
                hoverinfo="text",
                marker=dict(
                    size=5,
                    color=color,
                    opacity=0.8,
                    line=dict(width=0.5, color="white")
                )
            )
        )

        # Plot cluster centroid
        if cluster_name != "Outliers" and len(cluster_data) > 0:
            cx = cluster_data[x_col].mean()
            cy = cluster_data[y_col].mean()
            cz = cluster_data[z_col].mean()

            fig.add_trace(
                go.Scatter3d(
                    x=[cx],
                    y=[cy],
                    z=[cz],
                    mode="markers+text",
                    name=f"{cluster_name} Centroid",
                    text=[f"★ {cluster_name}"],
                    textposition="top center",
                    textfont=dict(size=12, color="#FFFFFF"),
                    showlegend=False,
                    marker=dict(
                        symbol="diamond",
                        size=10,
                        color=color,
                        line=dict(width=2, color="#FFFFFF")
                    )
                )
            )

    # Calculate trimmed axis ranges to eliminate empty space padding
    x_range = _get_trimmed_axis_range(df[x_col])
    y_range = _get_trimmed_axis_range(df[y_col])
    z_range = _get_trimmed_axis_range(df[z_col])

    fig.update_layout(
        title="<b>3D PCA Customer Segment Space</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(family="Inter, Roboto, sans-serif", color="#E2E8F0"),
        margin=dict(l=10, r=10, t=50, b=10),
        scene=dict(
            aspectmode="cube",
            xaxis=dict(
                title=x_col,
                range=x_range,
                backgroundcolor="rgba(15, 23, 42, 0.8)",
                gridcolor="rgba(255, 255, 255, 0.15)"
            ),
            yaxis=dict(
                title=y_col,
                range=y_range,
                backgroundcolor="rgba(15, 23, 42, 0.8)",
                gridcolor="rgba(255, 255, 255, 0.15)"
            ),
            zaxis=dict(
                title=z_col,
                range=z_range,
                backgroundcolor="rgba(15, 23, 42, 0.8)",
                gridcolor="rgba(255, 255, 255, 0.15)"
            )
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


PERSONA_COLOR_MAP = {
    "Whale VIPs / Champions": "#10B981",       # Vibrant Emerald
    "Loyal Growth Spenders": "#06B6D4",        # Luminous Cyan
    "Recent Promising Shoppers": "#6366F1",    # Electric Indigo
    "Potential Loyalists": "#8B5CF6",          # Royal Violet
    "At-Risk High Spenders": "#F43F5E",        # Vibrant Rose Coral
    "Hibernating / Churned": "#64748B",        # Cool Slate Muted
    "Promising / New Customers": "#3B82F6",    # Bright Blue
    "Need Attention": "#F59E0B",               # Amber Gold
    "About to Sleep": "#E11D48",               # Crimson
}


def plot_persona_treemap(persona_summary_df: pd.DataFrame) -> go.Figure:
    """
    Returns an executive, beautifully styled Treemap showing customer count and monetary metrics.
    Features custom HTML-formatted text labels, vibrant distinct segment colors, and rich tooltips.
    """
    if persona_summary_df is None or persona_summary_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Persona Summary Available", template="plotly_dark")
        return fig

    df = persona_summary_df.copy()
    if "Count" not in df.columns and "CustomerID" in df.columns:
        df = df.groupby("Persona").agg(
            Count=("CustomerID", "count"),
            Monetary=("Monetary", "mean")
        ).reset_index()

    # Determine monetary column
    monetary_col = "Avg_Monetary" if "Avg_Monetary" in df.columns else ("Monetary" if "Monetary" in df.columns else None)
    if monetary_col:
        df["_avg_spend"] = df[monetary_col]
        df["_total_rev"] = df[monetary_col] * df["Count"]
    else:
        df["_avg_spend"] = 0.0
        df["_total_rev"] = 0.0

    total_customers = df["Count"].sum()
    df["_pct_share"] = (df["Count"] / max(total_customers, 1) * 100.0).round(1)

    # Assign distinct colors
    df["_color"] = [
        PERSONA_COLOR_MAP.get(str(p).strip(), HIGH_CONTRAST_PALETTE[i % len(HIGH_CONTRAST_PALETTE)])
        for i, p in enumerate(df["Persona"])
    ]

    fig = px.treemap(
        df,
        path=["Persona"],
        values="Count",
        color="Persona",
        color_discrete_map=PERSONA_COLOR_MAP,
        title="<b>👥 Customer Persona Composition & Portfolio Share</b>",
        template="plotly_dark"
    )

    # Attach rich customdata for text and hover templates:
    # [0: Avg Spend, 1: Total Segment Revenue, 2: Pct Share]
    customdata = np.stack(
        (df["_avg_spend"].values, df["_total_rev"].values, df["_pct_share"].values),
        axis=-1
    )

    fig.update_traces(
        customdata=customdata,
        texttemplate=(
            "<b><span style='font-size:15px;color:#FFFFFF;'>%{label}</span></b><br><br>"
            "<span style='font-size:13px;font-weight:600;color:#F8FAFC;'>👥 %{value:,} Customers</span> "
            "<span style='font-size:12px;color:#94A3B8;'>(%{customdata[2]:.1f}%)</span><br>"
            "<span style='font-size:12px;font-weight:700;color:#FDE047;'>💰 Avg LTV: $%{customdata[0]:,.2f}</span>"
        ),
        hovertemplate=(
            "<b style='font-size:15px;color:#38BDF8;'>%{label}</b><br>"
            "────────────────────────────────<br>"
            "👥 <b>Customer Count:</b> %{value:,}<br>"
            "📊 <b>Segment Share:</b> %{customdata[2]:.1f}%<br>"
            "💳 <b>Average Customer LTV:</b> $%{customdata[0]:,.2f}<br>"
            "💵 <b>Estimated Segment Revenue:</b> $%{customdata[1]:,.2f}<br>"
            "<extra></extra>"
        ),
        marker=dict(
            cornerradius=8,
            line=dict(width=2, color="rgba(15, 23, 42, 0.95)")
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.75)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        font=dict(family="Inter, -apple-system, sans-serif", color="#E2E8F0"),
        title_font=dict(size=16, color="#F8FAFC"),
        margin=dict(l=15, r=15, t=55, b=15),
        showlegend=False
    )
    return fig


def plot_persona_revenue_comparison(persona_summary_df: pd.DataFrame) -> go.Figure:
    """
    Renders a modern horizontal bar chart comparing Total Estimated Revenue and Customer Share by Persona.
    """
    if persona_summary_df is None or persona_summary_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available", template="plotly_dark")
        return fig

    df = persona_summary_df.copy()
    monetary_col = "Avg_Monetary" if "Avg_Monetary" in df.columns else ("Monetary" if "Monetary" in df.columns else None)
    if monetary_col:
        df["Total_Revenue"] = df[monetary_col] * df["Count"]
    else:
        df["Total_Revenue"] = df["Count"]

    df = df.sort_values(by="Total_Revenue", ascending=True)

    colors = [
        PERSONA_COLOR_MAP.get(str(p).strip(), HIGH_CONTRAST_PALETTE[i % len(HIGH_CONTRAST_PALETTE)])
        for i, p in enumerate(df["Persona"])
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["Persona"],
            x=df["Total_Revenue"],
            orientation="h",
            marker=dict(
                color=colors,
                cornerradius=6,
                line=dict(width=1, color="rgba(255, 255, 255, 0.15)")
            ),
            text=[f"${rev:,.0f} ({cnt:,} users)" for rev, cnt in zip(df["Total_Revenue"], df["Count"])],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Total Revenue: $%{x:,.2f}<extra></extra>"
        )
    )

    fig.update_layout(
        title="<b>💵 Total Revenue Contribution by Persona Segment</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.75)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(family="Inter, -apple-system, sans-serif", color="#E2E8F0"),
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(title="Estimated Total Revenue ($)", showgrid=True, gridcolor="rgba(255, 255, 255, 0.08)"),
        yaxis=dict(title="", showgrid=False)
    )
    return fig


def plot_cluster_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str = "Customer Clusters"
) -> go.Figure:
    """Creates a 2D interactive scatter plot of clusters."""
    return plot_2d_clusters(df[[x_col, y_col]], labels=df[color_col], customer_ids=df.get("CustomerID"))


def plot_persona_pie(persona_counts: pd.DataFrame) -> go.Figure:
    """Generates a donut chart for customer personas with cohesive color mapping."""
    df = persona_counts.copy()
    colors = [
        PERSONA_COLOR_MAP.get(str(p).strip(), HIGH_CONTRAST_PALETTE[i % len(HIGH_CONTRAST_PALETTE)])
        for i, p in enumerate(df["Persona"])
    ]

    fig = px.pie(
        df,
        names="Persona",
        values="Count",
        hole=0.48,
        title="<b>Customer Distribution by Persona</b>",
        template="plotly_dark",
        color="Persona",
        color_discrete_map=PERSONA_COLOR_MAP
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#0F172A", width=2))
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.75)",
        font=dict(family="Inter, -apple-system, sans-serif", color="#E2E8F0"),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    return fig

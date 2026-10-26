"""
Customer Persona Generator & Marketing Strategy Module.
Maps unsupervised clusters dynamically into unique, non-overlapping marketing personas.
"""

from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd


PERSONA_LIBRARY: List[Dict[str, Any]] = [
    {
        "name": "Whale VIPs / Champions",
        "condition": lambda r, f, m, v, e: v >= 0.75 and r >= 0.5,
        "description": "Highest lifetime spend and frequent recent purchases. Brand advocates.",
        "action": "VIP dedicated support, early access product drops, and exclusive loyalty perks.",
        "color": "#10B981"
    },
    {
        "name": "At-Risk High Spenders",
        "condition": lambda r, f, m, v, e: v >= 0.55 and r < 0.45,
        "description": "High historical spend but long inactivity gap. High churn hazard.",
        "action": "Personalized executive win-back emails, premium reactivation discounts, and feedback surveys.",
        "color": "#F59E0B"
    },
    {
        "name": "Loyal Core Buyers",
        "condition": lambda r, f, m, v, e: f >= 0.5 and v >= 0.4 and r >= 0.4,
        "description": "Consistent repeat purchasers driving high volume and steady revenue.",
        "action": "Upsell higher-margin products, cross-sell adjacent categories, and solicit reviews.",
        "color": "#3B82F6"
    },
    {
        "name": "Recent Promising Shoppers",
        "condition": lambda r, f, m, v, e: r >= 0.65 and f < 0.6,
        "description": "Bought recently with high early conversion momentum but modest history.",
        "action": "Welcome email nurture series, 2nd-purchase time-limited coupon, and brand education.",
        "color": "#8B5CF6"
    },
    {
        "name": "High-Ticket Occasionals",
        "condition": lambda r, f, m, v, e: m >= 0.6 and f < 0.5,
        "description": "Large transaction values placed infrequently (e.g. seasonal wholesale or luxury buyers).",
        "action": "Annual holiday catalog outreach, volume tiers, and corporate gift reminders.",
        "color": "#06B6D4"
    },
    {
        "name": "Bargain Frequent Shoppers",
        "condition": lambda r, f, m, v, e: f >= 0.6 and m < 0.5,
        "description": "Shop very frequently but maintain smaller basket sizes.",
        "action": "Free shipping minimum basket thresholds and bundle upsells to increase AOV.",
        "color": "#EC4899"
    },
    {
        "name": "Developing Regulars",
        "condition": lambda r, f, m, v, e: v >= 0.3 and r >= 0.35,
        "description": "Average recency and spend with strong headroom for tier upgrade.",
        "action": "Loyalty point milestones and personalized recommendation feeds.",
        "color": "#14B8A6"
    },
    {
        "name": "Hibernating / Churned",
        "condition": lambda r, f, m, v, e: v <= 0.35 and r <= 0.35,
        "description": "Long time since last order, minimal orders, and low lifetime value.",
        "action": "Aggressive flash clearance promotions or automated sunsetting to protect deliverability.",
        "color": "#EF4444"
    }
]

PERSONA_MAP: Dict[str, Dict[str, str]] = {
    p["name"]: {"description": p["description"], "action": p["action"], "color": p["color"]}
    for p in PERSONA_LIBRARY
}
PERSONA_MAP["Anomalies / Noise Outliers"] = {
    "description": "Extreme transaction outliers or anomalous records identified by density clustering.",
    "action": "Manual account review for data entry anomalies, wholesale bulk buyers, or payment disputes.",
    "color": "#6B7280"
}
# Backward compatibility aliases
PERSONA_MAP["Champions"] = PERSONA_MAP["Whale VIPs / Champions"]
PERSONA_MAP["At Risk"] = PERSONA_MAP["At-Risk High Spenders"]
PERSONA_MAP["Loyal Customers"] = PERSONA_MAP["Loyal Core Buyers"]
PERSONA_MAP["Potential Loyalists"] = PERSONA_MAP["Recent Promising Shoppers"]
PERSONA_MAP["Hibernating"] = PERSONA_MAP["Hibernating / Churned"]


def generate_personas(
    rfm_df: pd.DataFrame,
    cluster_labels: Union[np.ndarray, pd.Series, list, str]
) -> pd.DataFrame:
    """
    Assigns unique, non-overlapping marketing persona names to each cluster based on centroid RFM ranks.
    """
    if rfm_df is None or rfm_df.empty:
        return pd.DataFrame(columns=["CustomerID", "Recency", "Frequency", "Monetary", "Cluster", "Persona"])

    df = rfm_df.copy()
    if isinstance(cluster_labels, str) and cluster_labels in df.columns:
        df["Cluster"] = df[cluster_labels]
    else:
        df["Cluster"] = cluster_labels

    unique_clusters = [c for c in df["Cluster"].unique() if str(c) != "-1" and c != -1]
    k = len(unique_clusters)
    cluster_persona_map: Dict[Any, str] = {}

    if -1 in df["Cluster"].values or "-1" in df["Cluster"].values:
        cluster_persona_map[-1] = "Anomalies / Noise Outliers"
        cluster_persona_map["-1"] = "Anomalies / Noise Outliers"

    if k == 0:
        df["Persona"] = df["Cluster"].map(cluster_persona_map).fillna("Anomalies / Noise Outliers")
        return df

    stats = df[df["Cluster"].isin(unique_clusters)].groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    if k == 1:
        cluster_persona_map[unique_clusters[0]] = "Loyal Core Buyers"
    else:
        # Lower recency is better (higher rank)
        r_scores = (stats["Recency"].max() - stats["Recency"]) / ((stats["Recency"].max() - stats["Recency"].min()) or 1.0)
        f_scores = (stats["Frequency"] - stats["Frequency"].min()) / ((stats["Frequency"].max() - stats["Frequency"].min()) or 1.0)
        m_scores = (stats["Monetary"] - stats["Monetary"].min()) / ((stats["Monetary"].max() - stats["Monetary"].min()) or 1.0)

        v_scores = 0.5 * m_scores + 0.3 * f_scores + 0.2 * r_scores
        e_scores = 0.6 * r_scores + 0.4 * f_scores

        assigned_personas = set()
        sorted_clusters = v_scores.sort_values(ascending=False).index.tolist()

        for c_id in sorted_clusters:
            r = float(r_scores[c_id])
            f = float(f_scores[c_id])
            m = float(m_scores[c_id])
            v = float(v_scores[c_id])
            e = float(e_scores[c_id])

            best_match = None
            for p in PERSONA_LIBRARY:
                if p["name"] not in assigned_personas and p["condition"](r, f, m, v, e):
                    best_match = p["name"]
                    break

            if not best_match:
                for p in PERSONA_LIBRARY:
                    if p["name"] not in assigned_personas:
                        best_match = p["name"]
                        break

            if not best_match:
                best_match = f"Segment Cohort {c_id}"

            assigned_personas.add(best_match)
            cluster_persona_map[c_id] = best_match

    df["Persona"] = df["Cluster"].map(cluster_persona_map).fillna("Developing Regulars")
    return df


def assign_personas_by_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based quantile persona assignment when clustering model has not been executed.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    if "R_Score" not in df.columns:
        from src.rfm import score_rfm
        df = score_rfm(df)

    def _get_persona(row):
        r = row.get("R_Score", 3)
        f = row.get("F_Score", 3)
        m = row.get("M_Score", 3)

        if r >= 4 and f >= 4 and m >= 4:
            return "Whale VIPs / Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Core Buyers"
        elif r >= 4 and f <= 2:
            return "Recent Promising Shoppers"
        elif r <= 2 and (f >= 3 or m >= 3):
            return "At-Risk High Spenders"
        return "Hibernating / Churned"

    res = df.copy()
    res["Persona"] = res.apply(_get_persona, axis=1)
    return res

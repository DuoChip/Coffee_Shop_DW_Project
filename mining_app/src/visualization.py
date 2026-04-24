import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

# ── Dark navy theme palette ─────────────────────────────────────────
BG_COLOR = "#0e1117"
CARD_BG = "#161b22"
GRID_COLOR = "#1f2937"
TEXT_COLOR = "#c9d1d9"
ACCENT = "#58a6ff"
ACCENT_COLORS = [
    "#58a6ff", "#3fb950", "#f0883e", "#bc8cff",
    "#ff7b72", "#79c0ff", "#56d364", "#ffa657",
    "#d2a8ff", "#ffa198",
]

_LAYOUT_COMMON = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif", size=12),
    title_font=dict(size=14, color="#ffffff"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)),
    colorway=ACCENT_COLORS,
)


def _apply_theme(fig):
    """Apply the dark navy theme to any plotly figure."""
    fig.update_layout(**_LAYOUT_COMMON)
    return fig


# ── Overview charts ──────────────────────────────────────────────────

def fig_sales_trend(df: pd.DataFrame):
    trend = (
        df.groupby("transaction_date", as_index=False)["revenue"]
          .sum()
          .sort_values("transaction_date")
    )
    fig = px.line(
        trend,
        x="transaction_date",
        y="revenue",
        title="Sales Trend Over Time",
    )
    fig.update_traces(line=dict(color=ACCENT, width=2))
    return _apply_theme(fig)


def fig_revenue_by_day(df: pd.DataFrame):
    temp = df.copy()
    temp["day_name"] = pd.Categorical(temp["day_name"], categories=DAY_ORDER, ordered=True)
    chart_df = (
        temp.groupby("day_name", as_index=False)["revenue"]
            .sum()
            .sort_values("day_name")
    )
    fig = px.bar(
        chart_df,
        x="day_name",
        y="revenue",
        title="Revenue by Day of Week",
        color_discrete_sequence=[ACCENT],
    )
    return _apply_theme(fig)


def fig_peak_hour(df: pd.DataFrame):
    chart_df = (
        df.groupby("hour_number", as_index=False)["revenue"]
          .sum()
          .sort_values("hour_number")
    )
    fig = px.line(
        chart_df,
        x="hour_number",
        y="revenue",
        markers=True,
        title="Revenue by Hour",
    )
    fig.update_traces(line=dict(color=ACCENT, width=2), marker=dict(size=6))
    return _apply_theme(fig)


def fig_top_products(df: pd.DataFrame, top_n: int = 10):
    chart_df = (
        df.groupby("product_detail", as_index=False)["revenue"]
          .sum()
          .sort_values("revenue", ascending=True)
          .tail(top_n)
    )
    fig = px.bar(
        chart_df,
        y="product_detail",
        x="revenue",
        orientation="h",
        title=f"Top {top_n} Products by Revenue",
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(yaxis_title="", xaxis_title="Revenue")
    return _apply_theme(fig)


def fig_top_categories(df: pd.DataFrame):
    chart_df = (
        df.groupby("product_category", as_index=False)["revenue"]
          .sum()
          .sort_values("revenue", ascending=False)
    )
    fig = px.bar(
        chart_df,
        x="product_category",
        y="revenue",
        title="Revenue by Product Category",
        color_discrete_sequence=[ACCENT],
    )
    return _apply_theme(fig)


def fig_category_pie(df: pd.DataFrame):
    """Pie / donut chart showing revenue share by product category."""
    chart_df = (
        df.groupby("product_category", as_index=False)["revenue"]
          .sum()
          .sort_values("revenue", ascending=False)
    )
    fig = px.pie(
        chart_df,
        names="product_category",
        values="revenue",
        title="Revenue by Category",
        hole=0.35,
        color_discrete_sequence=ACCENT_COLORS,
    )
    fig.update_traces(textinfo="percent+label", textfont_size=11)
    return _apply_theme(fig)


def fig_store_revenue(df: pd.DataFrame):
    chart_df = (
        df.groupby("store_location", as_index=False)["revenue"]
          .sum()
          .sort_values("revenue", ascending=False)
    )
    fig = px.bar(
        chart_df,
        x="store_location",
        y="revenue",
        title="Revenue by Store",
        color_discrete_sequence=[ACCENT],
    )
    return _apply_theme(fig)


# ── Association charts ───────────────────────────────────────────────

def fig_association_rules(rules: pd.DataFrame, top_n: int = 10):
    if rules.empty:
        return None

    chart_df = rules.head(top_n).copy()
    fig = px.bar(
        chart_df,
        x="rule",
        y="lift",
        hover_data=["support", "confidence"],
        title=f"Top {top_n} Association Rules by Lift",
        color_discrete_sequence=[ACCENT],
    )
    return _apply_theme(fig)


def fig_support_confidence_scatter(rules: pd.DataFrame):
    if rules.empty:
        return None

    fig = px.scatter(
        rules,
        x="support",
        y="confidence",
        size="lift",
        hover_name="rule",
        title="Association Rules: Support vs Confidence",
        color_discrete_sequence=ACCENT_COLORS,
    )
    return _apply_theme(fig)


# ── Clustering charts ────────────────────────────────────────────────

def fig_elbow(elbow_df: pd.DataFrame):
    fig = px.line(
        elbow_df,
        x="k",
        y="inertia",
        markers=True,
        title="Elbow Method",
    )
    fig.update_traces(line=dict(color=ACCENT, width=2))
    return _apply_theme(fig)


def fig_silhouette(elbow_df: pd.DataFrame):
    fig = px.line(
        elbow_df,
        x="k",
        y="silhouette_score",
        markers=True,
        title="Silhouette Score by K",
    )
    fig.update_traces(line=dict(color="#3fb950", width=2))
    return _apply_theme(fig)


def fig_cluster_pca(clustered_df: pd.DataFrame):
    fig = px.scatter(
        clustered_df,
        x="pca_1",
        y="pca_2",
        color=clustered_df["cluster"].astype(str),
        hover_data=["store_location", "date_only", "total_revenue", "peak_hour"],
        title="PCA Scatter Plot of Clusters",
        color_discrete_sequence=ACCENT_COLORS,
    )
    return _apply_theme(fig)


def fig_cluster_revenue(cluster_summary: pd.DataFrame):
    fig = px.bar(
        cluster_summary,
        x="cluster",
        y="avg_revenue",
        title="Average Revenue by Cluster",
        color_discrete_sequence=[ACCENT],
    )
    return _apply_theme(fig)


def fig_cluster_peak_hour(cluster_summary: pd.DataFrame):
    fig = px.bar(
        cluster_summary,
        x="cluster",
        y="avg_peak_hour",
        title="Average Peak Hour by Cluster",
        color_discrete_sequence=["#f0883e"],
    )
    return _apply_theme(fig)
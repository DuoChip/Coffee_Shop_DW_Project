import pandas as pd
import plotly.express as px


DAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]


def fig_sales_trend(df: pd.DataFrame):
    trend = (
        df.groupby("transaction_date", as_index=False)["revenue"]
          .sum()
          .sort_values("transaction_date")
    )
    return px.line(
        trend,
        x="transaction_date",
        y="revenue",
        title="Sales Trend Over Time"
    )


def fig_revenue_by_day(df: pd.DataFrame):
    temp = df.copy()
    temp["day_name"] = pd.Categorical(temp["day_name"], categories=DAY_ORDER, ordered=True)
    chart_df = (
        temp.groupby("day_name", as_index=False)["revenue"]
            .sum()
            .sort_values("day_name")
    )
    return px.bar(
        chart_df,
        x="day_name",
        y="revenue",
        title="Revenue by Day of Week"
    )


def fig_peak_hour(df: pd.DataFrame):
    chart_df = (
        df.groupby("hour_number", as_index=False)["revenue"]
          .sum()
          .sort_values("hour_number")
    )
    return px.line(
        chart_df,
        x="hour_number",
        y="revenue",
        markers=True,
        title="Revenue by Hour"
    )


def fig_top_products(df: pd.DataFrame, top_n: int = 10):
    chart_df = (
        df.groupby("product_detail", as_index=False)["revenue"]
          .sum()
          .sort_values("revenue", ascending=False)
          .head(top_n)
    )
    return px.bar(
        chart_df,
        x="product_detail",
        y="revenue",
        title=f"Top {top_n} Products by Revenue"
    )


def fig_top_categories(df: pd.DataFrame):
    chart_df = (
        df.groupby("product_category", as_index=False)["revenue"]
          .sum()
          .sort_values("revenue", ascending=False)
    )
    return px.bar(
        chart_df,
        x="product_category",
        y="revenue",
        title="Revenue by Product Category"
    )


def fig_store_revenue(df: pd.DataFrame):
    chart_df = (
        df.groupby("store_location", as_index=False)["revenue"]
          .sum()
          .sort_values("revenue", ascending=False)
    )
    return px.bar(
        chart_df,
        x="store_location",
        y="revenue",
        title="Revenue by Store"
    )


def fig_association_rules(rules: pd.DataFrame, top_n: int = 10):
    if rules.empty:
        return None

    chart_df = rules.head(top_n).copy()
    return px.bar(
        chart_df,
        x="rule",
        y="lift",
        hover_data=["support", "confidence"],
        title=f"Top {top_n} Association Rules by Lift"
    )


def fig_support_confidence_scatter(rules: pd.DataFrame):
    if rules.empty:
        return None

    return px.scatter(
        rules,
        x="support",
        y="confidence",
        size="lift",
        hover_name="rule",
        title="Association Rules: Support vs Confidence"
    )


def fig_elbow(elbow_df: pd.DataFrame):
    return px.line(
        elbow_df,
        x="k",
        y="inertia",
        markers=True,
        title="Elbow Method"
    )


def fig_silhouette(elbow_df: pd.DataFrame):
    return px.line(
        elbow_df,
        x="k",
        y="silhouette_score",
        markers=True,
        title="Silhouette Score by K"
    )


def fig_cluster_pca(clustered_df: pd.DataFrame):
    return px.scatter(
        clustered_df,
        x="pca_1",
        y="pca_2",
        color=clustered_df["cluster"].astype(str),
        hover_data=["store_location", "date_only", "total_revenue", "peak_hour"],
        title="PCA Scatter Plot of Clusters"
    )


def fig_cluster_revenue(cluster_summary: pd.DataFrame):
    return px.bar(
        cluster_summary,
        x="cluster",
        y="avg_revenue",
        title="Average Revenue by Cluster"
    )


def fig_cluster_peak_hour(cluster_summary: pd.DataFrame):
    return px.bar(
        cluster_summary,
        x="cluster",
        y="avg_peak_hour",
        title="Average Peak Hour by Cluster"
    )
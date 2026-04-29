import streamlit as st
import pandas as pd
#  streamlit run mining_app/app.py
from src.data_loader import build_analysis_dataframe
from src.feature_engineering import prepare_analysis_data, build_store_day_features
from src.association import run_apriori, get_top_rules
from src.clustering import calculate_elbow_and_silhouette, run_kmeans, summarize_clusters
from src.visualization import (
    fig_sales_trend,
    fig_revenue_by_day,
    fig_peak_hour,
    fig_top_products,
    fig_top_categories,
    fig_store_revenue,
    fig_association_rules,
    fig_support_confidence_scatter,
    fig_elbow,
    fig_silhouette,
    fig_cluster_pca,
    fig_cluster_revenue,
    fig_cluster_peak_hour,
)


st.set_page_config(
    page_title="Coffee Shop DSS Dashboard",
    layout="wide"
)


@st.cache_data
def load_data():
    df = build_analysis_dataframe()
    df = prepare_analysis_data(df)
    return df


def overview_page(df: pd.DataFrame):
    st.subheader("Executive Overview")

    total_revenue = df["revenue"].sum()
    total_qty = df["transaction_qty"].sum()
    total_transactions = df["transaction_id"].nunique()
    avg_order_value = total_revenue / total_transactions if total_transactions else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${total_revenue:,.2f}")
    c2.metric("Total Quantity", f"{int(total_qty):,}")
    c3.metric("Transactions", f"{int(total_transactions):,}")
    c4.metric("Avg Order Value", f"${avg_order_value:,.2f}")

    st.plotly_chart(fig_sales_trend(df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_revenue_by_day(df), use_container_width=True)
    with col2:
        st.plotly_chart(fig_peak_hour(df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_store_revenue(df), use_container_width=True)
    with col4:
        st.plotly_chart(fig_top_categories(df), use_container_width=True)

    st.plotly_chart(fig_top_products(df, top_n=10), use_container_width=True)


def association_page(df: pd.DataFrame):
    st.subheader("Association Rules")

    st.markdown("Dùng Apriori để tìm các sản phẩm thường xuất hiện cùng nhau trong cùng transaction.")

    col1, col2, col3 = st.columns(3)
    min_support = col1.slider("Min Support", 0.001, 0.100, 0.010, 0.001)
    min_confidence = col2.slider("Min Confidence", 0.10, 1.00, 0.20, 0.05)
    min_lift = col3.slider("Min Lift", 1.00, 5.00, 1.00, 0.10)

    product_col = st.selectbox(
        "Basket Product Level",
        options=["product_detail", "product_type", "product_category"],
        index=0
    )

    basket, frequent_itemsets, rules = run_apriori(
        df=df,
        product_col=product_col,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )

    st.write(f"Số transaction trong basket: {basket.shape[0]:,}")
    st.write(f"Số itemsets phổ biến: {len(frequent_itemsets):,}")
    st.write(f"Số luật kết hợp: {len(rules):,}")

    if rules.empty:
        st.warning("Không tìm thấy luật phù hợp với ngưỡng hiện tại.")
        return

    top_rules = get_top_rules(rules, top_n=10)

    st.dataframe(top_rules, use_container_width=True)

    fig1 = fig_association_rules(rules, top_n=10)
    if fig1 is not None:
        st.plotly_chart(fig1, use_container_width=True)

    fig2 = fig_support_confidence_scatter(rules)
    if fig2 is not None:
        st.plotly_chart(fig2, use_container_width=True)


def clustering_page(df: pd.DataFrame):
    st.subheader("Store Behavior Clustering")

    feature_df = build_store_day_features(df)

    st.markdown("Clustering được thực hiện trên mức **store-day** để tránh chỉ có 3 điểm dữ liệu nếu gom theo store thuần túy.")

    elbow_df = calculate_elbow_and_silhouette(
        feature_df.drop(columns=["store_id", "store_location", "date_only"]),
        k_min=2,
        k_max=6
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_elbow(elbow_df), use_container_width=True)
    with col2:
        st.plotly_chart(fig_silhouette(elbow_df), use_container_width=True)

    selected_k = st.selectbox("Chọn số cụm K", options=[2, 3, 4, 5, 6], index=1)

    clustered_df, centroids, scaler, model = run_kmeans(feature_df, n_clusters=selected_k)
    cluster_summary = summarize_clusters(clustered_df)

    st.dataframe(cluster_summary, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_cluster_pca(clustered_df), use_container_width=True)
    with col4:
        st.plotly_chart(fig_cluster_revenue(cluster_summary), use_container_width=True)

    st.plotly_chart(fig_cluster_peak_hour(cluster_summary), use_container_width=True)

    st.markdown("### Cluster Interpretation")
    for _, row in cluster_summary.iterrows():
        cluster_id = int(row["cluster"])
        st.write(
            f"- **Cluster {cluster_id}**: "
            f"Avg Revenue = ${row['avg_revenue']:.2f}, "
            f"Avg Peak Hour = {row['avg_peak_hour']:.1f}, "
            f"Weekend Ratio = {row['avg_weekend_ratio']:.2f}"
        )


def raw_data_page(df: pd.DataFrame):
    st.subheader("Joined Analysis Dataset")
    st.dataframe(df.head(100), use_container_width=True)
    st.write("Shape:", df.shape)


def main():
    st.title("Coffee Shop Data Mining & DSS Dashboard")

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Lỗi khi load dữ liệu: {e}")
        st.stop()

    page = st.sidebar.radio(
        "Chọn trang",
        ["Overview", "Association Rules", "Clustering", "Raw Data"]
    )

    st.sidebar.markdown("### Filter")
    if "store_location" in df.columns:
        stores = st.sidebar.multiselect(
            "Store Location",
            options=sorted(df["store_location"].dropna().unique().tolist()),
            default=sorted(df["store_location"].dropna().unique().tolist())
        )
        df = df[df["store_location"].isin(stores)]

    if "product_category" in df.columns:
        categories = st.sidebar.multiselect(
            "Product Category",
            options=sorted(df["product_category"].dropna().unique().tolist()),
            default=sorted(df["product_category"].dropna().unique().tolist())
        )
        df = df[df["product_category"].isin(categories)]

    if page == "Overview":
        overview_page(df)
    elif page == "Association Rules":
        association_page(df)
    elif page == "Clustering":
        clustering_page(df)
    else:
        raw_data_page(df)


if __name__ == "__main__":
    main()
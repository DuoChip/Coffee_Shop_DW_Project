import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def prepare_cluster_matrix(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    non_feature_cols = ["store_id", "store_location", "date_only"]
    feature_cols = [c for c in feature_df.columns if c not in non_feature_cols]

    X_df = feature_df[feature_cols].copy().fillna(0)
    meta_df = feature_df[non_feature_cols].copy()

    return meta_df, X_df


def calculate_elbow_and_silhouette(X_df: pd.DataFrame, k_min: int = 2, k_max: int = 6) -> pd.DataFrame:
    results = []

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_df)

        inertia = model.inertia_
        sil = silhouette_score(X_df, labels) if k > 1 else None

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": sil
        })

    return pd.DataFrame(results)


def run_kmeans(feature_df: pd.DataFrame, n_clusters: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, KMeans]:
    meta_df, X_df = prepare_cluster_matrix(feature_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    clustered_df = feature_df.copy()
    clustered_df["cluster"] = labels
    clustered_df["pca_1"] = X_pca[:, 0]
    clustered_df["pca_2"] = X_pca[:, 1]

    centroids = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=X_df.columns
    )
    centroids["cluster"] = centroids.index

    return clustered_df, centroids, scaler, model


def summarize_clusters(clustered_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        clustered_df.groupby("cluster", as_index=False)
        .agg(
            periods=("cluster", "size"),
            avg_revenue=("total_revenue", "mean"),
            avg_qty=("total_qty", "mean"),
            avg_transactions=("transaction_count", "mean"),
            avg_order_value=("avg_order_value", "mean"),
            avg_peak_hour=("peak_hour", "mean"),
            avg_weekend_ratio=("weekend_ratio", "mean"),
        )
        .sort_values("cluster")
    )
    return summary
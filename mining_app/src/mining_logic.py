import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def perform_association_mining(df, algorithm='FP-Growth', min_support=0.01, min_confidence=0.2):
    # Tạo giỏ hàng dựa trên tên chuẩn của tập raw
    basket = (df.groupby(['transaction_id', 'product_type'])['transaction_qty']
              .sum().unstack().reset_index().fillna(0)
              .set_index('transaction_id'))
    
    # Encode True/False cho bản mới của mlxtend
    basket_sets = basket.map(lambda x: True if x >= 1 else False)
    
    if algorithm == 'Apriori':
        frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = fpgrowth(basket_sets, min_support=min_support, use_colnames=True)
        
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules[rules['confidence'] >= min_confidence]
    return rules.sort_values('lift', ascending=False)

def prepare_cluster_features(df):
    """
    Trích xuất đặc trưng bám sát Requirements:
    sales volume, peak hour, top category, weekend ratio
    """
    # 1. Tạo hành vi theo Store + Ngày
    # Cột 'hour' và 'revenue' đã được tạo ở hàm load_data trong app.py
    store_day = df.groupby(['store_location', 'transaction_date']).agg({
        'transaction_id': 'count',      # Sales Volume (Số đơn hàng)
        'revenue': 'sum',               # Doanh thu
        'hour': lambda x: x.mode()[0] if not x.mode().empty else 0, # Peak Hour
        'is_weekend': 'first'           # Weekend/Weekday
    }).reset_index()

    # Đổi tên cho chuyên nghiệp
    store_day = store_day.rename(columns={
        'transaction_id': 'sales_volume',
        'hour': 'peak_hour'
    })

    # 2. Lấy Top Category name cho mỗi Store
    top_cats = df.groupby('store_location')['product_category'].agg(lambda x: x.mode()[0]).to_dict()
    store_day['top_category_name'] = store_day['store_location'].map(top_cats)
    
    # Label Encoding cho Category để đưa vào máy học
    le = LabelEncoder()
    store_day['top_category_encoded'] = le.fit_transform(store_day['top_category_name'])

    return store_day

def run_kmeans_analysis(feature_df, n_clusters):
    # Lựa chọn các đặc trưng số để phân cụm
    cluster_cols = ['sales_volume', 'peak_hour', 'revenue', 'is_weekend', 'top_category_encoded']
    data = feature_df[cluster_cols]
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Tính WCSS cho Elbow Method
    wcss = []
    for i in range(1, min(11, len(feature_df) + 1)):
        km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        km.fit(scaled_data)
        wcss.append(km.inertia_)
        
    # Chạy K-Means chính
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    feature_df['cluster'] = kmeans.fit_predict(scaled_data)
    
    # Tính Silhouette
    sil_score = silhouette_score(scaled_data, feature_df['cluster'])
    
    # PCA để vẽ biểu đồ scatter plot (Requirement)
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(scaled_data)
    feature_df['pca_x'] = pca_res[:, 0]
    feature_df['pca_y'] = pca_res[:, 1]
    
    return feature_df, wcss, sil_score
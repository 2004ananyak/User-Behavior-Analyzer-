#For labelling the data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("/content/cleaned_flipkart_user_dataset.csv")

# Feature columns
features_raw = [
    'avg_session_duration', 'cart_to_purchase_ratio', 'monthly_purchase_count',
    'weekend_shopper_ratio', 'discount_shopper_score',
    'category_switch_rate', 'time_on_product_page_avg',
    'products_viewed_per_session', 'return_rate'
]

# Feature engineering
df['activity_score'] = (
    df['avg_session_duration'] * 0.3 +
    df['cart_to_purchase_ratio'] * 0.2 +
    df['monthly_purchase_count'] * 0.4 +
    df['weekend_shopper_ratio'] * 0.1
)
df['engagement_score'] = df['discount_shopper_score']
features = features_raw + ['activity_score', 'engagement_score']

# Scale and reduce
scaler = StandardScaler()
scaled = scaler.fit_transform(df[features])
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled)

# KMeans clustering
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df['cluster_label'] = kmeans.fit_predict(pca_features)

# Map final user types
cluster_label_map = {
    0: "Salary Cycle Shopper",
    1: "Cart Abandoner",
    2: "Impulse Buyer",
    3: "Window Shopper",
    4: "Weekend Bulk Shopper",
    5: "Loyal High-Spender"
}
df['user_type'] = df['cluster_label'].map(cluster_label_map)

# Save the labeled dataset
df.to_csv("/content/final_labeled_flipkart_users.csv", index=False)
print("✅ Labeled dataset saved as 'final_labeled_flipkart_users.csv'")
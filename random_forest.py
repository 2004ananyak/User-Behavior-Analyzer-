#for training and testing
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the uploaded dataset
df = pd.read_csv("/content/final_labeled_flipkart_users.csv")

# Select relevant features
features = [
    'avg_session_duration',
    'cart_to_purchase_ratio',
    'monthly_purchase_count',
    'weekend_shopper_ratio',
    'discount_shopper_score',
    'category_switch_rate',
    'time_on_product_page_avg',
    'products_viewed_per_session',
    'return_rate'
]

X = df[features]
y = df['user_type']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and label encoder
joblib.dump(model, "user_type_model.pkl")
joblib.dump(le, "label_encoder.pkl")

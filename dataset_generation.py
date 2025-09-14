#10000K
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of synthetic users
n_samples = 10000

# Generate user IDs
user_ids = [f"U{100000 + i}" for i in range(n_samples)]

# Generate synthetic features
data = {
    "user_id": user_ids,
    "avg_session_duration": np.round(np.random.normal(7, 2, n_samples), 2),  # minutes
    "cart_to_purchase_ratio": np.clip(np.random.beta(2, 2, n_samples), 0, 1),
    "monthly_purchase_count": np.random.poisson(4, n_samples),
    "weekend_shopper_ratio": np.clip(np.random.beta(2, 2, n_samples), 0, 1),
    "discount_shopper_score": np.clip(np.random.beta(2.5, 1.5, n_samples), 0, 1),
    "cart_size_avg": np.round(np.random.normal(3, 1.2, n_samples), 2),
    "time_from_cart_to_purchase_avg": np.round(np.random.exponential(2, n_samples), 2),  # hours
    "days_since_last_purchase": np.random.randint(0, 60, n_samples),
    "avg_product_price": np.round(np.random.normal(500, 200, n_samples), 2),  # INR
    "category_switch_rate": np.clip(np.random.beta(1.8, 2.2, n_samples), 0, 1),
    "purchase_day_mode": np.random.randint(1, 32, n_samples),
    "is_end_of_month_buyer": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    "session_consistency_score": np.random.choice(["low", "medium", "high"], n_samples, p=[0.2, 0.5, 0.3]),
    "time_on_product_page_avg": np.round(np.random.normal(3, 1, n_samples), 2),  # minutes
    "products_viewed_per_session": np.random.poisson(5, n_samples),
    "return_rate": np.round(np.clip(np.random.normal(0.15, 0.1, n_samples), 0, 1), 2),
    "average_rating_given": np.round(np.random.uniform(1, 5, n_samples), 1),
    "number_of_reviews": np.random.poisson(2, n_samples),
    "browsing_days_per_month": np.random.randint(1, 30, n_samples),
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("synthetic_flipkart_user_dataset.csv", index=False)
print("✅ Synthetic dataset generated and saved as 'synthetic_flipkart_user_dataset.csv'")

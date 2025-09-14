#cleaning: removing negative values, converting strings to numbers
import pandas as pd

# Load your dataset
df = pd.read_csv("synthetic_flipkart_user_dataset.csv")

# Drop 'user_id' as it's not useful for modeling
df.drop(columns=["user_id"], inplace=True)

# Encode 'session_consistency_score' from categorical to numerical
df["session_consistency_score"] = df["session_consistency_score"].map({
    "low": 0, "medium": 1, "high": 2
})

# Optional: Remove rows with invalid product prices
df = df[df["avg_product_price"] > 0]

# Save the cleaned dataset
df.to_csv("cleaned_flipkart_user_dataset.csv", index=False)

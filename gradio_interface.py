import gradio as gr
import numpy as np
import pandas as pd
import joblib

# Load model, label encoder, and data
model = joblib.load("user_type_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
data = pd.read_csv("final_labeled_flipkart_users.csv")

# Define features used
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

# Compute z-score stats
zscore_means = data[features].mean()
zscore_stds = data[features].std()

# Explanations for each type
explanation_dict = {
    "Impulse Buyer": "Makes quick purchase decisions, usually triggered by discounts or urgency.",
    "Cart Abandoner": "Often adds items to the cart but doesn't buy. May need nudges or price reassurance.",
    "Loyal High-Spender": "High engagement and spending. Likely a retained, satisfied customer.",
    "Window Shopper": "Browses often but rarely buys. May be comparing prices or just exploring.",
    "Weekend Bulk Shopper": "Shops mainly during weekends and buys in bulk. May be a planner.",
    "One-time Buyer": "Limited interaction history. Possibly a new or lost customer."
}

# Prediction + anomaly detection function
def classify_user(username, avg_session_duration, cart_to_purchase_ratio,
                  monthly_purchase_count, weekend_shopper_ratio,
                  discount_shopper_score, category_switch_rate,
                  time_on_product_page_avg, products_viewed_per_session,
                  return_rate):

    if not username.strip():
        return "⚠️ Please enter a valid username.", "", "", ""

    # Prepare input
    input_array = np.array([
        avg_session_duration,
        cart_to_purchase_ratio,
        monthly_purchase_count,
        weekend_shopper_ratio,
        discount_shopper_score,
        category_switch_rate,
        time_on_product_page_avg,
        products_viewed_per_session,
        return_rate
    ])
    input_data = input_array.reshape(1, -1)

    # Predict
    prediction = model.predict(input_data)[0]
    user_type = label_encoder.inverse_transform([prediction])[0]
    insight = explanation_dict.get(user_type, "No explanation available.")

    # Anomaly Detection
    z_scores = (input_array - zscore_means.values) / zscore_stds.values
    anomaly_flags = np.abs(z_scores) > 3
    reasons = []

    if anomaly_flags.any():
        status = "🚨 Anomalous behavior detected!"
        for i, flag in enumerate(anomaly_flags):
            if flag:
                direction = "very high" if z_scores[i] > 0 else "very low"
                reasons.append(f"{features[i]} is {direction} (Z = {z_scores[i]:.2f})")
        reason_output = "\n".join(reasons)
    else:
        status = "✅ No anomaly detected. Behavior is within normal limits."
        reason_output = ""

    return (
        f"👤 Username: {username}",
        f"📊 Predicted User Type: {user_type}",
        f"🧠 Behavioral Insight: {insight}",
        f"{status}\n{reason_output}"
    )

# Gradio Blocks UI
with gr.Blocks() as demo:
    gr.Markdown("## 🛍️ Flipkart User Type Classifier + Anomaly Detection")
    gr.Markdown("Enter user behavior metrics to predict type and detect anomalies.")

    with gr.Row():
        username = gr.Textbox(label="Enter Username (e.g., Ananya, DALL)")

    with gr.Row():
        avg_session_duration = gr.Slider(0.0, 60.0, value=5.0, label="Average Session Duration")
        cart_to_purchase_ratio = gr.Slider(0.0, 1.0, value=0.5, label="Cart to Purchase Ratio")
        monthly_purchase_count = gr.Slider(0, 100, value=3, step=1, label="Monthly Purchase Count")

    with gr.Row():
        weekend_shopper_ratio = gr.Slider(0.0, 1.0, value=0.5, label="Weekend Shopper Ratio")
        discount_shopper_score = gr.Slider(0.0, 1.0, value=0.5, label="Discount Shopper Score")
        category_switch_rate = gr.Slider(0.0, 1.0, value=0.5, label="Category Switch Rate")

    with gr.Row():
        time_on_product_page_avg = gr.Slider(0.0, 100.0, value=3.0, label="Time on Product Page Avg")
        products_viewed_per_session = gr.Slider(0, 100, value=5, step=1, label="Products Viewed per Session")
        return_rate = gr.Slider(0.0, 1.0, value=0.2, label="Return Rate")

    predict_btn = gr.Button("Predict User Type")

    out_username = gr.Textbox(label="Username", interactive=False)
    out_type = gr.Textbox(label="Predicted User Type", interactive=False)
    out_insight = gr.Textbox(label="Behavioral Insight", lines=2, interactive=False)
    out_anomaly = gr.Textbox(label="Anomaly Detection", lines=5, interactive=False)

    predict_btn.click(
        classify_user,
        inputs=[
            username,
            avg_session_duration,
            cart_to_purchase_ratio,
            monthly_purchase_count,
            weekend_shopper_ratio,
            discount_shopper_score,
            category_switch_rate,
            time_on_product_page_avg,
            products_viewed_per_session,
            return_rate
        ],
        outputs=[
            out_username,
            out_type,
            out_insight,
            out_anomaly
        ]
    )

demo.launch()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time

# Download the VADER lexicon if not already
nltk.download('vader_lexicon')

# -------------------- Load Data -------------------- #
# Dummy revenue data
revenue_data = [27500, 30000, 32500, 35000, 37500]

# Dummy stock data
dates = pd.date_range(start="2024-06-01", periods=365)
stock_prices = np.cumsum(np.random.randn(365)) + 200
stock_data = pd.DataFrame({"Date": dates, "Close": stock_prices})

# Dummy Blockchain Transaction Validation
def validate_blockchain_tx():
    time.sleep(1)
    return False  # Replace with real validation later

# -------------------- Dashboard -------------------- #
st.set_page_config(page_title="Business Analytics Dashboard", layout="wide")

# Title
st.markdown("## 📊 **Business Analytics Dashboard**")

# -------------------- Revenue Forecast -------------------- #
st.subheader("Revenue Forecast")

fig_revenue = go.Figure()
fig_revenue.add_trace(go.Scatter(y=revenue_data, mode="lines+markers", name="Revenue Forecast"))
fig_revenue.update_layout(title="Predicted Revenue", xaxis_title="Quarter", yaxis_title="Revenue")
st.plotly_chart(fig_revenue)

# Display predicted revenue table
revenue_df = pd.DataFrame(revenue_data, columns=["value"])
st.markdown("📈 **Predicted Revenue:**")
st.dataframe(revenue_df)

# -------------------- Stock Market Forecast -------------------- #
st.subheader("Stock Market Forecast")

fig_stock = go.Figure()
fig_stock.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Close"], mode="lines", name="Stock Close"))
fig_stock.update_layout(title="Stock Forecast", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_stock)

# Dummy prediction for next 30 days
future_prices = np.round(np.linspace(stock_prices[-1], stock_prices[-1] + 1, 30), 4)
future_df = pd.DataFrame(future_prices, columns=["value"])
st.markdown("📉 **Next 30-Day Prediction:**")
st.dataframe(future_df)

# -------------------- Sentiment Analysis -------------------- #
st.subheader("📝 Review Sentiment Score")

review = st.text_area("Enter your product or business review:", height=100)

if st.button("Analyze Sentiment"):
    if review.strip() != "":
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(review)
        st.json(sentiment)
    else:
        st.warning("⚠️ Please enter some review text.")

# -------------------- Blockchain Connection -------------------- #
st.subheader("🔗 Blockchain Transaction Validation")

if st.button("Validate Blockchain Connection"):
    connected = validate_blockchain_tx()
    if connected:
        st.success("✅ Connected to Blockchain successfully.")
    else:
        st.error("❌ Connected to Blockchain: False")

import nltk
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from web3 import Web3

# Load Business Data
df = pd.read_csv("data/sales.csv")

# Train Sales Forecast Model
model = LinearRegression()
model.fit(df[["Month"]], df["Revenue"])
future_months = [[i] for i in range(len(df)+1, len(df)+6)]
predictions = model.predict(future_months)

# Stock Market Prediction
stock = yf.Ticker("AAPL")
data = stock.history(period="1y")
X = data.index.factorize()[0].reshape(-1, 1)
y = data["Close"]
stock_model = LinearRegression().fit(X, y)
future_days = [[len(X)+i] for i in range(30)]
stock_predictions = stock_model.predict(future_days)

# Sentiment Analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
review = "Great business performance!"
sentiment = sia.polarity_scores(review)

# Blockchain Setup
infura_url = "https://ropsten.infura.io/v3/YOUR_API_KEY"
web3 = Web3(Web3.HTTPProvider(infura_url))

# Streamlit Dashboard
st.title("📊 Business Analytics Dashboard")
st.header("Revenue Forecast")
st.line_chart(df["Revenue"])
st.write("📈 Predicted Revenue:", predictions)

st.header("Stock Market Forecast")
st.line_chart(data["Close"])
st.write("📉 Next 30-Day Prediction:", stock_predictions)

st.header("Sentiment Analysis")
st.write("📝 Review Sentiment Score:", sentiment)

st.header("Blockchain Transaction Validation")
st.write("🔗 Connected to Blockchain:", web3.is_connected())

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

def prediction(stock_ticker, days):
    # Fetch data from Yahoo Finance
    df = yf.download(stock_ticker, period="1y")

    # Check if data is empty
    if df.empty:
        raise ValueError(f"No stock data found for {stock_ticker}. Please check the stock ticker.")

    # Prepare data
    df.reset_index(inplace=True)
    df["Days"] = np.arange(len(df))

    X = df[["Days"]].values
    y = df["Close"].values

    # Ensure there are enough samples for train-test split
    if len(X) < 2:
        raise ValueError("Not enough data points available for prediction.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train SVR model
    model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train)

    # Predict future stock prices
    future_days = np.array([X.max() + i for i in range(1, days + 1)]).reshape(-1, 1)
    future_preds = model.predict(future_days)

    # Create forecast plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=pd.date_range(start=df["Date"].iloc[-1], periods=days, freq="D"),
                             y=future_preds, mode="lines", name="Predicted", line=dict(dash="dash")))

    fig.update_layout(title="Stock Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      template="plotly_dark")

    return fig

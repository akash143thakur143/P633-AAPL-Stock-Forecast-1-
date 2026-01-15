import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------------------
# Page Settings
# -----------------------------------------
st.set_page_config(page_title="Apple Stock Forecast", layout="wide")

st.title("🍎 Apple Stock Forecast Dashboard (Next 30 Days)")
st.markdown("### Project: P-633 Apple Stock Forecast | Target: Adj Close")

# -----------------------------------------
# Load Dataset
# -----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AAPL (5).csv")   # Make sure this file is in GitHub repo
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.reset_index(drop=True, inplace=True)
    return df

try:
    df = load_data()
except Exception as e:
    st.error("⚠️ Dataset file not found. Please make sure `AAPL (5).csv` is present in the GitHub repository.")
    st.stop()

# -----------------------------------------
# Sidebar Options
# -----------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")

days_to_forecast = st.sidebar.slider("📅 Forecast Days", 7, 60, 30)

# -----------------------------------------
# Feature Engineering
# -----------------------------------------
df["MA50"] = df["Adj Close"].rolling(50).mean()
df["MA200"] = df["Adj Close"].rolling(200).mean()
df["Daily_Return"] = df["Adj Close"].pct_change()

ts = df[["Date", "Adj Close"]].set_index("Date")

# -----------------------------------------
# Section: Overview
# -----------------------------------------
st.subheader("📌 Dataset Overview")
st.dataframe(df.head(10), use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Total Rows", df.shape[0])
c2.metric("Total Columns", df.shape[1])
c3.metric("Missing Values", int(df.isnull().sum().sum()))

# -----------------------------------------
# Plot 1: Adj Close Trend
# -----------------------------------------
st.subheader("📈 Adjusted Close Trend")

fig1 = plt.figure(figsize=(12,5))
plt.plot(df["Date"], df["Adj Close"], label="Adj Close")
plt.title("Apple Adj Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Adj Close Price")
plt.grid(True)
plt.legend()
st.pyplot(fig1)

# -----------------------------------------
# Plot 2: Moving Averages
# -----------------------------------------
st.subheader("📉 Moving Average Trend (MA50 vs MA200)")

fig2 = plt.figure(figsize=(12,5))
plt.plot(df["Date"], df["Adj Close"], label="Adj Close")
plt.plot(df["Date"], df["MA50"], label="MA50")
plt.plot(df["Date"], df["MA200"], label="MA200")
plt.title("Adj Close with MA50 and MA200")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
st.pyplot(fig2)

# -----------------------------------------
# Plot 3: Daily Returns
# -----------------------------------------
st.subheader("📊 Daily Returns")

fig3 = plt.figure(figsize=(12,4))
plt.plot(df["Date"], df["Daily_Return"])
plt.title("Daily Returns Over Time")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.grid(True)
st.pyplot(fig3)

# -----------------------------------------
# ARIMA Model Training
# -----------------------------------------
st.subheader("🤖 ARIMA Forecasting & Evaluation")

if len(ts) < 100:
    st.warning("⚠️ Dataset is too small for ARIMA training and evaluation.")
    st.stop()

train = ts.iloc[:-60]
test = ts.iloc[-60:]

model = ARIMA(train["Adj Close"], order=(5, 1, 0))
model_fit = model.fit()

forecast_test = model_fit.forecast(steps=len(test))

rmse = np.sqrt(mean_squared_error(test["Adj Close"], forecast_test))
mae = mean_absolute_error(test["Adj Close"], forecast_test)
mape = np.mean(np.abs((test["Adj Close"].values - forecast_test.values) / test["Adj Close"].values)) * 100

m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{rmse:.2f}")
m2.metric("MAE", f"{mae:.2f}")
m3.metric("MAPE (%)", f"{mape:.2f}")

# Plot forecast vs actual
fig4 = plt.figure(figsize=(12,5))
plt.plot(train.index, train["Adj Close"], label="Train")
plt.plot(test.index, test["Adj Close"], label="Test Actual")
plt.plot(test.index, forecast_test, label="ARIMA Forecast", linestyle="--")
plt.title("ARIMA Forecast vs Actual (Test Set)")
plt.xlabel("Date")
plt.ylabel("Adj Close")
plt.grid(True)
plt.legend()
st.pyplot(fig4)

# -----------------------------------------
# Future Forecasting
# -----------------------------------------
st.subheader(f"📅 Future Forecast (Next {days_to_forecast} Days)")

final_model = ARIMA(ts["Adj Close"], order=(5, 1, 0))
final_fit = final_model.fit()

future_forecast = final_fit.forecast(steps=days_to_forecast)

future_dates = pd.date_range(
    start=ts.index[-1] + pd.Timedelta(days=1),
    periods=days_to_forecast
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast_Adj_Close": future_forecast.values
})

fig5 = plt.figure(figsize=(12,5))
plt.plot(ts.index, ts["Adj Close"], label="Historical")
plt.plot(forecast_df["Date"], forecast_df["Forecast_Adj_Close"], label="Forecast", linestyle="--")
plt.title("Apple Stock Forecast (Future)")
plt.xlabel("Date")
plt.ylabel("Adj Close")
plt.grid(True)
plt.legend()
st.pyplot(fig5)

st.markdown("### 📌 Forecast Table")
st.dataframe(forecast_df, use_container_width=True)

# Download button
csv_data = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Forecast CSV",
    data=csv_data,
    file_name="AAPL_30days_forecast.csv",
    mime="text/csv"
)

st.success("✅ Dashboard loaded successfully!")

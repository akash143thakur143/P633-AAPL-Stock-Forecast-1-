import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Apple Stock Forecast Dashboard", layout="wide")

st.title("📈 Apple Stock Forecast Dashboard (Next 30 Days)")
st.markdown("**Project:** P-633 Apple Stock Forecast | **Target:** Adj Close Forecasting")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    df.reset_index(drop=True, inplace=True)
    return df

file_path = "AAPL (5).csv"
df = load_data(file_path)

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("⚙️ Controls")
days_to_forecast = st.sidebar.slider("Select Forecast Days", 7, 60, 30)

# -------------------------------
# Feature Engineering
# -------------------------------
df['MA50'] = df['Adj Close'].rolling(50).mean()
df['MA200'] = df['Adj Close'].rolling(200).mean()
df['Daily_Return'] = df['Adj Close'].pct_change()

ts = df[['Date', 'Adj Close']].set_index('Date')

# -------------------------------
# Section 1: Dataset Preview
# -------------------------------
st.subheader("📌 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

# -------------------------------
# Section 2: Adj Close Trend
# -------------------------------
st.subheader("📊 Adj Close Price Trend")

fig1 = plt.figure(figsize=(12,5))
plt.plot(df['Date'], df['Adj Close'], label="Adj Close")
plt.title("Apple Adjusted Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Adj Close")
plt.grid(True)
st.pyplot(fig1)

# -------------------------------
# Section 3: Moving Averages
# -------------------------------
st.subheader("📉 Moving Average Trend (MA50 vs MA200)")

fig2 = plt.figure(figsize=(12,5))
plt.plot(df['Date'], df['Adj Close'], label="Adj Close")
plt.plot(df['Date'], df['MA50'], label="MA50")
plt.plot(df['Date'], df['MA200'], label="MA200")
plt.title("Adj Close with MA50 and MA200")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
st.pyplot(fig2)

# -------------------------------
# Section 4: Daily Returns
# -------------------------------
st.subheader("📈 Daily Returns")

fig3 = plt.figure(figsize=(12,4))
plt.plot(df['Date'], df['Daily_Return'])
plt.title("Daily Returns Over Time")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.grid(True)
st.pyplot(fig3)

# -------------------------------
# Section 5: ARIMA Forecasting
# -------------------------------
st.subheader("🤖 Forecasting using ARIMA")

# Train-test split (last 60 days for test)
train = ts.iloc[:-60]
test = ts.iloc[-60:]

model = ARIMA(train['Adj Close'], order=(5,1,0))
model_fit = model.fit()

forecast_test = model_fit.forecast(steps=len(test))

# Evaluation metrics
rmse = np.sqrt(mean_squared_error(test['Adj Close'], forecast_test))
mae = mean_absolute_error(test['Adj Close'], forecast_test)
mape = np.mean(np.abs((test['Adj Close'].values - forecast_test.values) / test['Adj Close'].values)) * 100

st.markdown("### ✅ Model Evaluation Metrics")
m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{rmse:.2f}")
m2.metric("MAE", f"{mae:.2f}")
m3.metric("MAPE (%)", f"{mape:.2f}")

# Plot Forecast vs Actual
fig4 = plt.figure(figsize=(12,5))
plt.plot(train.index, train['Adj Close'], label="Train")
plt.plot(test.index, test['Adj Close'], label="Actual Test")
plt.plot(test.index, forecast_test, label="Forecast", linestyle="--")
plt.title("ARIMA Forecast vs Actual")
plt.legend()
plt.grid(True)
st.pyplot(fig4)

# -------------------------------
# Section 6: Future Forecast
# -------------------------------
st.subheader(f"📅 Future Forecast (Next {days_to_forecast} Days)")

final_model = ARIMA(ts['Adj Close'], order=(5,1,0))
final_fit = final_model.fit()

future_forecast = final_fit.forecast(steps=days_to_forecast)

future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=days_to_forecast)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast_Adj_Close": future_forecast.values
})

fig5 = plt.figure(figsize=(12,5))
plt.plot(ts.index, ts['Adj Close'], label="Historical Adj Close")
plt.plot(forecast_df['Date'], forecast_df['Forecast_Adj_Close'], label="Forecast", linestyle="--")
plt.title(f"Apple Stock Forecast - Next {days_to_forecast} Days")
plt.xlabel("Date")
plt.ylabel("Adj Close")
plt.legend()
plt.grid(True)
st.pyplot(fig5)

st.markdown("### 📌 Forecast Table")
st.dataframe(forecast_df, use_container_width=True)

# Download button
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Forecast CSV", csv, "AAPL_forecast.csv", "text/csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import pickle
import os

# Function to fetch weather data (dummy implementation)
def fetch_weather_data(start_date, end_date):
    # Replace this with actual code to fetch weather data
    dates = pd.date_range(start=start_date, end=end_date)
    weather_data = np.random.rand(len(dates))  # Dummy weather data
    return pd.DataFrame({'Date': dates, 'Weather': weather_data}).set_index('Date')

# Save the training dataset
def save_training_data(data, filename='training_data.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# Load the training dataset
def load_training_data(filename='training_data.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# Step 1: Fetch Stock Data from Google Finance
ticker = "REPCOHOME.NS"
start_date = "2023-01-01"
end_date = "2025-01-02"

# Download stock data using yfinance (Google Finance integrated support)
df = yf.Ticker(ticker).history(start=start_date, end=end_date)

# Ensure the datetime index is timezone-naive
df.index = df.index.tz_localize(None)

# Focus only on the 'Close' prices
df = df[['Close']]

# Fetch weather data
weather_df = fetch_weather_data(start_date, end_date)

# Ensure the datetime index is timezone-naive
weather_df.index = weather_df.index.tz_localize(None)

# Merge stock data with weather data
df = df.merge(weather_df, left_index=True, right_index=True)


# Step 2: Normalize Data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Split data into training (80%) and testing datasets (20%)
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Prepare the training data
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i])
    y_train.append(train_data[i, 0])  # Predicting 'Close' price

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

# Step 3: Define and Train the LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=1000, batch_size=32)  # Increased epochs for better training

# Step 4: Predict and Evaluate Stock Prices using LSTM
X_test, y_test = [], test_data[60:, 0]
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

predictions_lstm = model.predict(X_test)
predictions_lstm = scaler.inverse_transform(np.concatenate((predictions_lstm, np.zeros((predictions_lstm.shape[0], df.shape[1] - 1))), axis=1))[:, 0]

# Visualizing LSTM Predictions vs. Actual
plt.figure(figsize=(14, 5))
plt.plot(df.index[-len(predictions_lstm):], df['Close'][-len(predictions_lstm):], label="True Prices")
plt.plot(df.index[-len(predictions_lstm):], predictions_lstm, label="LSTM Predicted Prices", color="r")
plt.title(f"{ticker} - LSTM Predictions")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc="best")
plt.show()

# Step 5: Train and Predict using ARIMA
train_arima = df['Close'][:int(len(df) * 0.8)]
test_arima = df['Close'][int(len(df) * 0.8):]

# Fit ARIMA model
arima_model = ARIMA(train_arima, order=(5, 1, 0))
arima_model_fit = arima_model.fit()

# Forecast
predictions_arima = arima_model_fit.forecast(steps=len(test_arima))

# Visualizing ARIMA Predictions vs. Actual
plt.figure(figsize=(14, 5))
plt.plot(test_arima.values, label="True Prices")
plt.plot(predictions_arima, label="ARIMA Predicted Prices", color="g")
plt.title(f"{ticker} - ARIMA Predictions")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc="best")
plt.show()

# Print mean predictions for comparison
mean_prediction_lstm = np.mean(predictions_lstm)
mean_prediction_arima = np.mean(predictions_arima)
print(f"Predicted average stock price for this month (LSTM): {mean_prediction_lstm}")
print(f"Predicted average stock price for this month (ARIMA): {mean_prediction_arima}")

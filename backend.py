import requests
import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Constants
start_date = "2023-01-01"
end_date = "2025-01-01"
tickers = ["AAPL", "GOOGL", "AMZN", "MSFT"]
CITIES = {
    "New York": {"lat": 40.7128, "lon": -74.0060},
    "London": {"lat": 51.5072, "lon": -0.1276},
    "Tokyo": {"lat": 35.6764, "lon": 139.6500},
}

# Fetch historical stock prices
for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns.values]
    stock_data.rename(columns={'Date_': 'Datetime'}, inplace=True)
    stock_data["Datetime"] = pd.to_datetime(stock_data["Datetime"]).dt.tz_localize(None)
    file_name = f"{ticker}.csv"
    stock_data.to_csv(file_name, index=False)
    print(f"Saved {ticker} data to {file_name}")

# Fetch historical weather conditions
def get_weather_data(city, lat, lon, start_date, end_date):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,precipitation&timezone=GMT"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "hourly" in data and "time" in data["hourly"]:
            timestamps = pd.to_datetime(data["hourly"]["time"], errors="coerce", utc=True)
            temperatures = data["hourly"]["temperature_2m"]
            precipitation = data["hourly"].get("precipitation", [0] * len(timestamps))
            temp_df = pd.DataFrame({"Datetime": timestamps, "Temperature (°C)": temperatures})
            rain_df = pd.DataFrame({"Datetime": timestamps, "Rainfall (mm)": precipitation})
            temp_df["Datetime"] = pd.to_datetime(temp_df["Datetime"]).dt.tz_localize(None)
            rain_df["Datetime"] = pd.to_datetime(rain_df["Datetime"]).dt.tz_localize(None)
            temp_filename = f"{city.replace(' ', '_').lower()}_temperature.csv"
            rain_filename = f"{city.replace(' ', '_').lower()}_rainfall.csv"
            temp_df.to_csv(temp_filename, index=False)
            rain_df.to_csv(rain_filename, index=False)
            print(f"Data saved: {temp_filename}, {rain_filename}")
            return temp_df, rain_df
        else:
            print("No hourly data found in the response.")
            return None, None
    else:
        print(f"Failed to fetch weather data! Error {response.status_code}")
        return None, None

for city, coords in CITIES.items():
    get_weather_data(city, coords["lat"], coords["lon"], start_date, end_date)

# Merge stock and weather data
merged_data_dict = {}
for ticker in tickers:
    stock_data = pd.read_csv(f"{ticker}.csv")
    stock_data["Datetime"] = pd.to_datetime(stock_data["Datetime"])
    for city in CITIES:
        temp_df = pd.read_csv(f"{city.replace(' ', '_').lower()}_temperature.csv")
        rain_df = pd.read_csv(f"{city.replace(' ', '_').lower()}_rainfall.csv")
        temp_df["Datetime"] = pd.to_datetime(temp_df["Datetime"])
        rain_df["Datetime"] = pd.to_datetime(rain_df["Datetime"])
        merged_data_temp = pd.merge_asof(stock_data.sort_values('Datetime'), temp_df.sort_values('Datetime'), on='Datetime', direction='nearest')
        merged_data_rain = pd.merge_asof(stock_data.sort_values('Datetime'), rain_df.sort_values('Datetime'), on='Datetime', direction='nearest')
        merged_data_temp.dropna(inplace=True)
        merged_data_rain.dropna(inplace=True)
        merged_data_dict[(ticker, city, "temp")] = merged_data_temp
        merged_data_dict[(ticker, city, "rain")] = merged_data_rain
        merged_data_temp.to_csv(f"{ticker}_{city}_merged_temp.csv", index=False)
        merged_data_rain.to_csv(f"{ticker}_{city}_merged_rain.csv", index=False)

# Split datetime into key parts
for (ticker, city, data_type), merged_data in merged_data_dict.items():
    merged_data["DayOfWeek"] = merged_data["Datetime"].dt.dayofweek
    merged_data["DayOfMonth"] = merged_data["Datetime"].dt.day
    merged_data["Month"] = merged_data["Datetime"].dt.month
    merged_data["Year"] = merged_data["Datetime"].dt.year
    merged_data.drop(columns=["Datetime"], inplace=True)
    merged_data.to_csv(f"{ticker}_{city}_merged_{data_type}.csv", index=False)

# Normalize data
train_size = int(len(merged_data_temp) * 0.8)
train_temp, test_temp = merged_data_temp[:train_size], merged_data_temp[train_size:]
train_rain, test_rain = merged_data_rain[:train_size], merged_data_rain[train_size:]

close_columns = [col for col in merged_data_temp.columns if "Close" in col]
if not close_columns:
    raise KeyError("No 'Close' column found in merged_data_temp. Check the column names.")
else:
    close_column = close_columns[0]
    print(f"Using column '{close_column}' as Close price for training.")
    close_index = merged_data_temp.columns.get_loc(close_column)
    print(f"close_index is: {close_index}")

stock_columns = close_columns
weather_columns_temp = [col for col in merged_data_temp.columns if col not in stock_columns]
weather_columns_rain = [col for col in merged_data_rain.columns if col not in stock_columns]

scaler_stock = MinMaxScaler()
scaler_temp = MinMaxScaler()
scaler_rain = MinMaxScaler()

train_temp_scaled = scaler_temp.fit_transform(train_temp[weather_columns_temp])
test_temp_scaled = scaler_temp.transform(test_temp[weather_columns_temp])
train_rain_scaled = scaler_rain.fit_transform(train_rain[weather_columns_rain])
test_rain_scaled = scaler_rain.transform(test_rain[weather_columns_rain])
train_stock_scaled = scaler_stock.fit_transform(train_temp[stock_columns])
test_stock_scaled = scaler_stock.transform(test_temp[stock_columns])

train_temp_scaled = np.hstack([train_stock_scaled, train_temp_scaled])
test_temp_scaled = np.hstack([test_stock_scaled, test_temp_scaled])
train_rain_scaled = np.hstack([train_stock_scaled, train_rain_scaled])
test_rain_scaled = np.hstack([test_stock_scaled, test_rain_scaled])

print("Normalization completed: Stock prices and weather data are scaled separately.")

# Create sequences for LSTM
def create_sequences(data, lookback=30, close_index=0, num_features=None, future_days=10):
    if num_features is None or num_features <= 0:
        raise ValueError("num_features must be greater than 0!")
    X, y = [], []
    for i in range(len(data) - lookback - future_days):
        X.append(data[i:i + lookback, 1:num_features])
        y.append(data[i + lookback:i + lookback + future_days, close_index])
    return np.array(X), np.array(y)

num_features = len(weather_columns_temp) + len(stock_columns)
print(f"Number of features used: {num_features}")

X_temp, y_temp = create_sequences(train_temp_scaled, close_index=close_index, num_features=num_features)
X_rain, y_rain = create_sequences(train_rain_scaled, close_index=close_index, num_features=num_features)
print(f"Sequences created: X_temp shape = {X_temp.shape}, X_rain shape = {X_rain.shape}")

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(10)  # Predict stock 'Close' price
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train TEMP Model
print(f"Shape of X_temp: {X_temp.shape}")
print(f"Shape of y_temp: {y_temp.shape}")
input_shape = (X_temp.shape[1], X_temp.shape[2])
model_temp = build_lstm_model(input_shape)
history_temp = model_temp.fit(X_temp, y_temp, epochs=50, batch_size=32, validation_split=0.2)

# Plot training & validation loss values
plt.plot(history_temp.history['loss'], label='Train Loss')
plt.plot(history_temp.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Train RAIN Model
model_rain = build_lstm_model((X_rain.shape[1], X_rain.shape[2]))
history_rain = model_rain.fit(X_rain, y_rain, epochs=50, batch_size=32, validation_split=0.2)

# Plot training & validation loss values
plt.plot(history_rain.history['loss'], label='Train Loss')
plt.plot(history_rain.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Predict next day stock price
def predict_next_day(model, last_30_days, scaler, num_features):
     # Ensure the input data has the correct shape
    if last_30_days.shape != (30, num_features):
        raise ValueError(f"Input data must have shape (30, {num_features}), but got {last_30_days.shape}")
    
     # Reshape the data for the LSTM model
    last_30_days_scaled = scaler.transform(last_30_days)  # Scale the data
    last_30_days_scaled = last_30_days_scaled.reshape(1, 30, num_features)  # Reshape to (1, 30, num_features)

    predicted_scaled = model.predict(last_30_days_scaled)

    if predicted_scaled.shape[1] != num_features:
        print(f"Warning: Model output has {predicted_scaled.shape[1]} features, but scaler expects {num_features}. Trimming extra features.")
        predicted_scaled = predicted_scaled[:, :num_features]  # Trim extra features


    predicted_price = scaler.inverse_transform(predicted_scaled.reshape(1, -1))
    return predicted_price.flatten()

train_temp = train_temp[scaler_temp.feature_names_in_]
# Corrected call to predict_next_day
predicted_price = predict_next_day(model_temp, train_temp[-30:], scaler_temp, num_features=9)  # Use the correct number of features
print(f"Predicted Next Day Stock Price: {predicted_price}")

# Model Prediction & Plot
merged_data_temp = merged_data_temp[scaler_temp.feature_names_in_]
temp_scaled = scaler_temp.transform(merged_data_temp)
timesteps = 30
num_features = temp_scaled.shape[1]

if temp_scaled.shape[0] >= timesteps:
    # Prepare the input data for prediction
    X_test = []
    for i in range(timesteps, len(temp_scaled)):
        X_test.append(temp_scaled[i - timesteps:i, :])
    X_test = np.array(X_test)
    predictions = model_temp.predict(X_test)

    # Ensure the predicted output has the correct number of features
    if predictions.shape[1] != num_features:
        print(f"Warning: Model output has {predictions.shape[1]} features, but scaler expects {num_features}. Trimming extra features.")
        predictions = predictions[:, :num_features]  # Trim extra features

        full_predictions = np.zeros((len(predictions), merged_data_temp.shape[1]))
        full_predictions[:, 0] = predictions[:, 0]
        predicted_prices = scaler_temp.inverse_transform(full_predictions)[:, 0]
        # Extract the actual prices (target values) from y_temp
        actual_prices = scaler_temp.inverse_transform(temp_scaled[timesteps:, :])[:, 0]
        plt.figure(figsize=(10, 5))
        plt.plot(actual_prices, label="Actual Prices", color="blue")
        plt.plot(predicted_prices, label="Predicted Prices", color="red")
        plt.legend()
        plt.title("Stock Price Prediction vs Actual")
        plt.show()
        
    else:
        print(f"ERROR: Not enough data. Need at least {timesteps} rows!")
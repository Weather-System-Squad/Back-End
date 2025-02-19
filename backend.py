import requests
import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

start_date = "2023-01-01"
end_date = "2025-01-01"
tickers = ["AAPL", "GOOGL", "AMZN", "MSFT"]

CITIES = {
    "New York": {"lat": 40.7128, "lon": -74.0060},
    "London": {"lat": 51.5072, "lon": -0.1276},
    "Tokyo": {"lat": 35.6764, "lon": 139.6500},
}
#getting historical stock prices
for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data.reset_index(inplace=True)
        stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns.values]
        stock_data.rename(columns={'Date_': 'Datetime'}, inplace=True)
        file_name = f"{ticker}.csv"
        stock_data.to_csv(file_name)
        print(f"Saved {ticker} data to {file_name}")

#getting historical weather conditions
def get_weather_data(city, lat, lon, start_date, end_date):

    #API URL for historical weather data
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,precipitation&timezone=GMT"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        if "hourly" in data and "time" in data["hourly"]:
            timestamps = pd.to_datetime(data["hourly"]["time"], errors="coerce", utc=True)
            temperatures = data["hourly"]["temperature_2m"]
            precipitation = data["hourly"].get("precipitation", [0] * len(timestamps))

            # Create temperature DataFrame
            temp_df = pd.DataFrame({"Datetime": timestamps, "Temperature (°C)": temperatures})
            temp_df["Datetime"] = pd.to_datetime(temp_df["Datetime"], utc=True)  # Ensure UTC timezone

            # Create rainfall DataFrame
            rain_df = pd.DataFrame({"Datetime": timestamps, "Rainfall (mm)": precipitation})
            rain_df["Datetime"] = pd.to_datetime(rain_df["Datetime"], utc=True)  # Ensure UTC timezone

            # Save to CSV
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
        print(f" Failed to fetch weather data! Error {response.status_code}")
        return None, None

    
for city, coords in CITIES.items():
    get_weather_data(city, coords["lat"], coords["lon"], start_date, end_date)

#merging stock + weather data

for ticker in tickers:
    stock_data = pd.read_csv(f"{ticker}.csv")
    stock_data["Datetime"] = pd.to_datetime(stock_data["Datetime"])
    
    for city in CITIES:
        temp_df = pd.read_csv(f"{city.replace(',', '_').lower()}_temp.csv")
        rain_df = pd.read_csv(f"{city.replace(',', '_').lower()}_rainfall.csv")

        temp_df["Datetime"] = pd.to_datetime(temp_df["Datetime"])
        rain_df["Datetime"] = pd.to_datetime(rain_df["Datetime"])

        merged_data_temp = pd.merge_asof(stock_data.sort_values('Datetime'), temp_df.sort_values('Datetime'), on='Datetime', direction='nearest')
        merged_data_rain = pd.merge_asof(stock_data.sort_values('Datetime'), rain_df.sort_values('Datetime'), on='Datetime', direction='nearest')

        merged_data_temp.dropna(inplace=True)
        merged_data_rain.dropna(inplace=True)

        merged_data_temp.to_csv(f"{ticker}_{city}_merged_temp.csv", index=False)
        merged_data_rain.to_csv(f"{ticker}_{city}_merged_rain.csv", index=False)


#splitting datetime to key parts allowing the model to capture potential temporal patterns
merged_data_temp['DayOfWeek'] = merged_data_temp['Datetime'].dt.dayofweek
merged_data_temp['DayOfMonth'] = merged_data_temp['Datetime'].dt.day
merged_data_temp['Month'] = merged_data_temp['Datetime'].dt.month
merged_data_temp['Year'] = merged_data_temp['Datetime'].dt.year
merged_data_temp = merged_data_temp.drop(columns=['Datetime'])


merged_data_rain['DayOfWeek'] = merged_data_rain['Datetime'].dt.dayofweek
merged_data_rain['DayOfMonth'] = merged_data_rain['Datetime'].dt.day
merged_data_rain['Month'] = merged_data_rain['Datetime'].dt.month
merged_data_rain['Year'] = merged_data_rain['Datetime'].dt.year
merged_data_rain = merged_data_rain.drop(columns=['Datetime'])

#NORMALISING DATA before feeding model

train_size = int(len(merged_data_temp) * 0.8)
train_temp, test_temp = merged_data_temp[:train_size], merged_data_temp[train_size:]
train_rain, test_rain = merged_data_rain[:train_size], merged_data_rain[train_size:]

num_features = train_temp.shape[1] - 1  # Exclude 'Close' column from feature set
print(f"✅ Number of features: {num_features}")  # Debugging step

#USING past 30 days of Temperature & Rainfall to predict next day's stock price.
def create_sequences(data, lookback=30, close_index=0,num_features=None):

    if num_features is None or num_features <= 0:
        raise ValueError(" num_features must be greater than 0!")

    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback, 1:num_features + 1])  # Ensure it matches input_shape])  # Use all columns except 'Close'
        y.append(data[i + lookback, close_index])  # Get Close price
    return np.array(X), np.array(y)

close_columns = [col for col in merged_data_temp.columns if "Close" in col]
if not close_columns:
    raise KeyError("No 'Close' column found in merged_data_temp. Check the column names.")
else:
    close_index = list(merged_data_temp.columns).index(close_columns[0])  # Use the first 'Close' column
    print(f"Using column '{close_columns[0]}' as Close price for training.")

scaler_temp = MinMaxScaler()
scaler_rain = MinMaxScaler()

train_temp_scaled = scaler_temp.fit_transform(train_temp)
test_temp_scaled = scaler_temp.transform(test_temp)

train_rain_scaled = scaler_rain.fit_transform(train_rain)
test_rain_scaled = scaler_rain.transform(test_rain)    


# Create sequences ensuring shape matches LSTM input requirements
num_features = train_temp.shape[1] - 1  # Ensure it's correctly set

X_temp, y_temp = create_sequences(train_temp_scaled, close_index=close_index, num_features=num_features)
X_rain, y_rain = create_sequences(train_rain_scaled, close_index=close_index, num_features=num_features)


#implementing LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)  # Predict stock 'Close' price
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#Training TEMP Model:
print(f"Shape of X_temp: {X_temp.shape}")
print(f"Shape of y_temp: {y_temp.shape}")
input_shape = (X_temp.shape[1], X_temp.shape[2])  # (timesteps, num_features)
model_temp = build_lstm_model((X_temp.shape[1], X_temp.shape[2]))
history_temp = model_temp.fit(X_temp, y_temp, epochs=50, batch_size=32, validation_split=0.2)

# # Plot training & validation loss values for a clear visualization of how the model's performance evolves during training.
plt.plot(history_temp.history['loss'], label='Train Loss')
plt.plot(history_temp.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
#plt.show()

#Training RAIN Model:
model_rain = build_lstm_model((X_rain.shape[1], X_rain.shape[2]))
history_rain=model_rain.fit(X_rain, y_rain, epochs=50, batch_size=32, validation_split=0.2)

# Plot training & validation loss values for a clear visualization of how the model's performance evolves during training.
plt.plot(history_rain.history['loss'], label='Train Loss')
plt.plot(history_rain.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# forecasting Future stock prices with TRAIN MODEL 
def predict_next_day(model, last_30_days, scaler,num_features):

    if last_30_days.shape[0] != 30:
        print(f" ERROR: Expected 30 rows, but got {last_30_days.shape[0]}.")
        return None

    last_30_days_scaled = scaler.transform(last_30_days)
    print(f"🔍 Shape of last_30_days_scaled BEFORE slicing: {last_30_days_scaled.shape}")

    last_30_days_scaled = last_30_days_scaled[:, 1:num_features + 1]# Exclude the 'Close' column if it's the first column
    print(f"🔍 Shape of last_30_days_scaled AFTER slicing: {last_30_days_scaled.shape}")

    last_30_days_scaled = np.reshape(last_30_days_scaled, (1, 30, num_features))
    print(f"🔍 Shape of last_30_days_scaled AFTER slicing: {last_30_days_scaled.shape}")

    predicted_scaled = model.predict(last_30_days_scaled)

    predicted = np.zeros((1, scaler.n_features_in_))  # Empty array
    predicted[:, 0] = predicted_scaled  # Only modify the 'Close' column

    return scaler.inverse_transform(predicted)[0, 0]  # Return predicted closing price

predicted_price = predict_next_day(model_temp, train_temp[-30:], scaler_temp, num_features)
print(f"Predicted Next Day Stock Price: {predicted_price}")

# Model Prediction & Plot
temp_scaled = scaler_temp.transform(merged_data_temp)
timesteps = 30  # Ensure this is the same as training
num_features = temp_scaled.shape[1]

if temp_scaled.shape[0] >= timesteps:
    temp_scaled = temp_scaled[-timesteps:]  # Keep last 30 rows
    temp_scaled = np.reshape(temp_scaled, (1, timesteps, num_features))  # Correct shape
    predictions = model_temp.predict(temp_scaled)
else:
    print(f"❌ ERROR: Not enough data. Need at least {timesteps} rows!")
   
if predictions is not None:
    # Convert back to original scale
    full_predictions = np.zeros((len(predictions), merged_data_temp.shape[1]))
    full_predictions[:, 0] = predictions[:, 0]
    predicted_prices = scaler_temp.inverse_transform(full_predictions)[:, 0]

    # Extract actual stock prices
    actual_prices = scaler_temp.inverse_transform(np.c_[y_temp, np.zeros((len(y_temp), temp_scaled.shape[1] - 1))])[:, 0]

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 5))
    plt.plot(actual_prices, label="Actual Prices", color="blue")
    plt.plot(predicted_prices[:, 0], label="Predicted Prices", color="red")
    plt.legend()
    plt.title("Stock Price Prediction vs Actual")
    plt.show()
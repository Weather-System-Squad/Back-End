import yfinance as yf
import requests
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#getting historical stock prices

start_date = "2023-01-01"
end_date = "2025-01-01"
tickers = ["AAPL", "GOOGL", "AMZN", "MSFT"]

for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data.reset_index(inplace=True)  # Reset index to move 'Date' from index to column
        stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns.values]
        stock_data.rename(columns={'Date_': 'Datetime'}, inplace=True)  # Rename 'Date' to 'Datetime'
        file_name = f"{ticker}.csv"
        stock_data.to_csv(file_name)
        print(f"Saved {ticker} data to {file_name}")

#getting historical weather conditions

API_KEY = "9560d0573f55dc058e549b0f9d5fed4e"
CITIES = ["New York,US", "Tokyo,JP", "London,GB"]
BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"
temperature_records = []
rainfall_records = []
for city in CITIES:
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"  # Get data in Celsius
    }
response = requests.get(BASE_URL, params=params)
if response.status_code == 200:
    data = response.json()
    
    for item in data["list"]:
        timestamp = item["dt_txt"]
        temperature = item["main"]["temp"]
        rainfall = item.get("rain", {}).get("3h", 0)  # Rain in last 3 hours (mm)
        
        temperature_records.append({"Datetime": timestamp, "Temperature (°C)": temperature})
        
        rainfall_records.append({"Datetime": timestamp, "Rainfall (mm)": rainfall})

else:
        print(f"Failed to fetch data for {city}")

temp_df = pd.DataFrame(temperature_records)
rain_df = pd.DataFrame(rainfall_records)

temp_df['Datetime'] = pd.to_datetime(temp_df['Datetime'])
rain_df['Datetime'] = pd.to_datetime(rain_df['Datetime'])


temp_filename = f"{city.lower()}_temp.csv"
temp_df.to_csv(temp_filename, index=False)

rain_filename = f"{city.lower()}_rainfall.csv"
rain_df.to_csv(rain_filename, index=False)

print(f"Temperature data saved to {temp_filename}")
print(f"Rainfall data saved to {rain_filename}")

#merging stock + weather data

stock_data["Datetime"] = pd.to_datetime(stock_data["Datetime"])
merged_data_temp = pd.merge_asof(stock_data.sort_values('Datetime'), temp_df.sort_values('Datetime'), on='Datetime', direction='nearest')
merged_data_rain = pd.merge_asof(stock_data.sort_values('Datetime'), rain_df.sort_values('Datetime'), on='Datetime', direction='nearest')

merged_data_temp.dropna(inplace=True)
merged_data_rain.dropna(inplace=True)

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
scaler_temp = MinMaxScaler()
scaler_rain = MinMaxScaler()
temp_scaled = scaler_temp.fit_transform(merged_data_temp)
rain_scaled = scaler_rain.fit_transform(merged_data_rain)

#USING past 30 days of Temperature & Rainfall to predict next day's stock price.
def create_sequences(data, lookback=30):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback, 1:])  # Weather data
        y.append(data[i + lookback, 0])     # Stock 'Close' price
    return np.array(X), np.array(y)

X_temp, y_temp = create_sequences(temp_scaled)
X_rain, y_rain = create_sequences(rain_scaled)

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

model_temp = build_lstm_model((X_temp.shape[1], X_temp.shape[2]))
history_temp = model_temp.fit(X_temp, y_temp, epochs=50, batch_size=32, validation_split=0.2)

# Plot training & validation loss values for a clear visualization of how the model's performance evolves during training.
plt.plot(history_temp.history['loss'], label='Train Loss')
plt.plot(history_temp.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
#plt.show()

# #Training RAIN Model:
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


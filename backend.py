
import yfinance as yf
import requests
#import streamlit as st
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

# #getting historical stock prices

start_date = "2023-01-01"
end_date = "2025-01-01"
tickers = ["AAPL", "GOOGL", "AMZN", "MSFT"]

for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
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

            temp_df = pd.DataFrame(temperature_records)
            rain_df = pd.DataFrame(rainfall_records)

            temp_filename = f"{city.lower()}_temp.csv"
            temp_df.to_csv(temp_filename, index=False)

            rain_filename = f"{city.lower()}_rainfall.csv"
            rain_df.to_csv(rain_filename, index=False)
            
            print(f"Temperature data saved to {temp_filename}")
            print(f"Rainfall data saved to {rain_filename}")
        
        else:
            print(f"Failed to fetch data for {city}")





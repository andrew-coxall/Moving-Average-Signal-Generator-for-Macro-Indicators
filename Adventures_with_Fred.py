import pandas as pd
from fredapi import Fred

FRED_API_KEY = '559636d8c83399a1c2dbffce0bb2c897'

# The FRED ID for the S&P 500 Index
SP500_TICKER = 'SP500'

# Data Fetching Function
def fetch_fred_data(ticker):
    "Fetches time series data from FRED for the given ticker."
    try:
        fred = Fred(api_key=FRED_API_KEY)
        data = fred.get_series(ticker).to_frame(name='Close')
        # Remove row with no values
        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Stratgey funcation
def generate_ma_signals(df, short_window=50, long_window=200):
    "Calculates MAs and generates buy/sell signals."

    #Calculate moving averages
    df['SMA'] = df['Close'].rolling(window=short_window).mean()
    df['LMA'] = df['Close'].rolling(window=long_window).mean()

    


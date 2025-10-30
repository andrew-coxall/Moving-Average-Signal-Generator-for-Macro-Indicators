import pandas as pd
from fredapi import Fred

FRED_API_KEY = '559636d8c83399a1c2dbffce0bb2c897'

SP500_TICKER = 'SP500'

# Data Fetching Function
def fetch_fred_data(ticker):
    """Fetches time series data for a given FRED ticker."""
    try:
        fred = Fred(api_key=FRED_API_KEY)

        data = fred.get_series(ticker).to_frame(name='Close')
        # Remove rows with missing values
        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Strategy Function
def generate_ma_signals(df, short_window=50, long_window=200):
    """Calculates MAs and generates buy/sell signals."""

    # 1. Calculate Moving Averages (using 'Close' column)
    df['SMA'] = df['Close'].rolling(window=short_window).mean()
    df['LMA'] = df['Close'].rolling(window=long_window).mean()

    # 2. Generate Signal
    # Initialize the signal column to 0 (No position/Hold)
    df['Signal'] = 0.0

    # Buy/Risk-On Signal: Short MA crosses above Long MA
    df.loc[df.index[short_window:], 'Signal'] = (df['SMA'][short_window:] > df['LMA'][short_window:]).astype(float)

    df['Position'] = df['Signal'].diff()

    # Fill NaN values from rolling calculation (first 'long_window' rows)
    df.dropna(inplace=True)

    return df

if __name__ == "__main__":
    # 1. Fetch the data
    sp500_data = fetch_fred_data(SP500_TICKER)

    if not sp500_data.empty:
        print("✅ Data successfully fetched for S&P 500:")
        print(sp500_data.tail())

# 2. Generate MA signals and show result
sp500_signals = generate_ma_signals(sp500_data)
print("✅ Signals generated:")
print(sp500_signals.tail())
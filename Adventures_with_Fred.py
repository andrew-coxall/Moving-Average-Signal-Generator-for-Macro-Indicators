import pandas as pd
from fredapi import Fred

FRED_API_KEY = '559636d8c83399a1c2dbffce0bb2c897'

# Define the indicators and their FRED tickers
INDICATORS = {
    'SP500': 'SP500',           # S&P 500 Index (Daily)
    '10Y_Treasury': 'DGS10'     # 10-Year Treasury Yield (Daily)
}

# Data Fetching Function
def fetch_fred_data(ticker):
    """Fetches time series data for a given FRED ticker."""
    try:
        fred = Fred(api_key=FRED_API_KEY)

        data = fred.get_series(ticker).to_frame(name='Close')
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
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
    df.loc[df.index[long_window - 1:], 'Signal'] = (df['SMA'] > df['LMA']).astype(float)

    df['Position'] = df['Signal'].diff()

    # Fill NaN values from rolling calculation (first 'long_window' rows)
    df.dropna(inplace=True)

    return df

if __name__ == "__main__":
    merged_data = None
    for name, ticker in INDICATORS.items():
        data = fetch_fred_data(ticker)
        if data.empty: continue

        data.rename(columns={'Close': f'{name}_Close'}, inplace=True)
        if merged_data is None:
            merged_data = data
        else:
            merged_data = merged_data.merge(data, how='outer', left_index=True, right_index=True)

    if merged_data is not None:
        merged_data.dropna(inplace=True)
        print(f"✅ Data merged. Total points: {len(merged_data):,}")

        # Extract S&P 500 data for signal generation
        sp500_data = merged_data[['SP500_Close']].copy()
        sp500_data.rename(columns={'SP500_Close': 'Close'}, inplace=True)

        # 2. Generate MA signals and show result
        sp500_signals = generate_ma_signals(sp500_data)
        print("✅ Signals generated:")
        print(sp500_signals.tail())

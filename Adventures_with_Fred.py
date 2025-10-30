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

# Strategy Function (Refactored to be reusable for all indicators)
def calculate_ma_crossover_signal(series, asset_name, short_window=50, long_window=200):
    """Calculates MAs and returns a DataFrame with the Signal column."""
    
    df = series.to_frame(name='Close').copy()
    df['SMA'] = df['Close'].rolling(window=short_window).mean()
    df['LMA'] = df['Close'].rolling(window=long_window).mean()

    # 2. Generate Signal
    signal_col = f'{asset_name}_Signal'
    df[signal_col] = 0.0
    
    # Default Risk-On: Short MA > Long MA (e.g., for SP500)
    ma_crossover = (df['SMA'] > df['LMA'])

    # Inverse Risk-On logic for 10Y_Treasury (falling rates is risk-on)
    if asset_name == '10Y_Treasury':
        ma_crossover = (df['SMA'] < df['LMA'])
        
    start_index = long_window - 1 
    df.loc[df.index[start_index:], signal_col] = ma_crossover.iloc[start_index:].astype(float)

    # Drop NaNs from rolling calculation
    df.dropna(subset=[signal_col], inplace=True)

    return df[[signal_col]]

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

        # 2. Generate Signals for ALL Indicators and merge them back
        for name in INDICATORS.keys():
            close_col = f'{name}_Close'
            signal_df = calculate_ma_crossover_signal(merged_data[close_col], name)
            merged_data = merged_data.merge(signal_df, how='inner', left_index=True, right_index=True)

        # 3. Define the Trading Signal (Still using baseline SP500 for now)
        merged_data['Trading_Signal'] = merged_data['SP500_Signal']
        merged_data['Position'] = merged_data['Trading_Signal'].diff()

        # Final cleanup after MA calculation and diff
        merged_data.dropna(inplace=True)
        
        print("✅ Signals generated and merged (showing final 5 rows):")
        print(merged_data[['SP500_Signal', '10Y_Treasury_Signal', 'Position']].tail())

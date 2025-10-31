import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt

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

# --- Backtesting and PnL Simulation ---
def backtest_strategy(df, close_col, signal_col, capital=100000):
    """
    Simulates trades based on a specified 'Signal' column and calculates PnL.
    """
    df['Daily_Return'] = df[close_col].pct_change()
    # Strategy Return = Daily Return * Position (lagged by 1 day)
    df['Strategy_Return'] = df['Daily_Return'] * df[signal_col].shift(1)

    # Calculate cumulative strategy returns (PnL)
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    df['Cumulative_PnL'] = df['Cumulative_Return'] * capital

    return df

# --- Visualization ---
def visualize_strategy(df, asset_name):
    """
    Plots the Cumulative PnL and highlights Buy/Sell entry/exit points.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df['Cumulative_PnL'], label='Strategy PnL', color='dodgerblue', linewidth=2)
    
    # Plot trade entry/exit points (Position changes)
    plt.plot(df.loc[df['Position'] == 1.0].index,
             df['Cumulative_PnL'].loc[df['Position'] == 1.0],
             '^', markersize=10, color='green', label='Buy Signal', alpha=0.7)

    plt.plot(df.loc[df['Position'] == -1.0].index,
             df['Cumulative_PnL'].loc[df['Position'] == -1.0],
             'v', markersize=10, color='red', label='Sell Signal', alpha=0.7)

    plt.title(f'MA Meta-Signal PnL vs Time (Asset: {asset_name})', fontsize=16)
    plt.xlabel('Date'); plt.ylabel('Cumulative PnL ($)'); plt.legend(); plt.grid(True); plt.show()


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

        # 3. Define the Trading Signal (Meta-Signal: Both signals must be 'Risk-On')
        merged_data['Trading_Signal'] = merged_data['SP500_Signal'] * merged_data['10Y_Treasury_Signal']
        merged_data['Position'] = merged_data['Trading_Signal'].diff()

        # Final cleanup after MA calculation and diff
        merged_data.dropna(inplace=True)
        
        print("✅ Signals generated and merged (showing final 5 rows):")
        print(merged_data[['SP500_Signal', '10Y_Treasury_Signal', 'Trading_Signal', 'Position']].tail())
        
        ASSET_NAME = 'SP500'
        CLOSE_COLUMN = f'{ASSET_NAME}_Close'
        
        print("\n--- 4. Backtesting and PnL Simulation (Meta-Signal) ---")
        backtested_data = backtest_strategy(merged_data.copy(), 
                                            close_col=CLOSE_COLUMN, 
                                            signal_col='Trading_Signal')
        if backtested_data is not None:
            final_pnl = backtested_data['Cumulative_PnL'].iloc[-1]
            initial_capital = backtested_data['Cumulative_PnL'].iloc[0]
            total_return = (final_pnl / initial_capital - 1) * 100
            print(f"💰 Final PnL: ${final_pnl:,.2f} | Total Return: {total_return:,.2f}%")
            
            # 5. Visualize the PnL and trade points
            visualize_strategy(backtested_data, ASSET_NAME)

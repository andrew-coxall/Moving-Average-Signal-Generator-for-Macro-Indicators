import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import numpy as np

# NOTE: The FRED_API_KEY must be a valid key for the script to fetch live data.
# The key provided here is for example only.
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

# --- Performance Metrics Calculation ---
def calculate_performance_metrics(df, return_col, risk_free_rate=0.0):
    """Calculates key performance metrics like CAGR, Max Drawdown, and Sharpe Ratio."""
    
    # 1. Annualized Return (CAGR)
    days = (df.index[-1] - df.index[0]).days
    cumulative_return = (1 + df[return_col]).cumprod().iloc[-1]
    cagr = (cumulative_return ** (365.25 / days)) - 1
    
    # 2. Maximum Drawdown (MDD)
    # Calculate the running maximum (peak)
    peak = (1 + df[return_col]).cumprod().cummax()
    # Calculate the Drawdown
    drawdown = ((1 + df[return_col]).cumprod() / peak) - 1
    mdd = drawdown.min()
    
    # 3. Sharpe Ratio (Annualized)
    annualized_volatility = df[return_col].std() * (252**0.5) # 252 trading days
    sharpe_ratio = (cagr - risk_free_rate) / annualized_volatility

    return {
        'CAGR': cagr,
        'Max Drawdown': mdd,
        'Sharpe Ratio': sharpe_ratio
    }

# --- Visualization ---
def visualize_strategy(df, asset_name, benchmark_pnl):
    """
    Plots the Cumulative PnL, the benchmark, the underlying asset price, 
    and adds vertical lines for trade entry/exit points.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7)) # Create figure and first axis (for PnL)

    # Plot PnL curves on ax1 (Left Y-Axis)
    ax1.plot(df['Cumulative_PnL'], label='Strategy PnL (Meta-Signal)', color='dodgerblue', linewidth=2)
    ax1.plot(benchmark_pnl, label=f'Benchmark PnL (Buy & Hold {asset_name})', color='darkorange', linestyle='--', linewidth=1.5)
    ax1.set_ylabel('Cumulative PnL ($)', color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    
    # Plot trade entry/exit points (Position changes) on ax1
    # Vertical lines for Buy and Sell signals
    buy_signals = df.loc[df['Position'] == 1.0].index
    sell_signals = df.loc[df['Position'] == -1.0].index
    
    for date in buy_signals:
        ax1.axvline(x=date, color='green', linestyle=':', alpha=0.6, linewidth=1, label='_nolegend_')
    for date in sell_signals:
        ax1.axvline(x=date, color='red', linestyle=':', alpha=0.6, linewidth=1, label='_nolegend_')
        
    # Scatter plot on PnL curve to mark the points
    ax1.plot(buy_signals,
             df['Cumulative_PnL'].loc[buy_signals],
             '^', markersize=10, color='green', label='Buy Signal', alpha=0.7)

    ax1.plot(sell_signals,
             df['Cumulative_PnL'].loc[sell_signals],
             'v', markersize=10, color='red', label='Sell Signal', alpha=0.7)

    # Create a second Y-axis (Right Y-Axis) for the S&P 500 Price
    ax2 = ax1.twinx()
    close_col = f'{asset_name}_Close'
    ax2.plot(df[close_col], label=f'{asset_name} Price', color='gray', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel(f'{asset_name} Price (FRED Data)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.set_title(f'MA Meta-Signal vs. Buy & Hold Benchmark (Asset: {asset_name})', fontsize=16)
    ax1.set_xlabel('Date')
    ax1.grid(True)
    plt.show()


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
                                            
        # --- Buy and Hold Benchmark Calculation ---
        initial_capital = backtested_data['Cumulative_PnL'].iloc[0]
        sp500_returns = backtested_data[CLOSE_COLUMN].pct_change().dropna()
        benchmark_returns = (1 + sp500_returns).cumprod()
        benchmark_pnl = benchmark_returns * initial_capital
        
        if backtested_data is not None:
            final_pnl = backtested_data['Cumulative_PnL'].iloc[-1]
            total_return = (final_pnl / initial_capital - 1) * 100
            print(f"💰 Final Strategy PnL: ${final_pnl:,.2f} | Total Return: {total_return:,.2f}%")
            
            # Calculate and print strategy performance metrics
            metrics = calculate_performance_metrics(backtested_data.dropna(), 'Strategy_Return')
            print("📊 Strategy Performance Metrics:")
            print(f"   Annualized Return (CAGR): {metrics['CAGR'] * 100:.2f}%")
            print(f"   Max Drawdown (MDD): {metrics['Max Drawdown'] * 100:.2f}%")
            print(f"   Sharpe Ratio (Annualized): {metrics['Sharpe Ratio']:.2f}")

            # Calculate and print benchmark performance metrics
            benchmark_metrics = calculate_performance_metrics(backtested_data.dropna(), 'Daily_Return')
            print("\n📈 Benchmark (Buy & Hold) Performance:")
            print(f"   Annualized Return (CAGR): {benchmark_metrics['CAGR'] * 100:.2f}%")
            print(f"   Max Drawdown (MDD): {benchmark_metrics['Max Drawdown'] * 100:.2f}%")
            print(f"   Sharpe Ratio (Annualized): {benchmark_metrics['Sharpe Ratio']:.2f}")

            # 5. Visualize the PnL and trade points
            visualize_strategy(backtested_data, ASSET_NAME, benchmark_pnl)

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

# --- Backtesting and PnL Simulation with 5% Trailing Stop-Loss ---
# This is the function you wanted to use, which includes the trailing stop logic.
def backtest_strategy(df, close_col, signal_col, capital=1000, trailing_stop=0.05):
    """
    Simulates trades based on a specified 'Signal' column and calculates PnL.
    Fully integrates a 5% trailing stop-loss to mitigate large sudden losses.
    """
    df['Daily_Return'] = df[close_col].pct_change()
    
    cumulative_return = 1.0
    peak_return = 1.0
    position = 0  # 1 = long, 0 = flat

    cumulative_returns = []
    positions = []
    strategy_returns = []

    for i in range(len(df)):
        signal = df[signal_col].iloc[i]

        # Handle signal changes
        if signal == 1 and position == 0:
            position = 1  # Enter long
            peak_return = cumulative_return  # reset peak for new trade
        elif signal == 0 and position == 1:
            position = 0  # Exit to cash

        # Apply daily return if in position
        daily_ret = df['Daily_Return'].iloc[i] * position if not pd.isna(df['Daily_Return'].iloc[i]) else 0.0
        cumulative_return *= (1 + daily_ret)

        # --- Apply Trailing Stop ---
        if position == 1 and trailing_stop > 0:
            peak_return = max(peak_return, cumulative_return)
            if cumulative_return < peak_return * (1 - trailing_stop):
                position = 0  # Exit immediately
                peak_return = cumulative_return  # reset peak for next entry

        cumulative_returns.append(cumulative_return)
        positions.append(position)
        strategy_returns.append(daily_ret)

    df['Cumulative_Return'] = cumulative_returns
    df['Cumulative_PnL'] = df['Cumulative_Return'] * capital
    df['Position'] = positions
    df['Strategy_Return'] = strategy_returns

    return df

# --- Performance Metrics Calculation ---
def calculate_performance_metrics(df, return_col, risk_free_rate=0.0):
    """Calculates key performance metrics like CAGR, Max Drawdown, and Sharpe Ratio."""
    
    # 💥 CRITICAL FIX: Apply targeted dropna and check for empty DataFrame
    df_clean = df.copy().dropna(subset=[return_col])
    
    if df_clean.empty or len(df_clean) < 2:
        # Prevents IndexError if DataFrame is empty or has only one row
        return {
            'CAGR': 0.0,
            'Max Drawdown': 0.0,
            'Sharpe Ratio': 0.0
        }
        
    # 1. Annualized Return (CAGR)
    days = (df_clean.index[-1] - df_clean.index[0]).days
    
    # If the time span is too short (e.g., 0 days), handle it
    if days == 0:
        return {
            'CAGR': 0.0, 'Max Drawdown': 0.0, 'Sharpe Ratio': 0.0
        }

    cumulative_return = (1 + df_clean[return_col]).cumprod().iloc[-1]
    cagr = (cumulative_return ** (365.25 / days)) - 1
    
    # 2. Maximum Drawdown (MDD)
    # Calculate the running maximum (peak)
    peak = (1 + df_clean[return_col]).cumprod().cummax()
    # Calculate the Drawdown
    drawdown = ((1 + df_clean[return_col]).cumprod() / peak) - 1
    mdd = drawdown.min()
    
    # 3. Sharpe Ratio (Annualized)
    annualized_volatility = df_clean[return_col].std() * (252**0.5) # 252 trading days
    sharpe_ratio = (cagr - risk_free_rate) / annualized_volatility if annualized_volatility else 0.0

    return {
        'CAGR': cagr,
        'Max Drawdown': mdd,
        'Sharpe Ratio': sharpe_ratio
    }

# --- Visualization ---
def visualize_strategy(df, asset_name):
    """
    Plots the Strategy Cumulative PnL, adds text labels for "Total amount in account" 
    at each signal with alternating vertical placement for better readability, 
    moves the final balance to the bottom right, and updates the legend.
    """
    fig, ax1 = plt.subplots(figsize=(16, 8)) # Slightly larger figure for better display

    # Plot PnL curve
    ax1.plot(df['Cumulative_PnL'], label='Strategy PnL (Meta-Signal)', color='dodgerblue', linewidth=2)
    ax1.set_ylabel('Cumulative PnL ($)', color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    
    # Identify trade signals (where position status changes)
    df['Position_Change'] = df['Position'].diff()
    # Entry is a change from 0 to 1 (Position_Change = 1)
    # Exit is a change from 1 to 0 (Position_Change = -1)
    signals = df.loc[df['Position_Change'].abs() == 1.0].copy() 
    
    # Scatter plot on PnL curve to mark the points (Larger markers)
    buy_signals = signals.loc[signals['Position_Change'] == 1.0]
    sell_signals = signals.loc[signals['Position_Change'] == -1.0]
    
    ax1.plot(buy_signals.index,
             buy_signals['Cumulative_PnL'],
             '^', markersize=12, color='green', label='Buy Signal (Go Long)', alpha=0.9)

    ax1.plot(sell_signals.index,
             sell_signals['Cumulative_PnL'],
             'v', markersize=12, color='red', label='Sell Signal (Go Flat)', alpha=0.9)

    # Define the bounding box style for better readability
    bbox_style = {
        'facecolor': 'white', # White background
        'alpha': 0.7,         # Semi-transparent
        'boxstyle': 'round,pad=0.4', # Rounded corners
        'edgecolor': 'none'   # No border
    }

    # --- Alternating Vertical Placement for Readability ---
    vertical_offset_multiplier_base = 1.015 
    vertical_offset_multiplier_alt = 1.045 
    counter = 0

    for date, row in signals.iterrows():
        color = 'green' if row['Position_Change'] == 1.0 else 'red'
        
        # Calculate y_pos based on alternating counter
        if counter % 2 == 0:
            y_pos = row['Cumulative_PnL'] * vertical_offset_multiplier_base
        else:
            y_pos = row['Cumulative_PnL'] * vertical_offset_multiplier_alt
            
        counter += 1
        
        ax1.text(date, y_pos, f"${row['Cumulative_PnL']:.2f}", 
                 color=color, 
                 ha='center', 
                 va='bottom', 
                 fontsize=10,        
                 weight='bold',      
                 bbox=bbox_style)    

    # --- ADD FINAL PNL AMOUNT (Moved to Bottom Right) ---
    final_pnl = df['Cumulative_PnL'].iloc[-1] if not df.empty else np.nan
    
    # 💥 FIX: Ensure Final PnL text handles NaN gracefully
    if np.isnan(final_pnl):
        final_pnl_text = "Final PnL: $0.00 (No Data)"
    else:
        final_pnl_text = f"Final PnL: ${final_pnl:,.2f}"

    # Use a fixed fraction of the y-axis range for consistent placement
    ax1.text(0.98, 0.05, final_pnl_text, 
             transform=ax1.transAxes,
             color='black', 
             ha='right', 
             va='bottom', 
             fontsize=14,        
             weight='extra bold', 
             bbox={'facecolor': 'yellow', 'alpha': 0.8, 'boxstyle': 'round,pad=0.5', 'edgecolor': 'black'})


    # --- LEGEND UPDATE: Include explanation for PnL tracker numbers ---
    
    # Combine existing handles/labels
    lines, labels = ax1.get_legend_handles_labels()

    # Add the new descriptive label
    lines.append(plt.Line2D([0], [0], linestyle='none', marker='None')) # Dummy handle for text
    labels.append('\n**Account Balance Tracker**:\nGreen/Red numbers indicate the account value at the exact moment a Buy/Sell signal is generated.')

    # Place the updated legend
    ax1.legend(lines, labels, loc='upper left')

    ax1.set_title(f'MA Meta-Signal Strategy PnL (Asset: {asset_name})', fontsize=16)
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
        # Position column is redefined inside backtest_strategy for accurate plotting of PnL change
        merged_data['Position'] = merged_data['Trading_Signal'] 

        # Final cleanup after MA calculation
        merged_data.dropna(inplace=True)
        
        print("✅ Signals generated and merged (showing final 5 rows):")
        print(merged_data[['SP500_Signal', '10Y_Treasury_Signal', 'Trading_Signal', 'Position']].tail())
        
        ASSET_NAME = 'SP500'
        CLOSE_COLUMN = f'{ASSET_NAME}_Close'
        initial_capital = 1000 # Initial capital for backtesting
        
        print("\n--- 4. Backtesting and PnL Simulation (Meta-Signal with Trailing Stop-Loss) ---")
        backtested_data = backtest_strategy(merged_data.copy(), 
                                            close_col=CLOSE_COLUMN, 
                                            signal_col='Trading_Signal',
                                            capital=initial_capital,
                                            trailing_stop=0.05)  # 5% Trailing Stop
                                            
        
        if backtested_data is not None and not backtested_data.empty: 
            final_pnl = backtested_data['Cumulative_PnL'].iloc[-1]
            total_return = (final_pnl / initial_capital - 1) * 100
            print(f"💰 Final Strategy PnL: ${final_pnl:,.2f} | Total Return: {total_return:,.2f}%")
            
            # Calculate and print strategy performance metrics
            # 💥 CRITICAL FIX: Removed the unsafe .dropna() call
            metrics = calculate_performance_metrics(backtested_data, 'Strategy_Return')
            print("📊 Strategy Performance Metrics:")
            print(f"   Annualized Return (CAGR): {metrics['CAGR'] * 100:.2f}%")
            print(f"   Max Drawdown (MDD): {metrics['Max Drawdown'] * 100:.2f}%")
            print(f"   Sharpe Ratio (Annualized): {metrics['Sharpe Ratio']:.2f}")

            # Calculate and print benchmark performance metrics
            # 💥 CRITICAL FIX: Removed the unsafe .dropna() call
            benchmark_metrics = calculate_performance_metrics(backtested_data, 'Daily_Return')
            print("\n📈 Benchmark (Buy & Hold) Performance:")
            print(f"   Annualized Return (CAGR): {benchmark_metrics['CAGR'] * 100:.2f}%")
            print(f"   Max Drawdown (MDD): {benchmark_metrics['Max Drawdown'] * 100:.2f}%")
            print(f"   Sharpe Ratio (Annualized): {benchmark_metrics['Sharpe Ratio']:.2f}")

            # 5. Visualize the PnL and trade points (Chart GUI)
            visualize_strategy(backtested_data, ASSET_NAME)
        else:
            print("\n🚨 Backtested data is empty. Cannot calculate performance or visualize.")
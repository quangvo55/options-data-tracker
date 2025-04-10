import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf

def load_json_data(candle_file, trades_file):
    """
    Load candle and trade data from JSON files
    """
    # Load candle data
    with open(candle_file, 'r') as f:
        candle_data = json.load(f)
    
    # Load trade data
    with open(trades_file, 'r') as f:
        trade_data = json.load(f)
    
    return candle_data, trade_data

def convert_candles_to_dataframe(candles, last_n=None):
    """
    Convert candle data to pandas DataFrame for mplfinance
    
    Parameters:
    - candles: Dictionary containing candle data
    - last_n: Optional, number of most recent candles to include
    
    Returns:
    - DataFrame with OHLCV data
    """
    # Extract candle data
    candle_list = candles['candles']
    
    # Trim to last_n candles if specified
    if last_n is not None and last_n > 0 and last_n < len(candle_list):
        candle_list = candle_list[-last_n:]
    
    # Create DataFrame
    df = pd.DataFrame(candle_list)
    
    # Convert datetime to pandas datetime and set as index
    df['date'] = pd.to_datetime(df['datetime'], unit='ms')
    df = df.set_index('date')
    
    # Keep only OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    return df

def find_candle_index(target_date_str, candle_array):
    """
    Find the closest candle index for a given date string
    
    Parameters:
    - target_date_str: Date string in format '2025-02-24 20:45:00'
    - candle_array: Array of candle dictionaries
    
    Returns:
    - Index of the closest candle
    """
    # Parse the trade date using proper UTC handling
    # Convert 'YYYY-MM-DD HH:MM:SS' to ISO 8601 format
    iso_date_str = target_date_str.replace(' ', 'T') + 'Z'
    target_date = datetime.fromisoformat(iso_date_str.replace('Z', '+00:00'))
    target_timestamp = int(target_date.timestamp() * 1000)  # Convert to milliseconds
    
    # Find the closest candle
    closest_index = -1
    smallest_diff = float('inf')
    
    for i, candle in enumerate(candle_array):
        diff = abs(candle['datetime'] - target_timestamp)
        if diff < smallest_diff:
            smallest_diff = diff
            closest_index = i
    
    return closest_index

def create_trade_markers(trades, candles):
    """
    Create arrays for plotting trade markers on the chart
    
    Parameters:
    - trades: List of trade dictionaries
    - candles: Dictionary with candle data
    """
    total_candles = len(candles['candles'])
    
    # Initialize arrays with NaN values
    long_entries = np.full(total_candles, np.nan)
    long_exits = np.full(total_candles, np.nan)
    short_entries = np.full(total_candles, np.nan)
    short_exits = np.full(total_candles, np.nan)
    
    # Filter trades to those within our candle date range
    first_candle_time = candles['candles'][0]['datetime']
    last_candle_time = candles['candles'][-1]['datetime']
    
    valid_trades = []
    for trade in trades:
        # Convert trade dates to timestamp
        entry_date = datetime.strptime(trade['entry_date'], '%Y-%m-%d %H:%M:%S')
        exit_date = datetime.strptime(trade['exit_date'], '%Y-%m-%d %H:%M:%S')
        
        entry_timestamp = int(entry_date.timestamp() * 1000)
        exit_timestamp = int(exit_date.timestamp() * 1000)
        
        if entry_timestamp >= first_candle_time and exit_timestamp <= last_candle_time:
            valid_trades.append(trade)
    
    print(f"Found {len(valid_trades)} trades within the candle date range.")
    
    # Map trades to candle indices and create markers
    successful_trades = 0
    unsuccessful_trades = 0
    
    for trade in valid_trades:
        entry_index = find_candle_index(trade['entry_date'], candles['candles'])
        exit_index = find_candle_index(trade['exit_date'], candles['candles'])
        
        # Track trade performance
        if trade['profit'] > 0:
            successful_trades += 1
        else:
            unsuccessful_trades += 1
        
        # Place markers on the chart
        if trade['position'] == 'LONG':
            # Long entry: green arrow below the candle
            long_entries[entry_index] = trade['entry_price']
            # Long exit: green arrow above the candle
            long_exits[exit_index] = trade['exit_price']
        else:  # SHORT
            # Short entry: red arrow above the candle
            short_entries[entry_index] = trade['entry_price']
            # Short exit: red arrow below the candle
            short_exits[exit_index] = trade['exit_price']
    
    print(f"Trade summary: {successful_trades} profitable, {unsuccessful_trades} unprofitable")
    
    return long_entries, long_exits, short_entries, short_exits, valid_trades

def plot_candlestick_with_trades(df, long_entries, long_exits, short_entries, short_exits, 
                                symbol_name, start_date=None, end_date=None, save_path=None):
    """
    Plot candlestick chart with trade markers
    
    Parameters:
    - df: DataFrame with OHLCV data
    - long_entries, long_exits, short_entries, short_exits: Arrays with marker positions
    - symbol_name: Symbol name for the title
    - start_date: Optional start date for filtering (string in format 'YYYY-MM-DD')
    - end_date: Optional end date for filtering (string in format 'YYYY-MM-DD')
    - save_path: Optional path to save the chart
    """
    # Filter by date range if specified
    if start_date or end_date:
        mask = pd.Series(True, index=df.index)
        
        if start_date:
            start = pd.Timestamp(start_date)
            mask = mask & (df.index >= start)
        
        if end_date:
            end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask = mask & (df.index <= end)
        
        # Get indices in the range
        filtered_indices = np.where(mask)[0]
        
        if len(filtered_indices) == 0:
            print("No data in the specified date range.")
            return
        
        # Get the start and end indices
        start_idx = filtered_indices[0]
        end_idx = filtered_indices[-1] + 1
        
        # Slice the dataframe and the marker arrays
        filtered_df = df.iloc[start_idx:end_idx]
        filtered_long_entries = long_entries[start_idx:end_idx]
        filtered_long_exits = long_exits[start_idx:end_idx]
        filtered_short_entries = short_entries[start_idx:end_idx]
        filtered_short_exits = short_exits[start_idx:end_idx]
    else:
        # Use all data
        filtered_df = df
        filtered_long_entries = long_entries
        filtered_long_exits = long_exits
        filtered_short_entries = short_entries
        filtered_short_exits = short_exits
    
    # Define styles for trade markers
    long_entry_marker = dict(
        type='scatter',
        markersize=100,
        marker='^',  # Up triangle
        color='darkblue',
        alpha=1
    )
    
    long_exit_marker = dict(
        type='scatter',
        markersize=100,
        marker='v',  # Down triangle
        color='orange',
        alpha=0.7
    )
    
    short_entry_marker = dict(
        type='scatter',
        markersize=100,
        marker='v',  # Down triangle
        color='magenta',
        alpha=0.7
    )
    
    short_exit_marker = dict(
        type='scatter',
        markersize=100,
        marker='^',  # Up triangle
        color='cyan',
        alpha=1
    )
    
    # Create a list of additional plots for the trade markers
    add_plots = [
        mpf.make_addplot(filtered_long_entries, **long_entry_marker),
        mpf.make_addplot(filtered_long_exits, **long_exit_marker),
        mpf.make_addplot(filtered_short_entries, **short_entry_marker),
        mpf.make_addplot(filtered_short_exits, **short_exit_marker)
    ]
    
    # Define the style
    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit',
        wick={'up': 'green', 'down': 'red'},
        volume={'up': 'green', 'down': 'red'}
    )
    
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mc,
        gridstyle='--',
        y_on_right=True,
        facecolor='white'
    )
    
    # Create the plot title
    if start_date and end_date:
        title = f"{symbol_name} - 15-Minute Candles ({start_date} to {end_date}) with Trade Markers"
    elif start_date:
        title = f"{symbol_name} - 15-Minute Candles (From {start_date}) with Trade Markers"
    elif end_date:
        title = f"{symbol_name} - 15-Minute Candles (Until {end_date}) with Trade Markers"
    else:
        title = f"{symbol_name} - 15-Minute Candles with Trade Markers"
    
    # Plot the candlestick chart with markers
    kwargs = dict(
        type='candle',
        style=s,
        title=title,
        volume=True,
        addplot=add_plots,
        figsize=(15, 10),
        panel_ratios=(6, 1),
        returnfig=True
    )
    
    # Add extra options for different sized datasets
    if len(filtered_df) > 100:
        # For larger datasets, reduce width to accommodate more candles
        fig, axes = mpf.plot(filtered_df, **kwargs)
    else:
        # For smaller datasets, increase width for better visibility
        kwargs['figscale'] = 1.5
        fig, axes = mpf.plot(filtered_df, **kwargs)
    
    # Remove the automatically generated legend
    if hasattr(axes[0], 'legend_') and axes[0].legend_:
        axes[0].legend_.remove()
    
    # Add a custom legend with the correct markers
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='lightblue', markersize=10, label='Long Entry'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='darkblue', markersize=10, label='Long Exit'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='magenta', markersize=10, label='Short Entry'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='cyan', markersize=10, label='Short Exit')
    ]
    
    axes[0].legend(handles=custom_lines, loc='upper right', framealpha=0.9)
    # Save the chart if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def analyze_trades(trades):
    """
    Analyze trade performance and print statistics
    """
    if not trades:
        print("No trades to analyze.")
        return
    
    # Calculate statistics
    total_trades = len(trades)
    profitable_trades = sum(1 for t in trades if t['profit'] > 0)
    losing_trades = total_trades - profitable_trades
    
    total_profit = sum(t['profit'] for t in trades)
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    
    win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
    
    # Print statistics
    print("\nTrade Analysis")
    print("=" * 50)
    print(f"Total Trades: {total_trades}")
    print(f"Profitable Trades: {profitable_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades: {losing_trades} ({100-win_rate:.2f}%)")
    print(f"Total Profit/Loss: {total_profit:.2f}")
    print(f"Average Profit/Loss per Trade: {avg_profit:.2f}")
    
    # Analyze by position type
    long_trades = [t for t in trades if t['position'] == 'LONG']
    short_trades = [t for t in trades if t['position'] == 'SHORT']
    
    long_profit = sum(t['profit'] for t in long_trades)
    short_profit = sum(t['profit'] for t in short_trades)
    
    print("\nPosition Analysis")
    print("-" * 50)
    print(f"Long Trades: {len(long_trades)}")
    print(f"Long Profit/Loss: {long_profit:.2f}")
    print(f"Average Long P/L: {long_profit/len(long_trades):.2f}" if long_trades else "No long trades")
    
    print(f"Short Trades: {len(short_trades)}")
    print(f"Short Profit/Loss: {short_profit:.2f}")
    print(f"Average Short P/L: {short_profit/len(short_trades):.2f}" if short_trades else "No short trades")
    
    # Analyze by exit reason
    exit_reasons = set(t['exit_reason'] for t in trades)
    print("\nExit Reason Analysis")
    print("-" * 50)
    for reason in exit_reasons:
        reason_trades = [t for t in trades if t['exit_reason'] == reason]
        reason_profit = sum(t['profit'] for t in reason_trades)
        print(f"{reason}: {len(reason_trades)} trades, {reason_profit:.2f} profit")
    
    print("=" * 50)

def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='Plot candlestick chart with trade markers')
    parser.add_argument('--candles', type=str, default='data/test.json', help='Path to candle data JSON file')
    parser.add_argument('--trades', type=str, default='trades.json', help='Path to trade data JSON file')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--last', type=int, help='Show only the last N candles')
    parser.add_argument('--output', type=str, default='chart_with_trades.png', help='Output file path')
    parser.add_argument('--analyze', action='store_true', help='Show trade analysis statistics')
    return parser.parse_args()

def main():
    """
    Main function to load data and create the chart
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # File paths
    candle_file = args.candles
    trades_file = args.trades
    
    # Load data
    candles, trades = load_json_data(candle_file, trades_file)
    
    # Convert candle data to DataFrame with trimming if needed
    df = convert_candles_to_dataframe(candles, last_n=args.last)
    
    # Create trade markers based on the trimmed data
    # We need to pass only the relevant portion of candles array if trimming was done
    if args.last and args.last < len(candles['candles']):
        candles_for_markers = {'candles': candles['candles'][-args.last:], 'symbol': candles.get('symbol', 'Unknown')}
        long_entries, long_exits, short_entries, short_exits, valid_trades = create_trade_markers(trades, candles_for_markers)
    else:
        long_entries, long_exits, short_entries, short_exits, valid_trades = create_trade_markers(trades, candles)
    
    # Get the symbol name
    symbol_name = candles.get('symbol', 'Unknown')
    
    # Analyze trades if requested
    if args.analyze:
        analyze_trades(valid_trades)
    
    # Plot with date range
    plot_candlestick_with_trades(
        df,
        long_entries,
        long_exits,
        short_entries,
        short_exits,
        symbol_name,
        start_date=args.start,
        end_date=args.end,
        save_path=args.output
    )

if __name__ == "__main__":
    main()
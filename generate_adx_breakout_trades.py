import json
import pandas as pd
import numpy as np
from datetime import datetime
import ta  # Make sure to install this with: pip install ta

class TradingStrategy:
    def __init__(self, length=30, adx_threshold=40, adx_window=15, stop_loss=250, profit_target=1000, max_bars_in_trade=20):
        """
        Initialize the trading strategy with parameters
        
        Parameters:
        - length: Period for lookback window for highest high/lowest low
        - adx_threshold: ADX threshold for trading signals
        - adx_window: Window period for ADX calculation
        - stop_loss: Stop loss amount in points
        - profit_target: Profit target amount in points
        - max_bars_in_trade: Maximum number of bars to stay in a trade
        """
        self.length = length
        self.adx_threshold = adx_threshold
        self.adx_window = adx_window
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.max_bars_in_trade = max_bars_in_trade
        self.trades = []
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
    
    def load_data(self, file_path):
        """
        Load candlestick data from a JSON file
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Convert to DataFrame
        df = pd.DataFrame(data['candles'])
        
        # Convert datetime from milliseconds to readable format
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        
        return df
    
    def calculate_adx(self, dataframe):
        """
        Calculate the Average Directional Index (ADX) using the ta library
        
        Parameters:
        - dataframe: DataFrame with OHLC data
        
        Returns:
        - DataFrame with ADX values
        """
        df = dataframe.copy()
        
        # Calculate ADX using the ta library
        adx_indicator = ta.trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.adx_window,
            fillna=True
        )
        
        # Add ADX values to the dataframe
        df[f'adx_{self.adx_window}'] = adx_indicator.adx()
        df[f'adx_pos_{self.adx_window}'] = adx_indicator.adx_pos()
        df[f'adx_neg_{self.adx_window}'] = adx_indicator.adx_neg()
        
        return df
    
    def prepare_data(self, df):
        """
        Prepare data for strategy execution
        
        Parameters:
        - df: DataFrame with OHLC data
        
        Returns:
        - DataFrame with indicators and signals
        """
        # Calculate ADX
        df = self.calculate_adx(df)
        
        # Calculate rolling highest high and lowest low over the specified length period
        df['highest_high'] = df['high'].rolling(window=self.length).max()
        df['lowest_low'] = df['low'].rolling(window=self.length).min()
        
        # Generate trading signals based on ADX threshold
        df[f'adx_signal'] = df[f'adx_{self.adx_window}'] < self.adx_threshold
        
        return df
    
    def add_trade(self, entry_date, exit_date, position, entry_price, exit_price, profit, exit_reason, adx_value=None, highest_high=None, lowest_low=None):
        """
        Add a trade to the list of trades
        """
        trade = {
            'entry_date': entry_date.strftime("%Y-%m-%d %H:%M:%S") if isinstance(entry_date, datetime) else entry_date,
            'exit_date': exit_date.strftime("%Y-%m-%d %H:%M:%S") if isinstance(exit_date, datetime) and exit_date is not None else exit_date,
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit,
            'exit_reason': exit_reason
        }
        
        # Add additional data if provided
        if adx_value is not None:
            trade['adx_value'] = adx_value
        if highest_high is not None:
            trade['highest_high'] = highest_high
        if lowest_low is not None:
            trade['lowest_low'] = lowest_low
            
        self.trades.append(trade)
    
    def backtest(self, file_path):
        """
        Run the strategy backtest on the provided data
        
        Parameters:
        - file_path: Path to JSON file with candle data
        
        Returns:
        - DataFrame with processed data
        - List of trades
        """
        # Load and prepare data
        df = self.load_data(file_path)
        df = self.prepare_data(df)
        
        # Print strategy parameters
        print(f"Strategy Parameters:")
        print(f"Length for highest/lowest: {self.length}, ADX Window: {self.adx_window}, ADX Threshold: {self.adx_threshold}")
        print(f"Stop Loss: {self.stop_loss}, Profit Target: {self.profit_target}, Max Bars in Trade: {self.max_bars_in_trade}")
        
        # Initialize position tracking variables
        position = 0  # 0: no position, 1: long position, -1: short position
        entry_price = 0
        trade_index = None  # To track which trade we're currently in
        bars_in_trade = 0  # Counter for how many bars we've been in the current trade
        
        # Print signals
        print("\nTrading Signals and Position Management:")
        for i in range(self.length, len(df)-1):  # Stop one bar before the end to allow for next bar entry
            current_date = df['datetime'].iloc[i]
            next_date = df['datetime'].iloc[i+1]
            current_high = df['close'].iloc[i]
            current_low = df['close'].iloc[i]
            next_high = df['close'].iloc[i+1]
            next_low = df['close'].iloc[i+1]
            adx_value = df[f'adx_{self.adx_window}'].iloc[i]
            highest = df['highest_high'].iloc[i]
            lowest = df['lowest_low'].iloc[i]
            
            # Check if we should exit existing position based on current bar
            if position != 0:  # If we're in a position (long or short)
                bars_in_trade += 1  # Increment counter for bars in trade
                
                # Time-based exit: Close position after max_bars_in_trade
                if bars_in_trade >= self.max_bars_in_trade:
                    current_close = df['close'].iloc[i]
                    if position == 1:  # Long position
                        profit = current_close - entry_price
                        self.total_profit += profit
                        print(f"{current_date}: CLOSE LONG at {current_close} (time-based exit after {bars_in_trade} bars, profit/loss: {profit})")
                        
                        # Update the exit information for the current trade
                        self.trades[trade_index]['exit_date'] = current_date.strftime("%Y-%m-%d %H:%M:%S")
                        self.trades[trade_index]['exit_price'] = current_close
                        self.trades[trade_index]['profit'] = profit
                        self.trades[trade_index]['exit_reason'] = 'time_based'
                        
                        if profit > 0:
                            self.winning_trades += 1
                        else:
                            self.losing_trades += 1
                            
                        position = 0
                        trade_index = None
                        bars_in_trade = 0
                        continue  # Skip to next iteration to avoid checking other exit conditions
                    
                    elif position == -1:  # Short position
                        profit = entry_price - current_close
                        self.total_profit += profit
                        print(f"{current_date}: CLOSE SHORT at {current_close} (time-based exit after {bars_in_trade} bars, profit/loss: {profit})")
                        
                        # Update the exit information for the current trade
                        self.trades[trade_index]['exit_date'] = current_date.strftime("%Y-%m-%d %H:%M:%S")
                        self.trades[trade_index]['exit_price'] = current_close
                        self.trades[trade_index]['profit'] = profit
                        self.trades[trade_index]['exit_reason'] = 'time_based'
                        
                        if profit > 0:
                            self.winning_trades += 1
                        else:
                            self.losing_trades += 1
                            
                        position = 0
                        trade_index = None
                        bars_in_trade = 0
                        continue  # Skip to next iteration to avoid checking other exit conditions
            if position == 1:  # Long position
                # Check for profit target
                if current_high >= entry_price + self.profit_target:
                    profit = self.profit_target
                    self.total_profit += profit
                    print(f"{current_date}: CLOSE LONG at {entry_price + self.profit_target} (profit target reached, profit: {profit})")
                    # Update the exit information for the current trade
                    self.trades[trade_index]['exit_date'] = current_date.strftime("%Y-%m-%d %H:%M:%S")
                    self.trades[trade_index]['exit_price'] = entry_price + self.profit_target
                    self.trades[trade_index]['profit'] = profit
                    self.trades[trade_index]['exit_reason'] = 'profit_target'
                    
                    self.winning_trades += 1
                    position = 0
                    trade_index = None
                    bars_in_trade = 0
                    bars_in_trade = 0
                # Check for stop loss
                elif current_low <= entry_price - self.stop_loss:
                    loss = self.stop_loss
                    self.total_profit -= loss
                    print(f"{current_date}: CLOSE LONG at {entry_price - self.stop_loss} (stop loss hit, loss: {loss})")
                    # Update the exit information for the current trade
                    self.trades[trade_index]['exit_date'] = current_date.strftime("%Y-%m-%d %H:%M:%S")
                    self.trades[trade_index]['exit_price'] = entry_price - self.stop_loss
                    self.trades[trade_index]['profit'] = -loss
                    self.trades[trade_index]['exit_reason'] = 'stop_loss'
                    
                    self.losing_trades += 1
                    position = 0
                    trade_index = None
                    bars_in_trade = 0
                    bars_in_trade = 0
            
            elif position == -1:  # Short position
                # Check for profit target
                if current_low <= entry_price - self.profit_target:
                    profit = self.profit_target
                    self.total_profit += profit
                    print(f"{current_date}: CLOSE SHORT at {entry_price - self.profit_target} (profit target reached, profit: {profit})")
                    # Update the exit information for the current trade
                    self.trades[trade_index]['exit_date'] = current_date.strftime("%Y-%m-%d %H:%M:%S")
                    self.trades[trade_index]['exit_price'] = entry_price - self.profit_target
                    self.trades[trade_index]['profit'] = profit
                    self.trades[trade_index]['exit_reason'] = 'profit_target'
                    
                    self.winning_trades += 1
                    position = 0
                    trade_index = None
                # Check for stop loss
                elif current_high >= entry_price + self.stop_loss:
                    loss = self.stop_loss
                    self.total_profit -= loss
                    print(f"{current_date}: CLOSE SHORT at {entry_price + self.stop_loss} (stop loss hit, loss: {loss})")
                    # Update the exit information for the current trade
                    self.trades[trade_index]['exit_date'] = current_date.strftime("%Y-%m-%d %H:%M:%S")
                    self.trades[trade_index]['exit_price'] = entry_price + self.stop_loss
                    self.trades[trade_index]['profit'] = -loss
                    self.trades[trade_index]['exit_reason'] = 'stop_loss'
                    
                    self.losing_trades += 1
                    position = 0
                    trade_index = None
            
            # Check for new entry signals if we have no position
            if position == 0 and df[f'adx_signal'].iloc[i]:
                # Buy signal - enter on next bar at the highest high of lookback period
                buy_price = highest
                if next_high >= buy_price:  # Check if next bar reaches trigger price
                    position = 1
                    entry_price = buy_price
                    entry_date = next_date
                    self.trade_count += 1
                    print(f"{next_date}: OPEN LONG at {buy_price} (highest of last {self.length} bars)")
                    
                    # Log entry trade (exit details will be updated when position is closed)
                    self.add_trade(
                        entry_date=next_date,
                        exit_date=None,
                        position='LONG',
                        entry_price=buy_price,
                        exit_price=None,
                        profit=None,
                        exit_reason=None,
                        adx_value=adx_value,
                        highest_high=highest,
                        lowest_low=lowest
                    )
                    trade_index = len(self.trades) - 1  # Keep track of this trade's index
                
                # Sell signal - enter on next bar at the lowest low of lookback period
                sell_price = lowest
                if position == 0 and next_low <= sell_price:  # Check if next bar reaches trigger price
                    position = -1
                    entry_price = sell_price
                    entry_date = next_date
                    self.trade_count += 1
                    print(f"{next_date}: OPEN SHORT at {sell_price} (lowest of last {self.length} bars)")
                    
                    # Log entry trade (exit details will be updated when position is closed)
                    self.add_trade(
                        entry_date=next_date,
                        exit_date=None,
                        position='SHORT',
                        entry_price=sell_price,
                        exit_price=None,
                        profit=None,
                        exit_reason=None,
                        adx_value=adx_value,
                        highest_high=highest,
                        lowest_low=lowest
                    )
                    trade_index = len(self.trades) - 1  # Keep track of this trade's index
        
        # Print trading statistics
        self.print_statistics()
        
        return df, self.trades
    
    def print_statistics(self):
        """
        Print trading statistics
        """
        print("\nTrading Statistics:")
        print(f"Total trades: {self.trade_count}")
        print(f"Winning trades: {self.winning_trades}")
        print(f"Losing trades: {self.losing_trades}")
        if self.trade_count > 0:
            win_rate = self.winning_trades / self.trade_count * 100
            print(f"Win rate: {win_rate:.2f}%")
        print(f"Total profit/loss: {self.total_profit}")
        
        # Save trades to JSON file
        if self.trades:
            with open('trades.json', 'w') as file:
                json.dump(self.trades, file, indent=2)
            print(f"\nTrade data saved to trades.json")

if __name__ == "__main__":
    # Create strategy instance
    strategy = TradingStrategy(
        length=30,        # Lookback period for highest high/lowest low
        adx_threshold=40, # Threshold for ADX signal
        adx_window=15,    # Window period for ADX calculation
        stop_loss=250,
        profit_target=1000,
        max_bars_in_trade=20
    )
    
    # Run backtest
    file_path = "data/MNQ_price_data.json"
    result, trades = strategy.backtest(file_path)
    
    # Output summary
    print("\nData Summary:")
    print(result[['datetime', 'close', f'adx_{strategy.adx_window}', 'highest_high', 'lowest_low', 'adx_signal']].tail())
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

class OptionsData:
    """Class to load and provide access to options data"""
    
    def __init__(self, options_file):
        """Initialize with options data from file"""
        with open(options_file, 'r') as f:
            self.data = json.load(f)
        
        self.symbol = self.data['symbol']
        self.underlying_price = self.data['underlyingPrice']
        self.call_exp_map = self.data['callExpDateMap']
        self.put_exp_map = self.data['putExpDateMap']
    
    def get_expiration_dates(self):
        """Get all unique expiration dates"""
        return sorted(set(self.call_exp_map.keys()).union(set(self.put_exp_map.keys())))
    
    def get_first_expiration(self):
        """Get the first (closest) expiration date"""
        return list(self.call_exp_map.keys())[0]
    
    def filter_monthly_expirations(self):
        """Filter for monthly expirations (days 15-21)"""
        monthly_exps = []
        
        for exp_date_key in self.get_expiration_dates():
            date_parts = exp_date_key.split(':')[0]
            exp_date_obj = datetime.strptime(date_parts, '%Y-%m-%d')
            
            if 15 <= exp_date_obj.day <= 21:
                monthly_exps.append(exp_date_key)
        
        return monthly_exps


class OptionAnalyzer:
    """Class for analyzing options data"""
    
    def __init__(self, options_file):
        """Initialize with options data file path"""
        self.options_data = OptionsData(options_file)
        self.options_file = options_file
    
    def calculate_max_pain(self):
        """Calculate max pain point and pain by strike"""
        # Get the first expiration date
        exp_date = self.options_data.get_first_expiration()
        
        # Extract open interest by strike
        call_open_interest = {}
        put_open_interest = {}
        
        # Process calls
        for strike, options in self.options_data.call_exp_map[exp_date].items():
            strike_price = float(strike)
            call_open_interest[strike_price] = options[0]['openInterest']
        
        # Process puts
        for strike, options in self.options_data.put_exp_map[exp_date].items():
            strike_price = float(strike)
            put_open_interest[strike_price] = options[0]['openInterest']
        
        # Get all unique strike prices
        all_strikes = sorted(set(list(call_open_interest.keys()) + list(put_open_interest.keys())))
        
        # Calculate pain at each strike price
        pain_by_strike = {}
        
        for potential_price in all_strikes:
            total_pain = 0
            
            # Calculate call pain
            for strike, oi in call_open_interest.items():
                if potential_price > strike:
                    # For ITM calls
                    intrinsic_value = potential_price - strike
                    total_pain += intrinsic_value * oi * 100  # Each contract is for 100 shares
            
            # Calculate put pain
            for strike, oi in put_open_interest.items():
                if potential_price < strike:
                    # For ITM puts
                    intrinsic_value = strike - potential_price
                    total_pain += intrinsic_value * oi * 100  # Each contract is for 100 shares
            
            pain_by_strike[potential_price] = total_pain
        
        # Find the strike with minimum pain (max pain for option holders)
        max_pain_strike = min(pain_by_strike.items(), key=lambda x: x[1])[0]
        
        return max_pain_strike, pain_by_strike
    
    def calculate_dollar_weighted_ratio(self):
        """Calculate dollar-weighted put/call ratio for monthly expirations"""
        # Get monthly expirations
        monthly_exps = self.options_data.filter_monthly_expirations()
        
        # Variables to store results
        expiration_dates = []
        dollar_ratios = []
        call_dollar_values = []
        put_dollar_values = []
        total_dollar_values = []
        
        # Total sums across all filtered expirations
        total_call_dollar_value = 0
        total_put_dollar_value = 0
        
        # Process each monthly expiration
        for exp_date_key in sorted(monthly_exps):
            # Extract dates in a cleaner format
            date_parts = exp_date_key.split(':')[0]
            expiration_date = datetime.strptime(date_parts, '%Y-%m-%d').strftime('%Y-%m-%d')
            
            # Initialize counters for this expiration
            call_dollar_value = 0
            put_dollar_value = 0
            
            # Process calls for this expiration
            if exp_date_key in self.options_data.call_exp_map:
                for strike_price, options in self.options_data.call_exp_map[exp_date_key].items():
                    for option in options:
                        # Use mark price as the option value
                        option_price = option['mark']  
                        option_oi = option['openInterest']
                        
                        # Calculate dollar value
                        dollar_value = option_price * 100 * option_oi
                        call_dollar_value += dollar_value
            
            # Process puts for this expiration
            if exp_date_key in self.options_data.put_exp_map:
                for strike_price, options in self.options_data.put_exp_map[exp_date_key].items():
                    for option in options:
                        # Use mark price as the option value
                        option_price = option['mark']
                        option_oi = option['openInterest']
                        
                        # Calculate dollar value
                        dollar_value = option_price * 100 * option_oi
                        put_dollar_value += dollar_value
            
            # Calculate total dollar value for this expiration
            total_dollar_value = call_dollar_value + put_dollar_value
            
            # Calculate ratio (avoid division by zero)
            dollar_ratio = put_dollar_value / call_dollar_value if call_dollar_value > 0 else float('inf')
            
            # Add to totals
            total_call_dollar_value += call_dollar_value
            total_put_dollar_value += put_dollar_value
            
            # Store results
            expiration_dates.append(expiration_date)
            dollar_ratios.append(dollar_ratio)
            call_dollar_values.append(call_dollar_value)
            put_dollar_values.append(put_dollar_value)
            total_dollar_values.append(total_dollar_value)
        
        # Calculate overall values
        overall_total = total_call_dollar_value + total_put_dollar_value
        overall_dollar_ratio = total_put_dollar_value / total_call_dollar_value if total_call_dollar_value > 0 else float('inf')
        
        # Create a summary dataframe
        df = pd.DataFrame({
            'Expiration': expiration_dates,
            'Call Dollar Value': call_dollar_values,
            'Put Dollar Value': put_dollar_values,
            'Total Dollar Value': total_dollar_values,
            'Dollar-Weighted P/C Ratio': dollar_ratios
        })
        
        # Add a row for the total
        df.loc[len(df)] = ['TOTAL', total_call_dollar_value, total_put_dollar_value, 
                           overall_total, overall_dollar_ratio]
        
        return df
    
    def get_open_interest_distribution(self):
        """Get the open interest distribution for the first expiration"""
        exp_date = self.options_data.get_first_expiration()
        
        # Extract open interest by strike
        call_oi = defaultdict(int)
        put_oi = defaultdict(int)
        
        for strike, options in self.options_data.call_exp_map[exp_date].items():
            strike_price = float(strike)
            call_oi[strike_price] = options[0]['openInterest']
        
        for strike, options in self.options_data.put_exp_map[exp_date].items():
            strike_price = float(strike)
            put_oi[strike_price] = options[0]['openInterest']
        
        # Combine into a dataset
        all_strikes = sorted(set(list(call_oi.keys()) + list(put_oi.keys())))
        call_data = [call_oi.get(strike, 0) for strike in all_strikes]
        put_data = [put_oi.get(strike, 0) for strike in all_strikes]
        
        return all_strikes, call_data, put_data


class OptionsVisualizer:
    """Class for visualizing options data"""
    
    def __init__(self, analyzer):
        """Initialize with an OptionAnalyzer instance"""
        self.analyzer = analyzer
        self.symbol = analyzer.options_data.symbol
        self.underlying_price = analyzer.options_data.underlying_price
    
    def plot_max_pain(self, pain_by_strike, max_pain_strike):
        """Plot the pain at each strike price"""
        strikes = list(pain_by_strike.keys())
        pain_values = list(pain_by_strike.values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(strikes, pain_values, width=0.4)
        plt.axvline(x=max_pain_strike, color='r', linestyle='--', label=f'Max Pain: ${max_pain_strike}')
        
        plt.title(f'{self.symbol} Options Max Pain Analysis')
        plt.xlabel('Strike Price')
        plt.ylabel('Pain Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format y-axis with dollar signs and commas
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(f'outputs/{self.symbol}_max_pain.png')
        plt.show()
    
    def plot_open_interest(self):
        """Plot the open interest distribution"""
        all_strikes, call_data, put_data = self.analyzer.get_open_interest_distribution()
        
        plt.figure(figsize=(14, 7))
        
        # Plot stacked bar chart
        plt.bar(all_strikes, call_data, label='Calls', alpha=0.7, color='green')
        plt.bar(all_strikes, put_data, bottom=call_data, label='Puts', alpha=0.7, color='red')
        
        plt.title(f'{self.symbol} Option Open Interest by Strike Price')
        plt.xlabel('Strike Price ($)')
        plt.ylabel('Open Interest (# of contracts)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f'outputs/{self.symbol}_open_interest.png')
        plt.show()
    
    def plot_dollar_weighted_ratio(self, df):
        """Plot the dollar-weighted put/call ratio by expiration"""
        if len(df) <= 1:  # Only the total row, no data to plot
            print("No matching expirations to plot (days 15-21)")
            return
        
        # Create a copy without the total row for plotting
        plot_df = df.iloc[:-1].copy()
        
        plt.figure(figsize=(14, 10))
        
        # Create subplot for the ratio
        plt.subplot(2, 1, 1)
        plt.plot(plot_df['Expiration'], plot_df['Dollar-Weighted P/C Ratio'], 'ro-', linewidth=2, markersize=8)
        
        # Add horizontal line at ratio = 1.0
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Equal Put/Call Balance')
        
        # Set labels and title
        plt.xlabel('Expiration Date')
        plt.ylabel('Put/Call Ratio')
        plt.title(f'{self.symbol} Dollar-Weighted Put/Call Ratio by Expiration (Current Price: ${self.underlying_price})')
        
        # Format x-axis
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Create subplot for the dollar values
        plt.subplot(2, 1, 2)
        
        # Bar positions
        x = range(len(plot_df['Expiration']))
        width = 0.35
        
        # Create bars
        plt.bar([i - width/2 for i in x], plot_df['Call Dollar Value'], width, label='Call Dollar Value', color='green', alpha=0.7)
        plt.bar([i + width/2 for i in x], plot_df['Put Dollar Value'], width, label='Put Dollar Value', color='red', alpha=0.7)
        
        # Set labels
        plt.xlabel('Expiration Date')
        plt.ylabel('Dollar Value')
        plt.title(f'{self.symbol} Option Dollar Values by Expiration (Days 15-21)')
        
        # Format axes
        plt.xticks(x, plot_df['Expiration'], rotation=45)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'outputs/{self.symbol}_dollar_weighted_pc_analysis.png')
        plt.show()


class OptionsReport:
    """Class for generating options analysis reports"""
    
    def __init__(self, analyzer, visualizer):
        """Initialize with an analyzer and visualizer"""
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.symbol = analyzer.options_data.symbol
        self.underlying_price = analyzer.options_data.underlying_price
    
    def display_max_pain_results(self, max_pain_strike, pain_by_strike):
        """Display max pain analysis results"""
        print(f"Max Pain Strike: ${max_pain_strike}")
        
        # Display the pain values near max pain
        strikes = sorted(pain_by_strike.keys())
        max_pain_idx = strikes.index(max_pain_strike)
        start_idx = max(0, max_pain_idx - 3)
        end_idx = min(len(strikes), max_pain_idx + 4)
        
        print("\nPain values around max pain:")
        print(f"{'Strike':^10} | {'Pain Value':^15}")
        print("-" * 28)
        
        for i in range(start_idx, end_idx):
            strike = strikes[i]
            pain = pain_by_strike[strike]
            marker = " (MAX PAIN)" if strike == max_pain_strike else ""
            print(f"${strike:^8.1f} | ${pain:^13,.0f}{marker}")
    
    def display_dollar_weighted_report(self, df):
        """Display dollar-weighted analysis results"""
        if len(df) <= 1:  # Only the total row, no data
            print("No options expiring on days 15-21 of any month were found in the data.")
            return
        
        # Format dollar values for display
        display_df = df.copy()
        display_df['Call Dollar Value'] = display_df['Call Dollar Value'].map('${:,.2f}'.format)
        display_df['Put Dollar Value'] = display_df['Put Dollar Value'].map('${:,.2f}'.format)
        display_df['Total Dollar Value'] = display_df['Total Dollar Value'].map('${:,.2f}'.format)
        display_df['Dollar-Weighted P/C Ratio'] = display_df['Dollar-Weighted P/C Ratio'].map('{:.4f}'.format)
        
        # Print results
        print(f"\n{self.symbol} Dollar-Weighted Put/Call Analysis (Current Price: ${self.underlying_price})")
        print(f"Filtered for expirations on days 15-21 of the month")
        print("="*100)
        print(display_df.to_string(index=False))
        
        # Export to CSV
        csv_filename = f"outputs/{self.symbol}_dollar_weighted_pc_ratio_monthly.csv"
        df.to_csv(csv_filename, index=False)
    
    def generate_full_report(self):
        """Generate a full options analysis report"""
        print(f"Analyzing options data for {self.symbol}...")
        
        # Calculate max pain
        max_pain_strike, pain_by_strike = self.analyzer.calculate_max_pain()
        self.display_max_pain_results(max_pain_strike, pain_by_strike)
        self.visualizer.plot_max_pain(pain_by_strike, max_pain_strike)
        
        # Open interest analysis
        self.visualizer.plot_open_interest()
        
        # Dollar-weighted analysis
        df = self.analyzer.calculate_dollar_weighted_ratio()
        self.display_dollar_weighted_report(df)
        self.visualizer.plot_dollar_weighted_ratio(df)


def main():
    """Main function to run the options analysis"""
    options_file = 'data/LUNR_options_data.json'
    
    # Create analyzer and visualizer
    analyzer = OptionAnalyzer(options_file)
    visualizer = OptionsVisualizer(analyzer)
    
    # Generate report
    report = OptionsReport(analyzer, visualizer)
    report.generate_full_report()


if __name__ == "__main__":
    main()
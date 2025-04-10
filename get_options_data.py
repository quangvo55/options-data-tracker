from utils.schwab_auth import get_options_data
from analyze_options_data import main as analyze_options_data
import json

tickers = ["GME"]

for ticker in tickers:
    options_data = get_options_data(ticker)
    with open(f'data/{ticker}_options_data.json', 'w') as f:
        json.dump(options_data, f, indent=2)
    print(f"Saved data for {ticker}")
    analyze_options_data(ticker)
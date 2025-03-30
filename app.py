from flask import Flask, render_template, send_from_directory, redirect, url_for
import os
import re

app = Flask(__name__)

# Configuration
OUTPUT_DIR = 'outputs'  # Path to your output directory containing the images

# Excluded file patterns
EXCLUDED_PATTERNS = [
    r'.*_dollar_weighted_pc_analysis\.png$'  # Exclude dollar_weighted_pc_analysis.png files
]

# Chart type definitions
CHART_TYPES = {
    'max_pain': {
        'title': 'Max Pain Analysis',
        'description': 'Shows the pain value in dollars across different strike prices. The vertical line represents the max pain point.'
    },
    'open_interest': {
        'title': 'Option Open Interest by Strike Price',
        'description': 'Displays the number of contracts by strike price, separated into calls (green) and puts (red).'
    },
    'dollar_weighted_pc_analysis': {
        'title': 'Dollar-Weighted Put/Call Ratio Analysis',
        'description': 'Top chart shows the put/call ratio across different expiration dates. Bottom chart shows the actual dollar values of calls and puts for each expiration date.'
    },
    'dollar_weighted_pc_ratio_monthly': {
        'title': 'Dollar-Weighted Put/Call Ratio by Month',
        'description': 'Displays the dollar-weighted put/call ratio across different expiration months, showing the relative value of puts versus calls.'
    }
}

def is_excluded(filename):
    """Check if file should be excluded based on patterns"""
    for pattern in EXCLUDED_PATTERNS:
        if re.match(pattern, filename):
            return True
    return False

def extract_ticker_and_chart_type(filename):
    """Extract ticker symbol and chart type from filename"""
    # For PNG files
    if filename.endswith('.png'):
        pattern = r'([A-Z]+)_(.+)\.png'
        match = re.match(pattern, filename)
        if match:
            ticker = match.group(1)
            chart_type = match.group(2)
            return ticker, chart_type
    
    # For CSV files
    elif filename.endswith('.csv'):
        pattern = r'([A-Z]+)_(.+)\.csv'
        match = re.match(pattern, filename)
        if match:
            ticker = match.group(1)
            chart_type = match.group(2)
            return ticker, chart_type
            
    return None, None

def get_ticker_charts():
    """Get all charts organized by ticker symbol"""
    # Get all image and CSV files from the output directory
    files = [f for f in os.listdir(OUTPUT_DIR) 
              if f.endswith(('.png', '.jpg', '.jpeg', '.csv')) and not is_excluded(f)]
    
    # Organize files by ticker
    tickers = {}
    for filename in files:
        ticker, chart_type = extract_ticker_and_chart_type(filename)
        if ticker and chart_type:
            if ticker not in tickers:
                tickers[ticker] = []
            
            # Get chart description from chart types or use defaults
            chart_info = CHART_TYPES.get(chart_type, {
                'title': chart_type.replace('_', ' ').title(),
                'description': 'Options analysis chart'
            })
            
            is_csv = filename.endswith('.csv')
            
            tickers[ticker].append({
                'filename': filename,
                'chart_type': chart_type,
                'title': f"{ticker} {chart_info['title']}",
                'description': chart_info['description'],
                'is_csv': is_csv
            })
            
    # Sort tickers alphabetically
    return {ticker: sorted(charts, key=lambda x: x['chart_type']) 
            for ticker, charts in sorted(tickers.items())}

@app.route('/')
def index():
    """Main page showing all tickers"""
    tickers = get_ticker_charts()
    return render_template('index.html', tickers=tickers)

@app.route('/ticker/<ticker>')
def view_ticker(ticker):
    """View all charts for a specific ticker"""
    tickers = get_ticker_charts()
    if ticker not in tickers:
        return redirect(url_for('index'))
    
    return render_template('ticker.html', 
                          ticker=ticker,
                          charts=tickers[ticker],
                          tickers=sorted(tickers.keys()))

@app.route('/image/<filename>')
def serve_image(filename):
    """Serve image files from the output directory"""
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/view/<filename>')
def view_image(filename):
    """View a single image or CSV visualization with its description"""
    ticker, chart_type = extract_ticker_and_chart_type(filename)
    if not ticker or not chart_type:
        return redirect(url_for('index'))
    
    tickers = get_ticker_charts()
    if ticker not in tickers:
        return redirect(url_for('index'))
    
    # Find current chart
    current_chart = None
    for chart in tickers[ticker]:
        if chart['filename'] == filename:
            current_chart = chart
            break
    
    if not current_chart:
        return redirect(url_for('ticker', ticker=ticker))
    
    # Get previous and next charts within the same ticker
    charts = tickers[ticker]
    current_index = charts.index(current_chart)
    prev_chart = charts[current_index - 1]['filename'] if current_index > 0 else None
    next_chart = charts[current_index + 1]['filename'] if current_index < len(charts) - 1 else None
    
    # Get chart info
    chart_info = CHART_TYPES.get(chart_type, {
        'title': chart_type.replace('_', ' ').title(),
        'description': 'Options analysis chart'
    })
    
    # Check if this is a CSV file that needs to be visualized
    is_csv = current_chart.get('is_csv', False)
    csv_data = None
    
    if is_csv:
        try:
            with open(os.path.join(OUTPUT_DIR, filename), 'r') as csv_file:
                csv_data = csv_file.read()
        except Exception as e:
            print(f"Error reading CSV: {e}")
    
    return render_template('view.html', 
                          ticker=ticker,
                          image_url=f'/image/{filename}' if not is_csv else None,
                          title=f"{ticker} {chart_info['title']}",
                          description=chart_info['description'],
                          prev=prev_chart,
                          next=next_chart,
                          all_tickers=sorted(tickers.keys()),
                          is_csv=is_csv,
                          csv_data=csv_data,
                          filename=filename)

if __name__ == '__main__':
    # Instructions for file setup
    print("="*80)
    print("Options Analysis Visualization Dashboard")
    print("="*80)
    print("\nThis application automatically organizes your option analysis charts by ticker.")
    print("\nExpected file naming convention:")
    print("- TICKER_chart_type.png (e.g., GME_max_pain.png, VXX_open_interest.png)")
    print("\nExcluded file patterns:")
    for pattern in EXCLUDED_PATTERNS:
        print(f"- {pattern}")
    
    print("\nSupported chart types:")
    for chart_type, info in CHART_TYPES.items():
        print(f"- {chart_type}: {info['title']}")
    
    # Detect found tickers
    tickers = get_ticker_charts()
    if tickers:
        print("\nDetected tickers:")
        for ticker, charts in tickers.items():
            print(f"- {ticker}: {len(charts)} charts")
    else:
        print("\nNo properly named chart files found in the output directory.")
        print("Please ensure your files follow the naming convention: TICKER_chart_type.png")
    
    print("\nStarting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)
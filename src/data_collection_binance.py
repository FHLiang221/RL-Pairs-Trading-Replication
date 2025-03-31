import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("binance_data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

def fetch_klines(symbol, interval, start_time=None, end_time=None, limit=1000):
    """
    Fetch kline/candlestick data directly from Binance.US API
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '1m', '1h', '1d')
        start_time: Optional start time in milliseconds
        end_time: Optional end time in milliseconds
        limit: Maximum number of klines to return (default 1000, max 1000)
        
    Returns:
        List of klines data
    """
    url = "https://api.binance.us/api/v3/klines"
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"API Error: {response.status_code} - {response.text}")
        return []

def collect_historical_data(symbol, interval, start_date, end_date):
    """
    Collect historical kline data for a specific date range
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '1m', '1h', '1d')
        start_date: Start date as datetime
        end_date: End date as datetime
        
    Returns:
        DataFrame with historical data
    """
    logger.info(f"Collecting {symbol} {interval} data from {start_date} to {end_date}")
    
    # Convert dates to millisecond timestamps
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    all_klines = []
    current_start = start_ms
    
    # Calculate time delta based on interval to avoid requesting too much data at once
    if interval == '1m':
        # For 1m interval, fetch in chunks of 1000 minutes (about 16.7 hours)
        time_chunk = 1000 * 60 * 1000  # 1000 minutes in milliseconds
    elif interval == '3m':
        time_chunk = 1000 * 3 * 60 * 1000  # 3000 minutes
    elif interval == '5m':
        time_chunk = 1000 * 5 * 60 * 1000  # 5000 minutes
    elif interval == '15m':
        time_chunk = 1000 * 15 * 60 * 1000  # 15000 minutes
    elif interval == '30m':
        time_chunk = 1000 * 30 * 60 * 1000  # 30000 minutes
    elif interval == '1h':
        time_chunk = 1000 * 60 * 60 * 1000  # 1000 hours
    elif interval == '2h':
        time_chunk = 1000 * 2 * 60 * 60 * 1000  # 2000 hours
    elif interval == '4h':
        time_chunk = 1000 * 4 * 60 * 60 * 1000  # 4000 hours
    elif interval == '6h':
        time_chunk = 1000 * 6 * 60 * 60 * 1000  # 6000 hours
    elif interval == '8h':
        time_chunk = 1000 * 8 * 60 * 60 * 1000  # 8000 hours
    elif interval == '12h':
        time_chunk = 1000 * 12 * 60 * 60 * 1000  # 12000 hours
    elif interval == '1d':
        time_chunk = 1000 * 24 * 60 * 60 * 1000  # 1000 days
    elif interval == '3d':
        time_chunk = 1000 * 3 * 24 * 60 * 60 * 1000  # 3000 days
    elif interval == '1w':
        time_chunk = 1000 * 7 * 24 * 60 * 60 * 1000  # 1000 weeks
    elif interval == '1M':
        time_chunk = 1000 * 30 * 24 * 60 * 60 * 1000  # 1000 months
    else:
        # Default to 1 day chunks
        time_chunk = 24 * 60 * 60 * 1000  # 1 day in milliseconds
    
    # Fetch data in chunks to respect the 1000 limit
    while current_start < end_ms:
        current_end = min(current_start + time_chunk, end_ms)
        
        logger.info(f"Fetching chunk from {datetime.fromtimestamp(current_start/1000)} to {datetime.fromtimestamp(current_end/1000)}")
        
        chunk_klines = fetch_klines(
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=current_end,
            limit=1000
        )
        
        if not chunk_klines:
            logger.warning(f"No data received for this chunk. Moving to next time period.")
            current_start = current_end + 1
            continue
        
        all_klines.extend(chunk_klines)
        logger.info(f"Retrieved {len(chunk_klines)} klines")
        
        # Update start time for next chunk (add 1ms to avoid duplicates)
        if len(chunk_klines) > 0:
            # Use the last kline's open time + 1 as the new start time
            current_start = int(chunk_klines[-1][0]) + 1
        else:
            # If no klines, move to next chunk
            current_start = current_end + 1
        
        # Add delay to avoid rate limits
        time.sleep(0.5)
    
    if not all_klines:
        logger.error(f"No data retrieved for {symbol} {interval}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    
    # Convert numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert timestamps
    df['datetime'] = pd.to_datetime(df['open_time'].astype(float), unit='ms')
    
    logger.info(f"Collected {len(df)} records for {symbol} {interval}")
    
    return df

if __name__ == "__main__":
    # Define date range
    start_date = datetime(2023, 10, 1)
    end_date = datetime(2023, 12, 31)
    
    # List of symbols and intervals to fetch
    symbols_intervals = [
        ('BTCUSDT', '1m'),
        ('BTCUSDT', '1d'),
        ('ETHUSDT', '1m'),
        ('ETHUSDT', '1d')
    ]
    
    for symbol, interval in symbols_intervals:
        logger.info(f"Collecting data for {symbol} {interval}...")
        
        df = collect_historical_data(symbol, interval, start_date, end_date)
        
        if not df.empty:
            # Save raw data
            output_file = f"data/{symbol.lower()}_{interval}_raw.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} records to {output_file}")
        else:
            logger.error(f"Failed to collect data for {symbol} {interval}")
    
    logger.info("Data collection complete!")
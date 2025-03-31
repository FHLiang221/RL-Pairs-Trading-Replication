import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("synthetic_data_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

def generate_synthetic_btc_data(start_date, end_date, interval_minutes=1, base_price=30000, volatility=0.02):
    """
    Generate synthetic Bitcoin price data with realistic properties
    
    Args:
        start_date: Start date as datetime
        end_date: End date as datetime
        interval_minutes: Time interval in minutes
        base_price: Starting price in USD
        volatility: Daily volatility parameter
        
    Returns:
        DataFrame with synthetic BTC price data
    """
    logger.info(f"Generating synthetic BTC data from {start_date} to {end_date}...")
    
    # Calculate total minutes in the date range
    total_minutes = int((end_date - start_date).total_seconds() / 60)
    
    # Create date range with the specified interval
    date_range = [start_date + timedelta(minutes=i*interval_minutes) for i in range(total_minutes//interval_minutes + 1)]
    
    # Generate synthetic price series using a geometric Brownian motion
    # This is a common model for asset prices
    np.random.seed(42)  # For reproducibility
    
    # Parameters based on BTC behavior
    drift = 0.0001  # Small positive drift
    
    # Generate returns
    daily_volatility = volatility / np.sqrt(1440/interval_minutes)  # Scale volatility by time interval
    returns = np.random.normal(drift, daily_volatility, len(date_range))
    
    # Add some jumps and drops to simulate BTC behavior
    jumps = np.random.choice([0, 1], size=len(date_range), p=[0.995, 0.005])
    jump_sizes = np.random.normal(0.03, 0.01, len(date_range)) * jumps
    returns = returns + jump_sizes
    
    # Generate price series
    price_series = base_price * np.cumprod(1 + returns)
    
    # Create OHLC data
    high_low_spread = 0.01  # Typical high-low spread as a fraction of close price
    open_close_spread = 0.005  # Typical open-close spread as a fraction of close price
    
    close_prices = price_series
    high_prices = close_prices * (1 + np.random.uniform(0, high_low_spread, len(date_range)))
    low_prices = close_prices * (1 - np.random.uniform(0, high_low_spread, len(date_range)))
    open_prices = close_prices * (1 + np.random.normal(0, open_close_spread, len(date_range)))
    
    # Ensure high is always highest and low is always lowest
    for i in range(len(date_range)):
        max_val = max(open_prices[i], close_prices[i], high_prices[i])
        min_val = min(open_prices[i], close_prices[i], low_prices[i])
        high_prices[i] = max(high_prices[i], max_val)
        low_prices[i] = min(low_prices[i], min_val)
    
    # Generate realistic volume data
    volume_base = np.random.lognormal(12, 1, len(date_range))  
    
    # Volume tends to increase with volatility
    volume_scaling = 1 + 5 * np.abs(returns)  
    volumes = volume_base * volume_scaling
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': date_range,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    logger.info(f"Generated {len(df)} synthetic BTC price data points")
    
    return df

def create_synthetic_eur_gbp_from_usd(btc_usd_df, eur_rate=0.92, gbp_rate=0.79, rate_volatility=0.01):
    """
    Create synthetic BTC-EUR and BTC-GBP data from BTC-USD data
    
    Args:
        btc_usd_df: DataFrame with synthetic BTC-USD data
        eur_rate: Base EUR/USD exchange rate
        gbp_rate: Base GBP/USD exchange rate
        rate_volatility: Volatility of exchange rates
        
    Returns:
        Tuple with (BTC-EUR DataFrame, BTC-GBP DataFrame)
    """
    logger.info("Creating synthetic BTC-EUR and BTC-GBP data...")
    
    # Create copies for EUR and GBP
    btc_eur_df = btc_usd_df.copy()
    btc_gbp_df = btc_usd_df.copy()
    
    # Generate daily exchange rates with realistic time-varying properties
    # Group by date to ensure consistent rates within a day
    btc_usd_df['date'] = btc_usd_df['datetime'].dt.date
    unique_dates = btc_usd_df['date'].unique()
    
    np.random.seed(43)  # Different seed for exchange rates
    
    # Generate daily exchange rates with some autocorrelation
    n_days = len(unique_dates)
    
    # Start with base rates
    eur_rates = np.ones(n_days) * eur_rate
    gbp_rates = np.ones(n_days) * gbp_rate
    
    # Add some autocorrelated noise
    noise_eur = np.zeros(n_days)
    noise_gbp = np.zeros(n_days)
    
    # AR(1) process for exchange rate variations
    ar_param = 0.8  # High autocorrelation
    for i in range(1, n_days):
        noise_eur[i] = ar_param * noise_eur[i-1] + np.random.normal(0, rate_volatility * (1-ar_param))
        noise_gbp[i] = ar_param * noise_gbp[i-1] + np.random.normal(0, rate_volatility * (1-ar_param))
    
    # Apply noise to rates
    eur_rates = eur_rates * (1 + noise_eur)
    gbp_rates = gbp_rates * (1 + noise_gbp)
    
    # Create exchange rate lookup by date
    date_to_eur_rate = {date: rate for date, rate in zip(unique_dates, eur_rates)}
    date_to_gbp_rate = {date: rate for date, rate in zip(unique_dates, gbp_rates)}
    
    # Apply conversions with lookup
    for price_col in ['open', 'high', 'low', 'close']:
        btc_eur_df[price_col] = btc_eur_df.apply(
            lambda row: row[price_col] * date_to_eur_rate[row['date']], 
            axis=1
        )
        
        btc_gbp_df[price_col] = btc_gbp_df.apply(
            lambda row: row[price_col] * date_to_gbp_rate[row['date']], 
            axis=1
        )
    
    # Clean up date column
    btc_eur_df.drop('date', axis=1, inplace=True)
    btc_gbp_df.drop('date', axis=1, inplace=True)
    
    logger.info(f"Created synthetic BTC-EUR and BTC-GBP data with {len(btc_eur_df)} records each")
    
    return btc_eur_df, btc_gbp_df

def create_pair_trading_dataset(btc_eur_df, btc_gbp_df):
    """
    Create the pair trading dataset as used in the paper
    
    Args:
        btc_eur_df: DataFrame with BTC-EUR data
        btc_gbp_df: DataFrame with BTC-GBP data
        
    Returns:
        DataFrame with paired asset prices
    """
    logger.info("Creating pair trading dataset...")
    
    # Set datetime as index
    btc_eur_df.set_index('datetime', inplace=True)
    btc_gbp_df.set_index('datetime', inplace=True)
    
    # Create paired dataset
    paired_data = pd.DataFrame({
        'asset1_price': btc_eur_df['close'],
        'asset2_price': btc_gbp_df['close']
    })
    
    logger.info(f"Created paired dataset with {len(paired_data)} records")
    
    return paired_data

if __name__ == "__main__":
    # Define date range as in the paper
    start_date = datetime(2023, 10, 1)
    end_date = datetime(2023, 12, 31)
    
    # Generate 1-minute synthetic BTC-USD data
    btc_usd_df = generate_synthetic_btc_data(start_date, end_date, interval_minutes=1)
    
    # Save the BTC-USD data
    btc_usd_df.to_csv('data/btcusd_1m_SYNTHETIC.csv', index=False)
    logger.info("Saved synthetic BTC-USD data")
    
    # Create synthetic BTC-EUR and BTC-GBP data
    btc_eur_df, btc_gbp_df = create_synthetic_eur_gbp_from_usd(btc_usd_df)
    
    # Save the BTC-EUR and BTC-GBP data
    btc_eur_df.to_csv('data/btceur_1m_SYNTHETIC.csv', index=False)
    btc_gbp_df.to_csv('data/btcgbp_1m_SYNTHETIC.csv', index=False)
    logger.info("Saved synthetic BTC-EUR and BTC-GBP data")
    
    # Create the pair trading dataset
    paired_data = create_pair_trading_dataset(btc_eur_df, btc_gbp_df)
    
    # Save the paired dataset
    paired_data.to_csv('data/btceur_btcgbp_paired_1m_SYNTHETIC.csv')
    logger.info("Saved paired dataset")
    
    # Split into training and testing sets as in the paper
    training_cutoff = datetime(2023, 12, 1)
    
    training_data = paired_data[paired_data.index < training_cutoff].copy()
    testing_data = paired_data[paired_data.index >= training_cutoff].copy()
    
    # Reset indices to include datetime as a column
    training_data = training_data.reset_index()
    testing_data = testing_data.reset_index()
    
    # Save training and testing datasets
    training_data.to_csv('data/training_data_1m_SYNTHETIC.csv', index=False)
    logger.info(f"Saved training data with {len(training_data)} records")
    
    testing_data.to_csv('data/testing_data_1m_SYNTHETIC.csv', index=False)
    logger.info(f"Saved testing data with {len(testing_data)} records")
    
    # Check if we have enough data points
    expected_records = 263520  # As mentioned in the paper
    actual_records = len(paired_data)
    
    logger.info(f"Generated {actual_records} records out of {expected_records} expected")
    
    # Print statistics to verify the data has the right properties
    logger.info("\nData statistics:")
    logger.info(f"BTC-EUR mean price: {paired_data['asset1_price'].mean():.2f}")
    logger.info(f"BTC-GBP mean price: {paired_data['asset2_price'].mean():.2f}")
    logger.info(f"Correlation between assets: {paired_data['asset1_price'].corr(paired_data['asset2_price']):.4f}")
    
    # Check cointegration
    score, pvalue, _ = coint(paired_data['asset1_price'], paired_data['asset2_price'])
    logger.info(f"Cointegration test p-value: {pvalue:.4f} (lower is better)")
    
    logger.info("\nSynthetic data generation complete!")
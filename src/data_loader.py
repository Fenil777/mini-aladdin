"""Data loader for fetching historical price data."""

from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf

from src.config_loader import Config


def fetch_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch historical adjusted close prices for given tickers.
    
    Args:
        tickers: List of ticker symbols.
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        cache_path: Optional path to cache/load data as CSV.
        
    Returns:
        DataFrame with dates as index and tickers as columns,
        containing adjusted close prices.
        
    Raises:
        ValueError: If no data is returned for any ticker.
    """
    # Try to load from cache if path provided
    if cache_path and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df
    
    print(f"Downloading data for {len(tickers)} tickers...")
    
    # Download data using yfinance
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=True,
    )
    
    # Extract Adjusted Close prices
    if isinstance(data.columns, pd.MultiIndex):
        # Multiple tickers: MultiIndex columns
        prices = data["Adj Close"]
    else:
        # Single ticker: simple columns
        prices = data[["Adj Close"]]
        prices.columns = tickers
    
    # Validate we have data
    if prices.empty:
        raise ValueError("No price data returned. Check tickers and date range.")
    
    # Check for missing tickers
    missing_tickers = set(tickers) - set(prices.columns)
    if missing_tickers:
        print(f"Warning: No data for tickers: {missing_tickers}")
    
    # Drop rows with any NaN values (non-trading days alignment)
    prices = prices.dropna()
    
    if prices.empty:
        raise ValueError("No overlapping data for all tickers after cleaning.")
    
    # Cache if path provided
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)
        print(f"Cached data to {cache_path}")
    
    print(f"Loaded {len(prices)} trading days of data for {len(prices.columns)} tickers")
    
    return prices


def load_prices_from_config(
    config: Config,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load prices using configuration object.
    
    Args:
        config: Configuration object.
        use_cache: Whether to use/create cache file.
        
    Returns:
        DataFrame of adjusted close prices.
    """
    cache_path = None
    if use_cache:
        cache_path = Path("data/raw/prices.csv")
    
    return fetch_prices(
        tickers=config.data.tickers,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        cache_path=cache_path,
    )


if __name__ == "__main__":
    # Quick test
    from src.config_loader import load_config
    
    config = load_config()
    prices = load_prices_from_config(config)
    print("\nPrice data sample:")
    print(prices.head())
    print(f"\nShape: {prices.shape}")

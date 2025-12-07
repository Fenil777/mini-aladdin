"""
Mini Aladdin - Portfolio Optimization & Risk Engine

CLI entry point for running portfolio analysis.
"""

from src.config_loader import load_config
from src.data_loader import load_prices_from_config


def main():
    """Main entry point."""
    print("=" * 60)
    print("Mini Aladdin - Portfolio Optimization & Risk Engine")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/2] Loading configuration...")
    config = load_config()
    print(f"  Tickers: {config.data.tickers}")
    print(f"  Date range: {config.data.start_date} to {config.data.end_date}")
    print(f"  Risk-free rate: {config.portfolio.risk_free_rate:.2%}")
    
    # Load price data
    print("\n[2/2] Loading price data...")
    prices = load_prices_from_config(config)
    print(f"  Loaded {len(prices)} days of data for {len(prices.columns)} tickers")
    
    print("\n" + "=" * 60)
    print("Milestone 1 Complete - Data layer operational!")
    print("=" * 60)
    
    # TODO: Milestone 2+ will add returns, optimization, etc.


if __name__ == "__main__":
    main()

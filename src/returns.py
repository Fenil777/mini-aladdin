"""Returns computation module for Mini Aladdin."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from price data."""
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple (arithmetic) returns from price data."""
    simple_returns = prices.pct_change()
    return simple_returns.dropna()


def annualized_mean_returns(returns: pd.DataFrame) -> pd.Series:
    """Compute annualized mean returns from daily returns."""
    daily_mean = returns.mean()
    return daily_mean * TRADING_DAYS_PER_YEAR


def annualized_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute annualized covariance matrix from daily returns."""
    daily_cov = returns.cov()
    return daily_cov * TRADING_DAYS_PER_YEAR


def annualized_volatility(returns: pd.DataFrame) -> pd.Series:
    """Compute annualized volatility (std dev) for each asset."""
    daily_std = returns.std()
    return daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix from returns."""
    return returns.corr()

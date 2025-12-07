"""Risk metrics module for Mini Aladdin."""

import numpy as np
import pandas as pd
from typing import Union


def portfolio_return(weights: np.ndarray, mean_returns: Union[np.ndarray, pd.Series]) -> float:
    """Compute expected portfolio return."""
    weights = np.array(weights)
    mean_returns = np.array(mean_returns)
    return float(np.dot(weights, mean_returns))


def portfolio_volatility(weights: np.ndarray, cov_matrix: Union[np.ndarray, pd.DataFrame]) -> float:
    """Compute portfolio volatility (standard deviation)."""
    weights = np.array(weights)
    cov_matrix = np.array(cov_matrix)
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return float(np.sqrt(variance))


def portfolio_sharpe_ratio(portfolio_ret: float, portfolio_vol: float, risk_free_rate: float) -> float:
    """Compute portfolio Sharpe ratio."""
    if portfolio_vol == 0:
        return 0.0
    return (portfolio_ret - risk_free_rate) / portfolio_vol


def compute_portfolio_returns_series(weights: np.ndarray, asset_returns: pd.DataFrame) -> pd.Series:
    """Compute portfolio return time series from asset returns and weights."""
    weights = np.array(weights)
    return asset_returns.dot(weights)


def compute_var(returns: Union[pd.Series, np.ndarray], confidence: float = 0.95) -> float:
    """Compute historical Value-at-Risk (VaR). Returns positive number representing loss."""
    returns = np.array(returns)
    var = -np.percentile(returns, (1 - confidence) * 100)
    return float(var)


def compute_cvar(returns: Union[pd.Series, np.ndarray], confidence: float = 0.95) -> float:
    """Compute historical CVaR (Expected Shortfall). Returns positive number."""
    returns = np.array(returns)
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    cvar = -np.mean(returns[returns <= var_threshold])
    return float(cvar)


def compute_max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    """Compute maximum drawdown from returns series."""
    returns = np.array(returns)
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    return float(np.max(drawdowns))

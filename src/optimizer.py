"""Portfolio optimization module for Mini Aladdin."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Union
from dataclasses import dataclass

from src.risk_metrics import portfolio_return, portfolio_volatility, portfolio_sharpe_ratio


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float


def minimize_variance(
    mean_returns: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    risk_free_rate: float = 0.04,
) -> OptimizationResult:
    """Find the minimum variance portfolio (long-only)."""
    mean_returns = np.array(mean_returns)
    cov_matrix = np.array(cov_matrix)
    n_assets = len(mean_returns)
    
    def objective(weights):
        return portfolio_volatility(weights, cov_matrix) ** 2
    
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    x0 = np.ones(n_assets) / n_assets
    
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    
    weights = result.x
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    sharpe = portfolio_sharpe_ratio(ret, vol, risk_free_rate)
    
    return OptimizationResult(weights=weights, expected_return=ret, volatility=vol, sharpe_ratio=sharpe)


def maximize_sharpe(
    mean_returns: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    risk_free_rate: float = 0.04,
) -> OptimizationResult:
    """Find the maximum Sharpe ratio portfolio (long-only)."""
    mean_returns = np.array(mean_returns)
    cov_matrix = np.array(cov_matrix)
    n_assets = len(mean_returns)
    
    def objective(weights):
        ret = portfolio_return(weights, mean_returns)
        vol = portfolio_volatility(weights, cov_matrix)
        if vol == 0:
            return 0
        return -(ret - risk_free_rate) / vol
    
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    x0 = np.ones(n_assets) / n_assets
    
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    
    weights = result.x
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    sharpe = portfolio_sharpe_ratio(ret, vol, risk_free_rate)
    
    return OptimizationResult(weights=weights, expected_return=ret, volatility=vol, sharpe_ratio=sharpe)


def efficient_frontier(
    mean_returns: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    risk_free_rate: float = 0.04,
    n_points: int = 50,
) -> List[OptimizationResult]:
    """Compute the efficient frontier."""
    mean_returns = np.array(mean_returns)
    cov_matrix = np.array(cov_matrix)
    n_assets = len(mean_returns)
    
    min_var_result = minimize_variance(mean_returns, cov_matrix, risk_free_rate)
    max_sharpe_result = maximize_sharpe(mean_returns, cov_matrix, risk_free_rate)
    
    min_ret = min(min_var_result.expected_return, mean_returns.min())
    max_ret = max(max_sharpe_result.expected_return, mean_returns.max())
    
    target_returns = np.linspace(min_ret, max_ret, n_points)
    frontier = []
    
    for target_ret in target_returns:
        def objective(weights):
            return portfolio_volatility(weights, cov_matrix) ** 2
        
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target_ret: portfolio_return(w, mean_returns) - t},
        ]
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            if result.success:
                weights = result.x
                ret = portfolio_return(weights, mean_returns)
                vol = portfolio_volatility(weights, cov_matrix)
                sharpe = portfolio_sharpe_ratio(ret, vol, risk_free_rate)
                frontier.append(OptimizationResult(weights=weights, expected_return=ret, volatility=vol, sharpe_ratio=sharpe))
        except Exception:
            continue
    
    return frontier

"""Monte Carlo portfolio simulation module for Mini Aladdin."""

import numpy as np
import pandas as pd
from typing import Union

from src.risk_metrics import portfolio_return, portfolio_volatility, portfolio_sharpe_ratio


def simulate_random_portfolios(
    n_portfolios: int,
    mean_returns: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    risk_free_rate: float = 0.04,
    seed: int = None,
) -> pd.DataFrame:
    """Simulate random long-only portfolios using Dirichlet distribution."""
    if seed is not None:
        np.random.seed(seed)
    
    mean_returns_arr = np.array(mean_returns)
    cov_matrix_arr = np.array(cov_matrix)
    n_assets = len(mean_returns_arr)
    
    if hasattr(mean_returns, 'index'):
        asset_names = list(mean_returns.index)
    else:
        asset_names = [f"Asset_{i}" for i in range(n_assets)]
    
    results = []
    for _ in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        ret = portfolio_return(weights, mean_returns_arr)
        vol = portfolio_volatility(weights, cov_matrix_arr)
        sharpe = portfolio_sharpe_ratio(ret, vol, risk_free_rate)
        
        result = {"Return": ret, "Volatility": vol, "Sharpe": sharpe}
        for i, name in enumerate(asset_names):
            result[f"Weight_{name}"] = weights[i]
        results.append(result)
    
    return pd.DataFrame(results)


def get_simulation_statistics(sim_results: pd.DataFrame) -> dict:
    """Get summary statistics from simulation results."""
    return {
        "n_portfolios": len(sim_results),
        "return_mean": sim_results["Return"].mean(),
        "return_std": sim_results["Return"].std(),
        "return_min": sim_results["Return"].min(),
        "return_max": sim_results["Return"].max(),
        "volatility_mean": sim_results["Volatility"].mean(),
        "volatility_std": sim_results["Volatility"].std(),
        "volatility_min": sim_results["Volatility"].min(),
        "volatility_max": sim_results["Volatility"].max(),
        "sharpe_mean": sim_results["Sharpe"].mean(),
        "sharpe_std": sim_results["Sharpe"].std(),
        "sharpe_min": sim_results["Sharpe"].min(),
        "sharpe_max": sim_results["Sharpe"].max(),
        "best_sharpe_idx": sim_results["Sharpe"].idxmax(),
    }

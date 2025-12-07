"""Pipeline orchestration module for Mini Aladdin."""

from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np

from src.config_loader import Config, load_config
from src.data_loader import load_prices_from_config
from src.returns import (
    compute_log_returns, annualized_mean_returns, annualized_covariance_matrix,
    annualized_volatility, compute_correlation_matrix,
)
from src.risk_metrics import compute_portfolio_returns_series, compute_var, compute_cvar, compute_max_drawdown
from src.optimizer import minimize_variance, maximize_sharpe, efficient_frontier, OptimizationResult
from src.simulator import simulate_random_portfolios, get_simulation_statistics


@dataclass
class PortfolioAnalysis:
    """Analysis results for a single portfolio."""
    name: str
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name, "expected_return": self.expected_return,
            "volatility": self.volatility, "sharpe_ratio": self.sharpe_ratio,
            "var_95": self.var_95, "cvar_95": self.cvar_95, "max_drawdown": self.max_drawdown,
        }


@dataclass
class AnalysisResult:
    """Complete analysis results container."""
    config: Config
    prices: pd.DataFrame
    returns: pd.DataFrame
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    volatilities: pd.Series
    correlation_matrix: pd.DataFrame
    min_variance_portfolio: PortfolioAnalysis
    max_sharpe_portfolio: PortfolioAnalysis
    efficient_frontier: List[OptimizationResult]
    simulation_results: pd.DataFrame
    simulation_stats: Dict
    asset_names: List[str]


def analyze_portfolio(name, weights, mean_returns, cov_matrix, returns, risk_free_rate, var_confidence):
    """Analyze a single portfolio with all risk metrics."""
    from src.risk_metrics import portfolio_return, portfolio_volatility, portfolio_sharpe_ratio
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    sharpe = portfolio_sharpe_ratio(ret, vol, risk_free_rate)
    port_returns = compute_portfolio_returns_series(weights, returns)
    var = compute_var(port_returns, var_confidence)
    cvar = compute_cvar(port_returns, var_confidence)
    mdd = compute_max_drawdown(port_returns)
    return PortfolioAnalysis(name=name, weights=weights, expected_return=ret, volatility=vol,
                             sharpe_ratio=sharpe, var_95=var, cvar_95=cvar, max_drawdown=mdd)


def run_full_analysis(config_path: str = "config/config.yaml") -> AnalysisResult:
    """Run the complete portfolio analysis pipeline."""
    config = load_config(config_path)
    prices = load_prices_from_config(config, use_cache=True)
    returns = compute_log_returns(prices)
    mean_returns = annualized_mean_returns(returns)
    cov_matrix = annualized_covariance_matrix(returns)
    volatilities = annualized_volatility(returns)
    correlation_matrix = compute_correlation_matrix(returns)
    asset_names = list(prices.columns)
    rf = config.portfolio.risk_free_rate
    var_conf = config.risk.var_confidence
    
    min_var_opt = minimize_variance(mean_returns, cov_matrix, rf)
    max_sharpe_opt = maximize_sharpe(mean_returns, cov_matrix, rf)
    min_var_analysis = analyze_portfolio("Minimum Variance", min_var_opt.weights, mean_returns, cov_matrix, returns, rf, var_conf)
    max_sharpe_analysis = analyze_portfolio("Maximum Sharpe", max_sharpe_opt.weights, mean_returns, cov_matrix, returns, rf, var_conf)
    frontier = efficient_frontier(mean_returns, cov_matrix, rf, n_points=50)
    sim_results = simulate_random_portfolios(config.simulation.n_portfolios, mean_returns, cov_matrix, rf, seed=42)
    sim_stats = get_simulation_statistics(sim_results)
    
    return AnalysisResult(config=config, prices=prices, returns=returns, mean_returns=mean_returns,
        cov_matrix=cov_matrix, volatilities=volatilities, correlation_matrix=correlation_matrix,
        min_variance_portfolio=min_var_analysis, max_sharpe_portfolio=max_sharpe_analysis,
        efficient_frontier=frontier, simulation_results=sim_results, simulation_stats=sim_stats, asset_names=asset_names)

"""Tests for optimizer module."""

import numpy as np
import pytest
from src.optimizer import minimize_variance, maximize_sharpe, efficient_frontier, OptimizationResult


@pytest.fixture
def simple_assets():
    mean_returns = np.array([0.10, 0.15])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.09]])
    return mean_returns, cov_matrix


class TestMinimizeVariance:
    def test_weights_sum_to_one(self, simple_assets):
        mean_ret, cov_mat = simple_assets
        result = minimize_variance(mean_ret, cov_mat)
        assert np.isclose(np.sum(result.weights), 1.0)
    
    def test_weights_non_negative(self, simple_assets):
        mean_ret, cov_mat = simple_assets
        result = minimize_variance(mean_ret, cov_mat)
        assert np.all(result.weights >= -1e-10)


class TestMaximizeSharpe:
    def test_weights_sum_to_one(self, simple_assets):
        mean_ret, cov_mat = simple_assets
        result = maximize_sharpe(mean_ret, cov_mat)
        assert np.isclose(np.sum(result.weights), 1.0)
    
    def test_weights_non_negative(self, simple_assets):
        mean_ret, cov_mat = simple_assets
        result = maximize_sharpe(mean_ret, cov_mat)
        assert np.all(result.weights >= -1e-10)
    
    def test_sharpe_higher_than_equal_weight(self, simple_assets):
        mean_ret, cov_mat = simple_assets
        rf = 0.04
        result = maximize_sharpe(mean_ret, cov_mat, rf)
        
        from src.risk_metrics import portfolio_return, portfolio_volatility, portfolio_sharpe_ratio
        eq_weights = np.ones(len(mean_ret)) / len(mean_ret)
        eq_ret = portfolio_return(eq_weights, mean_ret)
        eq_vol = portfolio_volatility(eq_weights, cov_mat)
        eq_sharpe = portfolio_sharpe_ratio(eq_ret, eq_vol, rf)
        
        assert result.sharpe_ratio >= eq_sharpe - 1e-6


class TestEfficientFrontier:
    def test_returns_list(self, simple_assets):
        mean_ret, cov_mat = simple_assets
        frontier = efficient_frontier(mean_ret, cov_mat, n_points=10)
        assert isinstance(frontier, list)
        assert len(frontier) > 0

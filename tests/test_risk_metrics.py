"""Tests for risk metrics module."""

import numpy as np
import pandas as pd
import pytest
from src.risk_metrics import (
    portfolio_return, portfolio_volatility, portfolio_sharpe_ratio,
    compute_var, compute_cvar, compute_max_drawdown,
)


@pytest.fixture
def sample_returns_series():
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0004, 0.01, 252))


class TestPortfolioReturn:
    def test_portfolio_return_calculation(self):
        weights = np.array([0.5, 0.5])
        returns = np.array([0.10, 0.20])
        ret = portfolio_return(weights, returns)
        assert np.isclose(ret, 0.15)


class TestPortfolioVolatility:
    def test_portfolio_volatility_positive(self):
        weights = np.array([0.5, 0.5])
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])
        vol = portfolio_volatility(weights, cov)
        assert vol > 0

    def test_portfolio_volatility_single_asset(self):
        weights = np.array([1.0])
        cov = np.array([[0.04]])
        vol = portfolio_volatility(weights, cov)
        assert np.isclose(vol, 0.2)


class TestSharpeRatio:
    def test_sharpe_ratio_calculation(self):
        ret, vol, rf = 0.12, 0.20, 0.04
        sharpe = portfolio_sharpe_ratio(ret, vol, rf)
        expected = (0.12 - 0.04) / 0.20
        assert np.isclose(sharpe, expected)

    def test_sharpe_ratio_zero_vol(self):
        sharpe = portfolio_sharpe_ratio(0.10, 0.0, 0.04)
        assert sharpe == 0.0


class TestVaR:
    def test_var_positive(self, sample_returns_series):
        var = compute_var(sample_returns_series, 0.95)
        assert var > 0

    def test_var_confidence_ordering(self, sample_returns_series):
        var_90 = compute_var(sample_returns_series, 0.90)
        var_95 = compute_var(sample_returns_series, 0.95)
        var_99 = compute_var(sample_returns_series, 0.99)
        assert var_90 <= var_95 <= var_99


class TestCVaR:
    def test_cvar_greater_than_var(self, sample_returns_series):
        var = compute_var(sample_returns_series, 0.95)
        cvar = compute_cvar(sample_returns_series, 0.95)
        assert cvar >= var


class TestMaxDrawdown:
    def test_max_drawdown_bounds(self, sample_returns_series):
        mdd = compute_max_drawdown(sample_returns_series)
        assert 0 <= mdd <= 1

"""Tests for returns module."""

import numpy as np
import pandas as pd
import pytest
from src.returns import (
    compute_log_returns, compute_simple_returns, annualized_mean_returns,
    annualized_covariance_matrix, annualized_volatility, compute_correlation_matrix,
    TRADING_DAYS_PER_YEAR,
)


@pytest.fixture
def sample_prices():
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    data = {
        "AAPL": [100, 101, 102, 101, 103, 104, 103, 105, 106, 107],
        "MSFT": [200, 202, 201, 203, 205, 204, 206, 208, 207, 210],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_returns(sample_prices):
    return compute_log_returns(sample_prices)


class TestLogReturns:
    def test_log_returns_shape(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        assert len(returns) == len(sample_prices) - 1

    def test_log_returns_values(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        expected_first = np.log(101 / 100)
        assert np.isclose(returns["AAPL"].iloc[0], expected_first)

    def test_log_returns_no_nan(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        assert not returns.isna().any().any()


class TestAnnualization:
    def test_annualized_mean_returns(self, sample_returns):
        ann_returns = annualized_mean_returns(sample_returns)
        daily_mean = sample_returns.mean()
        expected = daily_mean * TRADING_DAYS_PER_YEAR
        pd.testing.assert_series_equal(ann_returns, expected)

    def test_annualized_volatility(self, sample_returns):
        ann_vol = annualized_volatility(sample_returns)
        daily_std = sample_returns.std()
        expected = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        pd.testing.assert_series_equal(ann_vol, expected)

    def test_volatility_positive(self, sample_returns):
        ann_vol = annualized_volatility(sample_returns)
        assert (ann_vol > 0).all()


class TestCorrelation:
    def test_correlation_diagonal(self, sample_returns):
        corr = compute_correlation_matrix(sample_returns)
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(len(corr)))

    def test_correlation_bounds(self, sample_returns):
        corr = compute_correlation_matrix(sample_returns)
        assert (corr >= -1).all().all()
        assert (corr <= 1).all().all()

"""Tests for simulator module."""

import numpy as np
import pandas as pd
import pytest
from src.simulator import simulate_random_portfolios, get_simulation_statistics


@pytest.fixture
def test_inputs():
    mean_returns = np.array([0.10, 0.12, 0.08])
    cov_matrix = np.array([[0.04, 0.01, 0.005], [0.01, 0.0625, 0.01], [0.005, 0.01, 0.0225]])
    return mean_returns, cov_matrix


class TestSimulateRandomPortfolios:
    def test_correct_number_of_portfolios(self, test_inputs):
        mean_ret, cov_mat = test_inputs
        results = simulate_random_portfolios(500, mean_ret, cov_mat, seed=42)
        assert len(results) == 500
    
    def test_weights_sum_to_one(self, test_inputs):
        mean_ret, cov_mat = test_inputs
        results = simulate_random_portfolios(100, mean_ret, cov_mat, seed=42)
        weight_cols = [c for c in results.columns if c.startswith("Weight_")]
        weight_sums = results[weight_cols].sum(axis=1)
        assert np.allclose(weight_sums, 1.0)
    
    def test_reproducibility_with_seed(self, test_inputs):
        mean_ret, cov_mat = test_inputs
        results1 = simulate_random_portfolios(50, mean_ret, cov_mat, seed=123)
        results2 = simulate_random_portfolios(50, mean_ret, cov_mat, seed=123)
        pd.testing.assert_frame_equal(results1, results2)


class TestGetSimulationStatistics:
    def test_returns_dict(self, test_inputs):
        mean_ret, cov_mat = test_inputs
        results = simulate_random_portfolios(100, mean_ret, cov_mat, seed=42)
        stats = get_simulation_statistics(results)
        assert isinstance(stats, dict)
        assert "sharpe_max" in stats

"""Tests for pipeline module."""

import pytest
import numpy as np
import pandas as pd
from src.pipeline import PortfolioAnalysis, AnalysisResult, analyze_portfolio, run_full_analysis


class TestPortfolioAnalysis:
    """Test PortfolioAnalysis dataclass."""
    
    def test_to_dict(self):
        """Test to_dict method."""
        weights = np.array([0.5, 0.3, 0.2])
        portfolio = PortfolioAnalysis(
            name="Test Portfolio",
            weights=weights,
            expected_return=0.10,
            volatility=0.15,
            sharpe_ratio=0.5,
            var_95=0.02,
            cvar_95=0.03,
            max_drawdown=0.10,
        )
        
        result = portfolio.to_dict()
        
        assert result["name"] == "Test Portfolio"
        assert result["expected_return"] == 0.10
        assert result["volatility"] == 0.15
        assert result["sharpe_ratio"] == 0.5
        assert result["var_95"] == 0.02
        assert result["cvar_95"] == 0.03
        assert result["max_drawdown"] == 0.10


class TestRunFullAnalysis:
    """Test run_full_analysis function."""
    
    def test_returns_analysis_result(self):
        """Test that run_full_analysis returns an AnalysisResult object."""
        result = run_full_analysis()
        
        assert isinstance(result, AnalysisResult)
        assert hasattr(result, 'config')
        assert hasattr(result, 'prices')
        assert hasattr(result, 'returns')
        assert hasattr(result, 'mean_returns')
        assert hasattr(result, 'cov_matrix')
        assert hasattr(result, 'volatilities')
        assert hasattr(result, 'correlation_matrix')
        assert hasattr(result, 'min_variance_portfolio')
        assert hasattr(result, 'max_sharpe_portfolio')
        assert hasattr(result, 'efficient_frontier')
        assert hasattr(result, 'simulation_results')
        assert hasattr(result, 'simulation_stats')
        assert hasattr(result, 'asset_names')
    
    def test_portfolio_analysis_structure(self):
        """Test that portfolio analyses have correct structure."""
        result = run_full_analysis()
        
        assert isinstance(result.min_variance_portfolio, PortfolioAnalysis)
        assert isinstance(result.max_sharpe_portfolio, PortfolioAnalysis)
        
        # Check weights sum to 1
        assert np.isclose(np.sum(result.min_variance_portfolio.weights), 1.0)
        assert np.isclose(np.sum(result.max_sharpe_portfolio.weights), 1.0)
        
        # Check weights are non-negative
        assert np.all(result.min_variance_portfolio.weights >= 0)
        assert np.all(result.max_sharpe_portfolio.weights >= 0)
    
    def test_efficient_frontier_list(self):
        """Test that efficient frontier is a list of OptimizationResult objects."""
        result = run_full_analysis()
        
        assert isinstance(result.efficient_frontier, list)
        assert len(result.efficient_frontier) > 0
        
        # Check that each point has expected attributes
        for point in result.efficient_frontier:
            assert hasattr(point, 'weights')
            assert hasattr(point, 'expected_return')
            assert hasattr(point, 'volatility')
            assert hasattr(point, 'sharpe_ratio')
    
    def test_simulation_results_dataframe(self):
        """Test that simulation results is a DataFrame with expected columns."""
        result = run_full_analysis()
        
        assert isinstance(result.simulation_results, pd.DataFrame)
        assert 'Return' in result.simulation_results.columns
        assert 'Volatility' in result.simulation_results.columns
        assert 'Sharpe' in result.simulation_results.columns
        assert len(result.simulation_results) == result.config.simulation.n_portfolios

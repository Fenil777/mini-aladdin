"""Tests for pipeline module."""
import pytest
import numpy as np
from src.pipeline import PortfolioAnalysis

class TestPortfolioAnalysis:
    def test_to_dict(self):
        pa = PortfolioAnalysis(name="Test", weights=np.array([0.5, 0.5]), expected_return=0.10,
            volatility=0.15, sharpe_ratio=0.4, var_95=0.02, cvar_95=0.03, max_drawdown=0.10)
        d = pa.to_dict()
        assert d["name"] == "Test"
        assert d["expected_return"] == 0.10

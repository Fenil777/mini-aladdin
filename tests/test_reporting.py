"""Tests for reporting module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from src.pipeline import run_full_analysis
from src.reporting import (
    plot_efficient_frontier,
    plot_weights_bar,
    plot_correlation_heatmap,
    save_summary_csv,
    print_analysis_summary,
)


class TestPlotEfficientFrontier:
    """Test plot_efficient_frontier function."""
    
    def test_returns_figure(self):
        """Test that plot_efficient_frontier returns a plotly figure."""
        analysis = run_full_analysis()
        fig = plot_efficient_frontier(analysis)
        
        # Check it's a plotly figure
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        assert len(fig.data) > 0
    
    def test_with_simulation(self):
        """Test plot with simulation results."""
        analysis = run_full_analysis()
        fig = plot_efficient_frontier(analysis, show_simulation=True)
        
        # Should have traces for simulation, frontier, portfolios, and assets
        assert len(fig.data) >= 4
    
    def test_without_simulation(self):
        """Test plot without simulation results."""
        analysis = run_full_analysis()
        fig = plot_efficient_frontier(analysis, show_simulation=False)
        
        # Should have traces for frontier, portfolios, and assets (no simulation)
        assert len(fig.data) >= 3


class TestPlotWeightsBar:
    """Test plot_weights_bar function."""
    
    def test_returns_figure(self):
        """Test that plot_weights_bar returns a plotly figure."""
        analysis = run_full_analysis()
        fig = plot_weights_bar(analysis.min_variance_portfolio, analysis.asset_names)
        
        # Check it's a plotly figure
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        assert len(fig.data) > 0
    
    def test_custom_title(self):
        """Test with custom title."""
        analysis = run_full_analysis()
        custom_title = "My Custom Portfolio"
        fig = plot_weights_bar(
            analysis.min_variance_portfolio, 
            analysis.asset_names,
            title=custom_title
        )
        
        assert custom_title in fig.layout.title.text


class TestPlotCorrelationHeatmap:
    """Test plot_correlation_heatmap function."""
    
    def test_returns_figure(self):
        """Test that plot_correlation_heatmap returns a plotly figure."""
        analysis = run_full_analysis()
        fig = plot_correlation_heatmap(analysis)
        
        # Check it's a plotly figure
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        assert len(fig.data) > 0


class TestSaveSummaryCsv:
    """Test save_summary_csv function."""
    
    def test_saves_file(self):
        """Test that CSV file is created."""
        analysis = run_full_analysis()
        
        # Use temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_summary.csv"
            save_summary_csv(analysis, str(output_path))
            
            # Check file exists
            assert output_path.exists()
            
            # Read and verify content
            df = pd.read_csv(output_path)
            assert len(df) == 2  # Two portfolios
            assert "name" in df.columns
            assert "expected_return" in df.columns
            assert "volatility" in df.columns
            assert "sharpe_ratio" in df.columns
            
            # Check portfolio names
            assert "Minimum Variance" in df["name"].values
            assert "Maximum Sharpe" in df["name"].values
            
            # Check weights columns exist
            for asset_name in analysis.asset_names:
                assert f"weight_{asset_name}" in df.columns


class TestPrintAnalysisSummary:
    """Test print_analysis_summary function."""
    
    def test_runs_without_error(self, capsys):
        """Test that print_analysis_summary runs without error."""
        analysis = run_full_analysis()
        print_analysis_summary(analysis)
        
        # Capture output
        captured = capsys.readouterr()
        
        # Check key information is in output
        assert "PORTFOLIO ANALYSIS SUMMARY" in captured.out
        assert "Minimum Variance Portfolio" in captured.out
        assert "Maximum Sharpe Portfolio" in captured.out
        assert "Return:" in captured.out
        assert "Vol:" in captured.out
        assert "Sharpe:" in captured.out
        assert "VaR:" in captured.out
        assert "CVaR:" in captured.out
        assert "MaxDD:" in captured.out

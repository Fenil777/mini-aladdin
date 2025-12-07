"""Reporting module for Mini Aladdin with Plotly visualizations."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import List, Optional
from src.pipeline import AnalysisResult, PortfolioAnalysis


def plot_efficient_frontier(analysis: AnalysisResult, show_simulation: bool = True) -> go.Figure:
    """Create efficient frontier plot."""
    fig = go.Figure()
    
    if show_simulation:
        sim_marker = dict(
            size=4,
            color=analysis.simulation_results["Sharpe"],
            colorscale="Viridis",
            colorbar=dict(title="Sharpe"),
            opacity=0.6,
        )
        fig.add_trace(go.Scatter(
            x=analysis.simulation_results["Volatility"],
            y=analysis.simulation_results["Return"],
            mode="markers",
            marker=sim_marker,
            name="Random Portfolios",
        ))
    
    frontier_vols = [p.volatility for p in analysis.efficient_frontier]
    frontier_rets = [p.expected_return for p in analysis.efficient_frontier]
    fig.add_trace(go.Scatter(
        x=frontier_vols,
        y=frontier_rets,
        mode="lines",
        line=dict(color="red", width=3),
        name="Efficient Frontier",
    ))
    
    # Add optimal portfolios
    fig.add_trace(go.Scatter(
        x=[analysis.min_variance_portfolio.volatility],
        y=[analysis.min_variance_portfolio.expected_return],
        mode="markers",
        marker=dict(size=15, color="blue", symbol="star"),
        name="Min Variance",
    ))
    fig.add_trace(go.Scatter(
        x=[analysis.max_sharpe_portfolio.volatility],
        y=[analysis.max_sharpe_portfolio.expected_return],
        mode="markers",
        marker=dict(size=15, color="green", symbol="star"),
        name="Max Sharpe",
    ))
    
    # Add individual assets
    fig.add_trace(go.Scatter(
        x=list(analysis.volatilities),
        y=list(analysis.mean_returns),
        mode="markers+text",
        marker=dict(size=10, color="orange"),
        text=analysis.asset_names,
        textposition="top center",
        name="Assets",
    ))
    
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
    )
    return fig


def plot_weights_bar(portfolio: PortfolioAnalysis, asset_names: List[str], title: Optional[str] = None) -> go.Figure:
    """Create bar chart of portfolio weights."""
    title = title or f"{portfolio.name} Portfolio Weights"
    fig = go.Figure(data=[go.Bar(x=asset_names, y=portfolio.weights, text=[f"{w:.1%}" for w in portfolio.weights], textposition="outside")])
    fig.update_layout(title=title, xaxis_title="Asset", yaxis_title="Weight", yaxis_tickformat=".0%")
    return fig


def plot_correlation_heatmap(analysis: AnalysisResult) -> go.Figure:
    """Create correlation matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(z=analysis.correlation_matrix.values, x=analysis.asset_names, y=analysis.asset_names,
        colorscale="RdBu_r", zmid=0, text=np.round(analysis.correlation_matrix.values, 2), texttemplate="%{text}"))
    fig.update_layout(title="Asset Correlation Matrix")
    return fig


def save_summary_csv(analysis: AnalysisResult, output_path: str = "data/processed/summary.csv"):
    """Save analysis summary to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = []
    for portfolio in [analysis.min_variance_portfolio, analysis.max_sharpe_portfolio]:
        row = portfolio.to_dict()
        for i, name in enumerate(analysis.asset_names):
            row[f"weight_{name}"] = portfolio.weights[i]
        summary_data.append(row)
    df = pd.DataFrame(summary_data)
    df.to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


def print_analysis_summary(analysis: AnalysisResult):
    """Print formatted analysis summary."""
    print("\n" + "=" * 60)
    print("PORTFOLIO ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nData: {len(analysis.prices)} days, {len(analysis.asset_names)} assets")
    print(f"Risk-free rate: {analysis.config.portfolio.risk_free_rate:.2%}")
    for portfolio in [analysis.min_variance_portfolio, analysis.max_sharpe_portfolio]:
        print(f"\n{portfolio.name} Portfolio:")
        print(f"  Return: {portfolio.expected_return:.2%}, Vol: {portfolio.volatility:.2%}, Sharpe: {portfolio.sharpe_ratio:.3f}")
        print(f"  VaR: {portfolio.var_95:.2%}, CVaR: {portfolio.cvar_95:.2%}, MaxDD: {portfolio.max_drawdown:.2%}")
        print("  Weights:", {n: f"{w:.1%}" for n, w in zip(analysis.asset_names, portfolio.weights) if w > 0.01})
    print("=" * 60)

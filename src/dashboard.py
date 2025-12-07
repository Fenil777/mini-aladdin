"""Streamlit dashboard for Mini Aladdin."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Mini Aladdin", page_icon="🏰", layout="wide")

from src.returns import compute_log_returns, annualized_mean_returns, annualized_covariance_matrix, annualized_volatility, compute_correlation_matrix
from src.risk_metrics import portfolio_return, portfolio_volatility, portfolio_sharpe_ratio, compute_portfolio_returns_series, compute_var, compute_cvar, compute_max_drawdown
from src.optimizer import minimize_variance, maximize_sharpe, efficient_frontier
from src.simulator import simulate_random_portfolios

@st.cache_data
def load_data(tickers, start_date, end_date):
    import yfinance as yf
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"]
    else:
        prices = data[["Adj Close"]]
        prices.columns = tickers if isinstance(tickers, list) else [tickers]
    return prices.dropna()

def main():
    st.title("🏰 Mini Aladdin")
    st.markdown("**Portfolio Optimization & Risk Engine**")
    
    st.sidebar.header("⚙️ Configuration")
    tickers_input = st.sidebar.text_area("Tickers", value="AAPL, MSFT, GOOGL, AMZN, JPM, GLD, TLT")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", value=pd.to_datetime("2022-01-01"))
    end_date = col2.date_input("End", value=pd.to_datetime("2024-12-31"))
    
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.25) / 100
    n_portfolios = st.sidebar.slider("MC Portfolios", 500, 10000, 5000, 500)
    var_conf = st.sidebar.slider("VaR Confidence (%)", 90, 99, 95) / 100
    
    if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Running analysis..."):
            prices = load_data(tickers, str(start_date), str(end_date))
            if prices.empty:
                st.error("No data. Check tickers/dates.")
                return
            
            returns = compute_log_returns(prices)
            mean_ret = annualized_mean_returns(returns)
            cov_mat = annualized_covariance_matrix(returns)
            vols = annualized_volatility(returns)
            corr = compute_correlation_matrix(returns)
            
            min_var = minimize_variance(mean_ret, cov_mat, risk_free_rate)
            max_sharpe = maximize_sharpe(mean_ret, cov_mat, risk_free_rate)
            frontier = efficient_frontier(mean_ret, cov_mat, risk_free_rate, 50)
            sim = simulate_random_portfolios(n_portfolios, mean_ret, cov_mat, risk_free_rate, 42)
            
            st.session_state.data = {
                "prices": prices, "returns": returns, "mean_ret": mean_ret,
                "cov_mat": cov_mat, "vols": vols, "corr": corr,
                "min_var": min_var, "max_sharpe": max_sharpe,
                "frontier": frontier, "sim": sim, "tickers": list(prices.columns),
                "rf": risk_free_rate, "var_conf": var_conf
            }
            st.success("Done!")
    
    if "data" in st.session_state:
        d = st.session_state.data
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Frontier", "⚖️ Weights", "📊 Risk", "🔗 Corr", "📉 Assets"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d["sim"]["Volatility"], y=d["sim"]["Return"], mode="markers",
                marker=dict(size=4, color=d["sim"]["Sharpe"], colorscale="Viridis", opacity=0.6, colorbar=dict(title="Sharpe")), name="Random"))
            fig.add_trace(go.Scatter(x=[p.volatility for p in d["frontier"]], y=[p.expected_return for p in d["frontier"]],
                mode="lines", line=dict(color="red", width=3), name="Frontier"))
            fig.add_trace(go.Scatter(x=[d["min_var"].volatility], y=[d["min_var"].expected_return],
                mode="markers", marker=dict(size=16, color="blue", symbol="star"), name="Min Var"))
            fig.add_trace(go.Scatter(x=[d["max_sharpe"].volatility], y=[d["max_sharpe"].expected_return],
                mode="markers", marker=dict(size=16, color="green", symbol="star"), name="Max Sharpe"))
            fig.add_trace(go.Scatter(x=list(d["vols"]), y=list(d["mean_ret"]), mode="markers+text",
                marker=dict(size=10, color="orange"), text=d["tickers"], textposition="top center", name="Assets"))
            fig.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%", height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Min Variance**")
                fig = go.Figure(go.Bar(x=d["tickers"], y=d["min_var"].weights, text=[f"{w:.1%}" for w in d["min_var"].weights], textposition="outside"))
                fig.update_layout(yaxis_tickformat=".0%", height=400)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("**Max Sharpe**")
                fig = go.Figure(go.Bar(x=d["tickers"], y=d["max_sharpe"].weights, text=[f"{w:.1%}" for w in d["max_sharpe"].weights], textposition="outside"))
                fig.update_layout(yaxis_tickformat=".0%", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            mv_ret = compute_portfolio_returns_series(d["min_var"].weights, d["returns"])
            ms_ret = compute_portfolio_returns_series(d["max_sharpe"].weights, d["returns"])
            metrics = pd.DataFrame({
                "Metric": ["Return", "Volatility", "Sharpe", "VaR", "CVaR", "MaxDD"],
                "Min Var": [f"{d['min_var'].expected_return:.2%}", f"{d['min_var'].volatility:.2%}", f"{d['min_var'].sharpe_ratio:.3f}",
                    f"{compute_var(mv_ret, d['var_conf']):.2%}", f"{compute_cvar(mv_ret, d['var_conf']):.2%}", f"{compute_max_drawdown(mv_ret):.2%}"],
                "Max Sharpe": [f"{d['max_sharpe'].expected_return:.2%}", f"{d['max_sharpe'].volatility:.2%}", f"{d['max_sharpe'].sharpe_ratio:.3f}",
                    f"{compute_var(ms_ret, d['var_conf']):.2%}", f"{compute_cvar(ms_ret, d['var_conf']):.2%}", f"{compute_max_drawdown(ms_ret):.2%}"]
            })
            st.table(metrics.set_index("Metric"))
        
        with tab4:
            fig = go.Figure(go.Heatmap(z=d["corr"].values, x=d["tickers"], y=d["tickers"], colorscale="RdBu_r", zmid=0,
                text=np.round(d["corr"].values, 2), texttemplate="%{text}"))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            stats = pd.DataFrame({"Ticker": d["tickers"], "Return": [f"{r:.2%}" for r in d["mean_ret"]], "Vol": [f"{v:.2%}" for v in d["vols"]]})
            st.dataframe(stats.set_index("Ticker"), use_container_width=True)
            norm = d["prices"] / d["prices"].iloc[0] * 100
            st.plotly_chart(px.line(norm, title="Normalized Prices"), use_container_width=True)
    else:
        st.info("👈 Configure and click Run Analysis!")

if __name__ == "__main__":
    main()

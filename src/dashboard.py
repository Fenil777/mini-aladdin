"""Enhanced Streamlit dashboard for Mini Aladdin with modern dark theme."""

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
from plotly.subplots import make_subplots

st.set_page_config(page_title="Mini Aladdin", page_icon="🏰", layout="wide")

from src.returns import compute_log_returns, annualized_mean_returns, annualized_covariance_matrix, annualized_volatility, compute_correlation_matrix
from src.risk_metrics import portfolio_return, portfolio_volatility, portfolio_sharpe_ratio, compute_portfolio_returns_series, compute_var, compute_cvar, compute_max_drawdown
from src.optimizer import minimize_variance, maximize_sharpe, efficient_frontier
from src.simulator import simulate_random_portfolios

# Custom CSS for modern dark theme
def inject_custom_css():
    st.markdown("""
    <style>
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(233, 69, 96, 0.3);
    }
    
    /* Metric cards with glassmorphism */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(233, 69, 96, 0.4);
        border-color: rgba(233, 69, 96, 0.5);
    }
    
    div[data-testid="stMetric"] label {
        color: #e94560 !important;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Styled tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px 24px;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(233, 69, 96, 0.2);
        border-color: rgba(233, 69, 96, 0.5);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e94560 0%, #d63447 100%);
        border-color: #e94560;
    }
    
    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(135deg, #e94560 0%, #d63447 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.5);
    }
    
    /* Gradient divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e94560, transparent);
        margin: 2rem 0;
    }
    
    /* Hero title styling */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e94560 0%, #ff6b9d 50%, #ffa06a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 20px rgba(233, 69, 96, 0.3);
    }
    
    .hero-subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateX(10px);
        border-color: rgba(233, 69, 96, 0.5);
        box-shadow: 0 8px 32px rgba(233, 69, 96, 0.2);
    }
    
    .feature-card h3 {
        color: #e94560;
        font-size: 1.3rem;
        margin-bottom: 8px;
    }
    
    .feature-card p {
        color: rgba(255, 255, 255, 0.7);
        line-height: 1.6;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 4rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.5);
    }
    
    .footer a {
        color: #e94560;
        text-decoration: none;
        font-weight: 600;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #e94560 0%, #ff6b9d 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Preset portfolios
PRESET_PORTFOLIOS = {
    "Custom": [],
    "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "Diversified": ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "VNQ"],
    "Conservative": ["BND", "TLT", "GLD", "VNQ", "JNJ", "PG", "KO"],
    "Aggressive": ["TSLA", "NVDA", "AMD", "COIN", "SQ", "ARKK"]
}

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

def create_donut_chart(weights, tickers, title):
    """Create a donut chart for portfolio weights."""
    non_zero_mask = weights > 0.001
    filtered_weights = weights[non_zero_mask]
    filtered_tickers = [tickers[i] for i in range(len(tickers)) if non_zero_mask[i]]
    
    fig = go.Figure(data=[go.Pie(
        labels=filtered_tickers,
        values=filtered_weights,
        hole=0.4,
        marker=dict(
            colors=px.colors.sequential.Plasma,
            line=dict(color='#000000', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.2%}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='white'), x=0.5, xanchor='center'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            bgcolor='rgba(255,255,255,0.05)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1
        )
    )
    return fig

def create_gauge_chart(value, title, range_max, color):
    """Create a gauge chart for risk metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 16, 'color': 'white'}},
        number={'suffix': '%', 'font': {'size': 32, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, range_max * 100], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'borderwidth': 2,
            'bordercolor': 'rgba(255,255,255,0.2)',
            'steps': [
                {'range': [0, range_max * 33.33], 'color': 'rgba(0,255,0,0.2)'},
                {'range': [range_max * 33.33, range_max * 66.67], 'color': 'rgba(255,255,0,0.2)'},
                {'range': [range_max * 66.67, range_max * 100], 'color': 'rgba(255,0,0,0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=300
    )
    return fig

def create_dark_theme_figure():
    """Return common layout settings for dark theme."""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,50,0.3)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            color='white'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            color='white'
        )
    )

def create_monthly_returns_heatmap(returns, tickers):
    """Create a monthly returns heatmap for assets."""
    monthly_returns = returns.resample('M').sum()
    
    if len(tickers) > 0:
        asset = tickers[0]
        monthly_data = monthly_returns[asset]
        
        monthly_data.index = pd.to_datetime(monthly_data.index)
        df_heatmap = pd.DataFrame({
            'Year': monthly_data.index.year,
            'Month': monthly_data.index.month,
            'Return': monthly_data.values
        })
        
        pivot = df_heatmap.pivot(index='Month', columns='Year', values='Return')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot.values * 100, 2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title='Return', ticksuffix='%')
        ))
        
        fig.update_layout(
            title=dict(text=f'Monthly Returns Heatmap - {asset}', font=dict(size=18, color='white')),
            **create_dark_theme_figure(),
            height=400
        )
        
        return fig
    return None

def main():
    inject_custom_css()
    
    # Hero Section
    st.markdown('<h1 class="hero-title">🏰 Mini Aladdin</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Portfolio Optimization & Risk Engine • Modern Analytics Platform</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar Configuration
    st.sidebar.markdown("### ⚙️ Configuration")
    
    preset_choice = st.sidebar.selectbox(
        "Portfolio Preset",
        options=list(PRESET_PORTFOLIOS.keys()),
        help="Select a preset portfolio or choose Custom to enter your own tickers"
    )
    
    if preset_choice == "Custom":
        tickers_input = st.sidebar.text_area(
            "Tickers (comma-separated)",
            value="AAPL, MSFT, GOOGL, AMZN, JPM, GLD, TLT",
            help="Enter ticker symbols separated by commas"
        )
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    else:
        tickers = PRESET_PORTFOLIOS[preset_choice]
        st.sidebar.info(f"**{preset_choice}**: {', '.join(tickers)}")
    
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
    end_date = col2.date_input("End Date", value=pd.to_datetime("2024-12-31"))
    
    st.sidebar.markdown("### 📊 Parameters")
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.25) / 100
    n_portfolios = st.sidebar.slider("Monte Carlo Portfolios", 500, 10000, 5000, 500)
    var_conf = st.sidebar.slider("VaR Confidence (%)", 90, 99, 95) / 100
    
    if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
        with st.spinner("🔄 Fetching data..."):
            progress_bar = st.progress(0)
            
            try:
                prices = load_data(tickers, str(start_date), str(end_date))
                progress_bar.progress(20)
                
                if prices.empty:
                    st.error("❌ No data retrieved. Please check your tickers and date range.")
                    return
                
                returns = compute_log_returns(prices)
                mean_ret = annualized_mean_returns(returns)
                cov_mat = annualized_covariance_matrix(returns)
                vols = annualized_volatility(returns)
                corr = compute_correlation_matrix(returns)
                progress_bar.progress(40)
                
                min_var = minimize_variance(mean_ret, cov_mat, risk_free_rate)
                max_sharpe = maximize_sharpe(mean_ret, cov_mat, risk_free_rate)
                progress_bar.progress(60)
                
                frontier = efficient_frontier(mean_ret, cov_mat, risk_free_rate, 50)
                sim = simulate_random_portfolios(n_portfolios, mean_ret, cov_mat, risk_free_rate, 42)
                progress_bar.progress(100)
                
                st.session_state.data = {
                    "prices": prices, "returns": returns, "mean_ret": mean_ret,
                    "cov_mat": cov_mat, "vols": vols, "corr": corr,
                    "min_var": min_var, "max_sharpe": max_sharpe,
                    "frontier": frontier, "sim": sim, "tickers": list(prices.columns),
                    "rf": risk_free_rate, "var_conf": var_conf
                }
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                return
    
    if "data" in st.session_state:
        d = st.session_state.data
        
        st.markdown("### 📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Max Sharpe Ratio",
                value=f"{d['max_sharpe'].sharpe_ratio:.3f}",
                delta="Optimal"
            )
        
        with col2:
            st.metric(
                label="Best Return",
                value=f"{d['max_sharpe'].expected_return:.2%}",
                delta="Annualized"
            )
        
        with col3:
            st.metric(
                label="Lowest Risk",
                value=f"{d['min_var'].volatility:.2%}",
                delta="Min Variance"
            )
        
        with col4:
            st.metric(
                label="Assets Analyzed",
                value=len(d['tickers']),
                delta=None
            )
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Efficient Frontier",
            "⚖️ Portfolio Weights",
            "📊 Risk Analysis",
            "🔗 Correlation",
            "📉 Asset Details"
        ])
        
        with tab1:
            st.markdown("### Efficient Frontier Visualization")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=d["sim"]["Volatility"],
                y=d["sim"]["Return"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=d["sim"]["Sharpe"],
                    colorscale="Plasma",
                    opacity=0.6,
                    colorbar=dict(
                        title="Sharpe<br>Ratio",
                        titlefont=dict(color='white'),
                        tickfont=dict(color='white'),
                        bgcolor='rgba(0,0,0,0.5)',
                        bordercolor='rgba(255,255,255,0.2)'
                    ),
                    showscale=True
                ),
                name="Random Portfolios",
                hovertemplate='Return: %{y:.2%}<br>Volatility: %{x:.2%}<br><extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=[p.volatility for p in d["frontier"]],
                y=[p.expected_return for p in d["frontier"]],
                mode="lines",
                line=dict(color="#e94560", width=3),
                name="Efficient Frontier"
            ))
            
            fig.add_trace(go.Scatter(
                x=[d["min_var"].volatility],
                y=[d["min_var"].expected_return],
                mode="markers",
                marker=dict(size=20, color="#00d9ff", symbol="star", line=dict(color='white', width=2)),
                name="Min Variance",
                hovertemplate='Min Variance<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=[d["max_sharpe"].volatility],
                y=[d["max_sharpe"].expected_return],
                mode="markers",
                marker=dict(size=20, color="#00ff88", symbol="star", line=dict(color='white', width=2)),
                name="Max Sharpe",
                hovertemplate='Max Sharpe<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=list(d["vols"]),
                y=list(d["mean_ret"]),
                mode="markers+text",
                marker=dict(size=12, color="#ffa500", line=dict(color='white', width=1)),
                text=d["tickers"],
                textposition="top center",
                textfont=dict(color='white', size=10),
                name="Individual Assets",
                hovertemplate='%{text}<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text="Risk-Return Tradeoff", font=dict(size=20, color='white')),
                xaxis_title="Volatility (Risk)",
                yaxis_title="Expected Return",
                xaxis_tickformat=".1%",
                yaxis_tickformat=".1%",
                height=600,
                hovermode='closest',
                **create_dark_theme_figure()
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            frontier_df = pd.DataFrame({
                'Return': [p.expected_return for p in d["frontier"]],
                'Volatility': [p.volatility for p in d["frontier"]],
                'Sharpe': [p.sharpe_ratio for p in d["frontier"]]
            })
            st.download_button(
                label="📥 Download Frontier Data",
                data=frontier_df.to_csv(index=False),
                file_name="efficient_frontier.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.markdown("### Portfolio Allocations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔷 Minimum Variance Portfolio")
                fig_donut_mv = create_donut_chart(d["min_var"].weights, d["tickers"], "Min Variance Weights")
                st.plotly_chart(fig_donut_mv, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔶 Maximum Sharpe Portfolio")
                fig_donut_ms = create_donut_chart(d["max_sharpe"].weights, d["tickers"], "Max Sharpe Weights")
                st.plotly_chart(fig_donut_ms, use_container_width=True)
            
            st.markdown("#### 📊 Weight Comparison")
            fig_bar = go.Figure()
            
            fig_bar.add_trace(go.Bar(
                name='Min Variance',
                x=d["tickers"],
                y=d["min_var"].weights,
                text=[f"{w:.1%}" for w in d["min_var"].weights],
                textposition='outside',
                marker_color='#00d9ff',
                hovertemplate='%{x}<br>Weight: %{y:.2%}<extra></extra>'
            ))
            
            fig_bar.add_trace(go.Bar(
                name='Max Sharpe',
                x=d["tickers"],
                y=d["max_sharpe"].weights,
                text=[f"{w:.1%}" for w in d["max_sharpe"].weights],
                textposition='outside',
                marker_color='#00ff88',
                hovertemplate='%{x}<br>Weight: %{y:.2%}<extra></extra>'
            ))
            
            fig_bar.update_layout(
                barmode='group',
                yaxis_tickformat=".0%",
                height=450,
                title=dict(text="Portfolio Weight Distribution", font=dict(size=18, color='white')),
                **create_dark_theme_figure()
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
            weights_df = pd.DataFrame({
                'Ticker': d['tickers'],
                'Min Variance': d['min_var'].weights,
                'Max Sharpe': d['max_sharpe'].weights
            })
            st.download_button(
                label="📥 Download Portfolio Weights",
                data=weights_df.to_csv(index=False),
                file_name="portfolio_weights.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.markdown("### Risk Metrics Dashboard")
            
            mv_ret = compute_portfolio_returns_series(d["min_var"].weights, d["returns"])
            ms_ret = compute_portfolio_returns_series(d["max_sharpe"].weights, d["returns"])
            
            mv_var = compute_var(mv_ret, d['var_conf'])
            mv_cvar = compute_cvar(mv_ret, d['var_conf'])
            mv_maxdd = compute_max_drawdown(mv_ret)
            
            ms_var = compute_var(ms_ret, d['var_conf'])
            ms_cvar = compute_cvar(ms_ret, d['var_conf'])
            ms_maxdd = compute_max_drawdown(ms_ret)
            
            st.markdown("#### 🎯 Risk Gauges")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_gauge = create_gauge_chart(
                    d['min_var'].volatility,
                    "Min Variance<br>Volatility",
                    0.5,
                    "#00d9ff"
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                fig_gauge = create_gauge_chart(
                    d['max_sharpe'].volatility,
                    "Max Sharpe<br>Volatility",
                    0.5,
                    "#00ff88"
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col3:
                fig_gauge = create_gauge_chart(
                    abs(ms_maxdd),
                    "Max Sharpe<br>Max Drawdown",
                    0.5,
                    "#e94560"
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("#### 📋 Comprehensive Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Expected Return", "Volatility", "Sharpe Ratio", f"VaR ({d['var_conf']:.0%})", f"CVaR ({d['var_conf']:.0%})", "Max Drawdown"],
                "Min Variance": [
                    f"{d['min_var'].expected_return:.2%}",
                    f"{d['min_var'].volatility:.2%}",
                    f"{d['min_var'].sharpe_ratio:.3f}",
                    f"{mv_var:.2%}",
                    f"{mv_cvar:.2%}",
                    f"{mv_maxdd:.2%}"
                ],
                "Max Sharpe": [
                    f"{d['max_sharpe'].expected_return:.2%}",
                    f"{d['max_sharpe'].volatility:.2%}",
                    f"{d['max_sharpe'].sharpe_ratio:.3f}",
                    f"{ms_var:.2%}",
                    f"{ms_cvar:.2%}",
                    f"{ms_maxdd:.2%}"
                ]
            })
            
            st.dataframe(metrics_df.set_index("Metric"), use_container_width=True)
            
            st.markdown("#### 📉 Returns Distribution")
            fig_dist = go.Figure()
            
            fig_dist.add_trace(go.Histogram(
                x=mv_ret,
                name='Min Variance',
                opacity=0.6,
                marker_color='#00d9ff',
                nbinsx=50
            ))
            
            fig_dist.add_trace(go.Histogram(
                x=ms_ret,
                name='Max Sharpe',
                opacity=0.6,
                marker_color='#00ff88',
                nbinsx=50
            ))
            
            fig_dist.update_layout(
                barmode='overlay',
                title=dict(text="Portfolio Returns Distribution", font=dict(size=18, color='white')),
                xaxis_title="Daily Returns",
                yaxis_title="Frequency",
                height=400,
                **create_dark_theme_figure()
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab4:
            st.markdown("### Correlation Analysis")
            
            fig = go.Figure(data=go.Heatmap(
                z=d["corr"].values,
                x=d["tickers"],
                y=d["tickers"],
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(d["corr"].values, 2),
                texttemplate='%{text}',
                textfont={"size": 12, "color": "white"},
                colorbar=dict(
                    title='Correlation',
                    titlefont=dict(color='white'),
                    tickfont=dict(color='white'),
                    bgcolor='rgba(0,0,0,0.5)'
                ),
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text="Asset Correlation Matrix", font=dict(size=20, color='white')),
                height=600,
                **create_dark_theme_figure()
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 🔍 Correlation Insights")
            
            corr_matrix = d["corr"].values
            n = len(d["tickers"])
            
            upper_tri = np.triu_indices(n, k=1)
            correlations = corr_matrix[upper_tri]
            
            if len(correlations) > 0:
                max_corr_idx = np.argmax(correlations)
                min_corr_idx = np.argmin(correlations)
                
                max_i, max_j = upper_tri[0][max_corr_idx], upper_tri[1][max_corr_idx]
                min_i, min_j = upper_tri[0][min_corr_idx], upper_tri[1][min_corr_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="feature-card">
                        <h3>📈 Highest Correlation</h3>
                        <p><strong>{d['tickers'][max_i]}</strong> & <strong>{d['tickers'][max_j]}</strong></p>
                        <p>Correlation: <strong>{correlations[max_corr_idx]:.3f}</strong></p>
                        <p>These assets move together strongly, offering less diversification benefit.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="feature-card">
                        <h3>📉 Lowest Correlation</h3>
                        <p><strong>{d['tickers'][min_i]}</strong> & <strong>{d['tickers'][min_j]}</strong></p>
                        <p>Correlation: <strong>{correlations[min_corr_idx]:.3f}</strong></p>
                        <p>These assets provide excellent diversification potential.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.download_button(
                label="📥 Download Correlation Matrix",
                data=d["corr"].to_csv(),
                file_name="correlation_matrix.csv",
                mime="text/csv"
            )
        
        with tab5:
            st.markdown("### Individual Asset Analysis")
            
            st.markdown("#### 📊 Asset Statistics")
            stats_df = pd.DataFrame({
                "Ticker": d["tickers"],
                "Annualized Return": [f"{r:.2%}" for r in d["mean_ret"]],
                "Annualized Volatility": [f"{v:.2%}" for v in d["vols"]],
                "Sharpe Ratio": [f"{(d['mean_ret'][i] - d['rf']) / d['vols'][i]:.3f}" for i in range(len(d['tickers']))]
            })
            st.dataframe(stats_df.set_index("Ticker"), use_container_width=True)
            
            st.markdown("#### 📈 Normalized Price Performance")
            norm_prices = d["prices"] / d["prices"].iloc[0] * 100
            
            fig_norm = go.Figure()
            for ticker in d["tickers"]:
                fig_norm.add_trace(go.Scatter(
                    x=norm_prices.index,
                    y=norm_prices[ticker],
                    mode='lines',
                    name=ticker,
                    hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
                ))
            
            fig_norm.update_layout(
                title=dict(text="Normalized Prices (Base 100)", font=dict(size=18, color='white')),
                xaxis_title="Date",
                yaxis_title="Normalized Value",
                height=500,
                hovermode='x unified',
                **create_dark_theme_figure()
            )
            
            st.plotly_chart(fig_norm, use_container_width=True)
            
            st.markdown("#### 🔥 Monthly Returns Heatmap")
            fig_heatmap = create_monthly_returns_heatmap(d["returns"], d["tickers"])
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.download_button(
                label="📥 Download Price Data",
                data=d["prices"].to_csv(),
                file_name="historical_prices.csv",
                mime="text/csv"
            )
    
    else:
        st.markdown("### 🌟 Welcome to Mini Aladdin!")
        st.markdown("""
        Configure your portfolio analysis using the sidebar and click **Run Analysis** to get started.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>📈 Portfolio Optimization</h3>
                <p>Find the optimal portfolio allocation using Modern Portfolio Theory. Calculate minimum variance and maximum Sharpe ratio portfolios.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>🎯 Efficient Frontier</h3>
                <p>Visualize the risk-return tradeoff with an interactive efficient frontier plot showing optimal portfolios.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>📊 Risk Metrics</h3>
                <p>Comprehensive risk analysis including VaR, CVaR, Sharpe ratio, and maximum drawdown calculations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>🔗 Correlation Analysis</h3>
                <p>Discover asset relationships and diversification opportunities through correlation heatmaps and insights.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>🎲 Monte Carlo Simulation</h3>
                <p>Simulate thousands of random portfolios to explore the investment opportunity set.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>📉 Asset Analytics</h3>
                <p>Deep dive into individual asset performance, statistics, and historical price movements.</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit & Plotly | 
        <a href="https://github.com/Fenil777/mini-aladdin" target="_blank">View on GitHub</a></p>
        <p>© 2024 Mini Aladdin • Portfolio Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# Mini Aladdin 🏰

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Portfolio Optimization & Risk Engine**

A Python-based mini portfolio risk and optimization engine inspired by institutional-grade systems like BlackRock's Aladdin. Perform mean-variance optimization, Monte Carlo simulation, and comprehensive risk analysis with an interactive Streamlit dashboard.

## ✨ Features

- 📊 **Historical Data Fetching** - Download price data via yfinance
- 📈 **Returns Analysis** - Log returns and annualized statistics
- ⚖️ **Portfolio Optimization** - Minimum variance & maximum Sharpe ratio portfolios
- 🎯 **Efficient Frontier** - Generate and visualize the efficient frontier
- 🎲 **Monte Carlo Simulation** - Simulate thousands of random portfolios
- 📉 **Risk Metrics** - VaR, CVaR, Sharpe ratio, maximum drawdown
- 🔗 **Correlation Analysis** - Asset correlation heatmap
- 🖥️ **Interactive Dashboard** - Streamlit-based UI with real-time analysis
- ✅ **Comprehensive Testing** - pytest-based test suite

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Fenil777/mini-aladdin.git
cd mini-aladdin

# Install dependencies
pip install -r requirements.txt
```

### Run Dashboard

```bash
streamlit run src/dashboard.py
```

### Run CLI

```bash
# Basic analysis with default config
python main.py

# With custom config
python main.py --config config/config.yaml

# Save plots and CSV output
python main.py --save-plots --save-csv
```

### Run Tests

```bash
pytest
```

## 📁 Project Structure

```
mini-aladdin/
├── config/
│   └── config.yaml          # Configuration file
├── data/
│   ├── cache/               # Cached price data
│   └── plots/               # Generated plots
├── src/
│   ├── __init__.py
│   ├── config_loader.py     # Configuration loading
│   ├── data_loader.py       # Price data fetching
│   ├── returns.py           # Returns calculations
│   ├── risk_metrics.py      # Risk metrics (VaR, CVaR, etc.)
│   ├── optimizer.py         # Portfolio optimization
│   ├── simulator.py         # Monte Carlo simulation
│   ├── pipeline.py          # Analysis orchestration
│   ├── reporting.py         # Report generation
│   └── dashboard.py         # Streamlit dashboard
├── tests/
│   ├── test_returns.py
│   ├── test_risk_metrics.py
│   ├── test_optimizer.py
│   ├── test_simulator.py
│   ├── test_pipeline.py
│   └── test_reporting.py
├── main.py                  # CLI entry point
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
data:
  tickers:
    - AAPL
    - MSFT
    - GOOGL
  start_date: "2022-01-01"
  end_date: "2024-12-31"

portfolio:
  risk_free_rate: 0.04

simulation:
  n_portfolios: 5000

risk:
  var_confidence: 0.95
```

## 📊 Example Output

### Efficient Frontier
Visualizes the efficient frontier with:
- Monte Carlo simulated portfolios (colored by Sharpe ratio)
- Efficient frontier curve
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio
- Individual assets

### Portfolio Weights
Bar charts showing asset allocations for:
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio

### Risk Metrics
Comparison table including:
- Expected return
- Volatility
- Sharpe ratio
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Maximum drawdown

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_optimizer.py

# Run with verbose output
pytest -v
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **Data** | yfinance, pandas, numpy |
| **Optimization** | scipy |
| **Visualization** | plotly |
| **Dashboard** | streamlit |
| **Configuration** | PyYAML |
| **Testing** | pytest |

## 📚 Methodology

### Portfolio Optimization

**Minimum Variance Portfolio:**
- Minimizes portfolio volatility: `σ_p = √(w^T Σ w)`
- Subject to: `Σw_i = 1, w_i ≥ 0`

**Maximum Sharpe Ratio Portfolio:**
- Maximizes: `(μ_p - r_f) / σ_p`
- Where `μ_p = w^T μ` is expected return
- And `r_f` is the risk-free rate

### Risk Metrics

- **VaR (Value at Risk)**: Maximum expected loss at given confidence level
- **CVaR (Conditional VaR)**: Expected loss beyond VaR threshold
- **Sharpe Ratio**: `(Return - Risk-Free Rate) / Volatility`
- **Maximum Drawdown**: Largest peak-to-trough decline

### Monte Carlo Simulation

Generates random portfolio weights and calculates risk-return profiles to visualize the investment opportunity set.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Inspired by BlackRock's Aladdin risk management platform. Built for educational purposes to demonstrate portfolio optimization and risk analysis techniques.

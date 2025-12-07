# Mini Aladdin 🏰

**Portfolio Optimization & Risk Engine**

A Python-based mini portfolio risk and optimization engine inspired by institutional-grade systems like BlackRock's Aladdin.

## 🚧 Status

This project is under active development.

- [x] Milestone 1: Project Setup & Data Layer
- [ ] Milestone 2: Returns & Risk Metrics Engine  
- [ ] Milestone 3: Optimizer & Monte Carlo Simulation
- [ ] Milestone 4: Pipeline Orchestration & CLI
- [ ] Milestone 5: Streamlit Dashboard & Documentation

## Features (Planned)

- 📊 Historical price data fetching via yfinance
- 📈 Log returns and annualized statistics
- ⚖️ Mean-variance optimization (min-variance, max-Sharpe)
- 🎲 Monte Carlo portfolio simulation
- 📉 Risk metrics: VaR, CVaR, Sharpe ratio
- 🖥️ Interactive Streamlit dashboard
- ✅ Comprehensive test suite

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI
python main.py

# Run dashboard (coming in Milestone 5)
# streamlit run src/dashboard.py

# Run tests
pytest
```

## Tech Stack

- **Data**: yfinance, pandas, numpy
- **Optimization**: scipy
- **Visualization**: plotly
- **Dashboard**: streamlit
- **Testing**: pytest

## License

MIT

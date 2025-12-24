# Quant Research Framework

<div align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/status-active-success.svg" alt="Status">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</div>

<br />

> **A professional, modular, and extensible Quantitative Trading Research Framework built from scratch.**
> Designed to bridge the gap between academic theory and real-world quantitative strategy development.

---

## **Overview**

This project is the culmination of advanced studies in **Artificial Intelligence and Data Science**, applied to the domain of **Quantitative Finance**. 

The goal was to build a research-grade environment capable of:
1.  **Ingesting and cleaning** financial market data.
2.  **Backtesting** systematic trading strategies with rigor (accounting for transaction costs, lag, etc.).
3.  **Analyzing performance** using industry-standard metrics (Sharpe, Drawdown, CAGR).
4.  **Applying Machine Learning** to predict market movements.

It avoids "black-box" backtesting libraries to ensure full transparency and control over every calculation, making it an ideal showcase of **data engineering**, **software architecture**, and **financial modeling** skills.

---

## **Key Features**

### **Core Architecture**
- **Modular Design:** Clear separation of concerns: `Data` → `Preprocessing` → `Features` → `Backtesting` → `Analysis`.
- **Robust Data Pipeline:** Automated fetching from Yahoo Finance, cleaning, and standardization.
- **Vectorized Backtester:** High-performance engine using pandas vectorization for instantaneous strategy simulation.

### **Machine Learning Integration**
- **Feature Engineering:** Automated calculation of Technical Indicators (RSI, Momentum, Volatility, Moving Averages, MACD, Bollinger Bands, ATR).
- **ML Workflow:** End-to-end pipeline for training, validating, and testing models (Random Forest, XGBoost) to predict market direction.
- **Validation:** Robust Walk-Forward Validation (Rolling Window) to prevent overfitting.
- **Evaluation:** Visual tools for Feature Importance, Confusion Matrices, and Classification Reports.

### **Interactive Dashboard**
- Built with **Streamlit**, offering a "Command Center" experience.
- **Strategy Backtester:** Run, visualize, and compare strategies (e.g., *Buy & Hold* vs *MA Crossover*) in real-time.
- **ML Laboratory:** Train AI models on the fly and inspect their learning process deeply.
- **Professional Visualization:** Interactive charts using `Plotly` for Equity Curves, Drawdowns, and Signals.

---

## **Installation**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/quant-research-framework.git
    cd quant-research-framework
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    > **Note for Mac Users:** XGBoost requires OpenMP. If you encounter errors, run:
    > ```bash
    > brew install libomp
    > ```

---

## **Usage**

### **1. Launch the Dashboard (Recommended)**
The easiest way to interact with the framework is through the web interface.
```bash
python -m streamlit run dashboard.py
```
This will open a local web server where you can:
- Select assets (SPY, BTC, NVDA...)
- Run backtests interactively.
- Train ML models without writing code.

### **2. Run Scripts Manually**
You can also run individual modules from the command line:

*   **Download Data:**
    ```bash
    python -m scripts.download_default_data
    ```

*   **Run a CLI Backtest:**
    ```bash
    python -m scripts.run_simple_backtest
    ```

---

## **Project Structure**

```text
quant-research-framework/
│
├── dashboard.py                  # Main Entry Point (Web App)
├── requirements.txt              # Project Dependencies
│
├── src/
│   ├── data/
│   │   ├── loaders.py            # Data ingestion (YFinance)
│   │   ├── preprocessing.py      # Cleaning & Normalization
│   │   └── features.py           # Feature Engineering for ML
│   │
│   ├── models/
│   │   ├── trainer.py            # ML Model Training & Evaluation Logic
│   │   └── validation.py         # Walk-Forward Validation Logic
│   │
│   ├── backtesting/
│   │   ├── engine.py             # Vectorized Backtest Engine
│   │   └── strategies.py         # Strategy Logic (MA Cross, Buy&Hold, AI)
│   │
│   ├── analysis/
│   │   ├── performance.py        # Metrics (Sharpe, CAGR, Volatility)
│   │   └── plotting.py           # Visualization Utilities
│   │
│   └── utils/
│       └── config.py             # Configuration Management
│
└── notebooks/                    # Jupyter Notebooks for Research
```

---

## **Roadmap**

- [x] **Core Framework:** Data Pipeline, Backtester, and Metrics.
- [x] **Dashboard:** Interactive Web App for visualizing results.
- [x] **Machine Learning:** Feature Engineering & Model Training Module.
- [x] **AI Strategy:** Deploying the trained ML model as a live trading strategy.
- [x] **Optimization:** Implementing Walk-Forward Analysis.
- [ ] **Advanced Optimization:** Hyperparameter Tuning (Optuna/GridSearch).
- [ ] **Portfolio Management:** Support for multi-asset portfolios and rebalancing.

---

## **Author**

**Joan Batlló**  
*AI & Data Science Student | FinTech Enthusiast*

---

*Disclaimer: This project is for educational and research purposes only. It is not financial advice.*
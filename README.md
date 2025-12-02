# Quant Research Framework

A modular quantitative trading research framework built from scratch.  
The project follows a clean, professional architecture used in real quant environments:

**Data → Preprocessing → Backtesting → Strategies → Performance Analysis**

The goal is to provide a flexible pipeline for developing, testing, and evaluating trading strategies.

---

## 📁 Project Structure
```text
quant-research-framework/
│
├── data/                         # Local datasets
│   ├── raw/                      # Raw market data (e.g., yfinance downloads)
│   ├── processed/                # Clean, standardized datasets
│   └── metadata/                 # Data descriptions, metadata, symbol lists
│
├── notebooks/                    # Exploration & research notebooks
│
├── src/
│   ├── data/
│   │   ├── loaders.py            # Download / load OHLCV price data
│   │   └── preprocessing.py      # Cleaning, normalization, returns, prep pipeline
│   │
│   ├── backtesting/
│   │   ├── engine.py             # Backtest engine (positions → returns → equity)
│   │   └── strategies.py         # Strategy definitions (signals)
│   │
│   ├── analysis/
│   │   └── performance.py        # Sharpe, volatility, drawdown, performance stats
│   │
│   └── utils/
│       └── config.py             # Global YAML configuration loader
│
├── scripts/                      # Executable scripts
│   ├── download_default_data.py  # Download SPY or configured symbol
│   └── run_simple_backtest.py    # First full pipeline test (always long)
│
├── config/
│   └── settings.yaml             # Project-wide configuration file
│
├── .gitignore
├── README.md
└── requirements.txt
```


---

## 🚀 Getting Started

### 1. Install dependencies
pip install -r requirements.txt
### 2. Download default historical data
python -m scripts.download_default_data
### 3. Run a simple backtest
python -m scripts.run_simple_backtest

## 🔧 Pipeline Overview

### **1. Data Loading**
- Uses Yahoo Finance (`yfinance`)
- Downloads OHLCV data for the configured symbol
- Saves a copy under `data/raw/`

### **2. Preprocessing**
- Flattens MultiIndex columns from yfinance
- Normalizes column names
- Ensures clean, sorted `DatetimeIndex`
- Computes simple & log returns
- Outputs a clean DataFrame ready for backtesting

### **3. Backtesting Engine**
- Converts strategy signals into portfolio returns
- Applies transaction costs
- Calculates equity curve through compounding
- Returns a `BacktestResult` object containing:
  - positions  
  - strategy returns  
  - equity curve  

### **4. Strategies**
- Strategies generate **signals** (position time series)
- Example implemented:
  - `always_long` (baseline: 100% invested)
- Future strategies will include:
  - moving averages  
  - RSI  
  - volatility filters  
  - ML-based signals  

### **5. Performance Analysis** *(coming next)*
- Total return  
- Annualized return (CAGR)  
- Volatility  
- Sharpe ratio  
- Maximum drawdown  
- Win rate  
- Summary performance table  

---

## 🎯 Purpose of the Project

This project is built step-by-step as a learning exercise to understand:

- Financial data engineering  
- Quantitative strategy research  
- Backtesting methodology  
- Software architecture for quant systems  
- Modular, reusable design practices  

The final goal is to build a **research-grade trading environment** that can scale to:

- multiple assets  
- multiple strategies  
- hyperparameter sweeps  
- ML models  
- portfolio-level analysis  
- risk overlays  

---

## 📌 Roadmap

- [x] Project structure and configuration system  
- [x] Data loader (Yahoo Finance)  
- [x] Preprocessing pipeline  
- [x] Backtesting engine  
- [x] Baseline strategy (`always_long`)  
- [ ] Performance metrics module  
- [ ] Visualization tools (equity curves, drawdowns)  
- [ ] Additional strategies (MA Cross, RSI, etc.)  
- [ ] Portfolio engine (multi-asset)  
- [ ] ML-based strategy prototypes  

---

## 🧠 About This Project

This framework is built step-by-step with strong emphasis on:

- clear explanations  
- best practices  
- modular design  
- interview-ready understanding  

Every component is documented so it can be explained confidently in a technical interview or used as a foundation for larger quant projects.

---

If you want, I can also generate:

- a **technical architecture diagram**  
- an **interview explanation section**  
- a **roadmap.md** file  
- a **CONTRIBUTING.md** file  

Just ask!
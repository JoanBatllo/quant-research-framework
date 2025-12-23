# Steps Followed & Explanations

## Step 1- Define the project settings + config system.

**config/Settings.yaml**
He creat el settings.yaml, es el control panel del meu projecte. Inclou totes les variables globals (data paths, default symbols, tim ranges, backtest parameters, and risk assumptions). No fem hardcoding de values, tot viu aquí. Es pot tornar per canviar quan es vulgui.

**src/utils/config.py**
This file creates a clean interface to load the YAML file. Instead of `yaml.safe_load(…)` everywhere, you call the function `get_config`.

## Step 2- Data loading Module
He començat insalant yfinance per treure la informació de internet.

**src/data/loaders.py**
The intend of this file is download historical OHLCV data (Open, High, Low, Close, Volume) from Yahoo Finance and bring it into a clean pandas DataFrame that the rest of your system can use.
`get_price_data` is a function designed to download the data from yahoo finance, and converte it to a df. It cleans the values, drop Nan, sort index.
`_map_frequency_to_yf_interval`. In the config file, hem especificat la data com a D (daily), W, H… Però Yahoo funciona 1d, 1wk, 1h. Canviem de internal notation to Yahoo’s API.
`_safe_raw_data()`. Descarreguem la data i la posem a un CSV, així podem veure si es correcte o no. Dataset contains two columna levels (level 0: Price type (Open, Close, etc…, level 1: ticket) Multindex. Next step is gonna be flatten the index.

**Scripts/download_default_data**
Test everything we did above is correct.

**src/data/preprocessing**
`standardize_price_dataframe`. Take the data from Yahoo that is ugly, and clean it into a standard table.
First we flatten the multicolumn index (drop ticker level, and keep names)
Rename columns, standardize to AdjClose
Keep only relevant columns usually is all of them, but just in case there is extra data.
Ensure a proper DatetimeIndex, convert string to datatype time. Sort by. Date, and name the index Date.
Keep the dates to the ones in the config file, even if yfinance gives more.
Drop rows that are fully Nan
`add_return_columns`. Compute how much the asset changes from one period to the next. We add Return = (Pt /pt-1) -1 and logReturn. Adjusted close includes dividends and splits, so it reflects returns more accurately. By default we use AdjClose.
`prepare_price_data`. Applies all the previous and returns the data cleaned.

## Step 3 – Backtest

**src/backtesting/engine.py**
This file is the core of our backtesting engine for a single Asset. Given cleaned price data with returns and a position signal over time, it computes strategy returns (with trading costs) The evolution of the portfolio value (equity curve )
`BacktestResult` (dataclass)
This is the container for the equity_curve (portfolio value over time), returns (daily strategy returns after costs), positions (the position 0-1 over time.
`run_backtest(…)` Aligns the position signal with the data index. Uses yesterday’s position to get today’s return (conceptually important) Applies transaction costs when the position changes Builds the equity curve from net returns.

**src/backtesting/strategies.py**
Provide Strategy definitions: functions that take data and output a position signal over time. The first implementation has been `always_long`.
`always_long`, ignores the content of data, returns a series of 1.0 with the same index. Interpreted by `run_backtest()` as: 100% invested in the asset all times.
Financial meaning; equivalent to buy & hold from beginning to the end. Very important benchmark, (other strategies will be compared against this one.

## Explanation of Project folder.

**Scripts/**
Holds small executable Python scripts whose goal is to run something, not define library code. Contents Quick tests, Workflows, data management utilities, experiment scripts.

**Backtesting/**
Contains everything needed to simulate a trading strategy in the past. Its purpose is convert decisions of a strategy in money through time. Takes data, a strategy and calculates if you would have gained money.

---

## Session 2 — Performance Analysis Module

**WHAT WE DID**
- Implemented `src/analysis/performance.py` with core financial metrics:
  - `calculate_total_return()`: Overall percentage profit/loss.
  - `calculate_cagr()`: Compound Annual Growth Rate.
  - `calculate_volatility()`: Annualized standard deviation of returns.
  - `calculate_sharpe_ratio()`: Risk-adjusted return (Return / Volatility).
  - `calculate_max_drawdown()`: Maximum peak-to-valley drop.
  - `summarize_performance()`: Wrapper to compute all metrics at once.
- Created `scripts/test_performance_metrics.py` to verify the math with unit tests.
- Verified that all tests passed.

**WHY THIS MATTERS**
- Moves beyond simple "profit" to "risk-adjusted profit".
- Allows us to scientifically compare strategies (e.g., Strategy A made more money, but Strategy B had much less risk).
- Standardizes how we evaluate success across the entire project.

**NEXT STEPS**
- Integrate these metrics into the backtest engine (`src/backtesting/engine.py`) so every backtest automatically reports these stats.
- Visualize the equity curve and drawdowns.
- Implement new strategies (e.g. Moving Average Crossover).

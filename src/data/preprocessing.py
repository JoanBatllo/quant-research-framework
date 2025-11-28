# src/data/preprocessing.py

from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np

from src.utils.config import get_config
from src.data.loaders import get_price_data


def standardize_price_dataframe(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Clean and standardize a raw OHLCV DataFrame (e.g. from yfinance).

    Steps:
    - Flatten MultiIndex columns (drop ticker level).
    - Ensure consistent column names and ordering.
    - Sort by date and ensure a DatetimeIndex.
    - Clip to the date range specified in the config.

    Parameters
    ----------
    df : pd.DataFrame
        Raw price DataFrame returned by get_price_data or yfinance.
    symbol : str, optional
        Symbol used for logging or future extensions. Not mandatory for now.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a DatetimeIndex named "Date" and columns:
        ["Open", "High", "Low", "Close", "AdjClose", "Volume"] (when available).
    """
    cfg = get_config()

    # ---- 1. Handle MultiIndex columns by keeping only the first level ----
    if isinstance(df.columns, pd.MultiIndex):
        # For a single ticker, second level is usually the ticker name ("SPY").
        # We drop it and keep only the price type ("Open", "Close", etc.).
        df.columns = df.columns.get_level_values(0)

    # ---- 2. Normalize column names ----
    # We want standard names regardless of how yfinance spells them.
    rename_map = {
        "Adj Close": "AdjClose",
        "Adj close": "AdjClose",
        "adjclose": "AdjClose",
        "Close": "Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    # Keep only the columns we care about (if they exist)
    desired_cols = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    existing_cols = [c for c in desired_cols if c in df.columns]
    df = df[existing_cols]

    # ---- 3. Ensure index is a DatetimeIndex and sorted ----
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df.index.name = "Date"

    # ---- 4. Clip to the configured date range (just to be safe) ----
    start = pd.to_datetime(cfg["data"]["start_date"])
    end = pd.to_datetime(cfg["data"]["end_date"])

    df = df.loc[(df.index >= start) & (df.index <= end)]

    # ---- 5. Drop rows where everything is NaN ----
    df = df.dropna(how="all")

    return df


def add_return_columns(
    df: pd.DataFrame,
    use_adj_close: bool = True,
) -> pd.DataFrame:
    """
    Add simple and log returns to the price DataFrame.

    Returns measure how much the price changes from one period to the next.

    Simple return (R_t):
        R_t = (P_t / P_{t-1}) - 1

    Log return (r_t):
        r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    df : pd.DataFrame
        Clean price DataFrame with at least "Close" or "AdjClose".
    use_adj_close : bool, default True
        If True, use "AdjClose" for returns (includes dividends & splits).
        If False, use "Close".

    Returns
    -------
    pd.DataFrame
        Same DataFrame with two new columns:
        - "Return"   (simple return)
        - "LogReturn" (logarithmic return)
    """
    price_col = "AdjClose" if use_adj_close and "AdjClose" in df.columns else "Close"

    if price_col not in df.columns:
        raise ValueError(
            f"Cannot compute returns because '{price_col}' column is missing."
        )

    prices = df[price_col]

    # Simple return
    df["Return"] = prices.pct_change()

    # Log return
    df["LogReturn"] = (prices / prices.shift(1)).apply(lambda x: pd.NA if pd.isna(x) else np.log(x))

    # First row will have NaN returns (no previous price); that's expected.
    return df


def prepare_price_data(
    symbol: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: Optional[str] = None,
    use_adj_close_for_returns: bool = True,
) -> pd.DataFrame:
    """
    Convenience function:
    1) Download data using get_price_data.
    2) Standardize the DataFrame structure.
    3) Add simple and log returns.

    This gives you a ready-to-use dataset for backtesting and analysis.

    Parameters
    ----------
    symbol : str, optional
        Ticker to download. If None, uses default_symbol from config.
    start : str, optional
        Start date in "YYYY-MM-DD". If None, uses config.
    end : str, optional
        End date in "YYYY-MM-DD". If None, uses config.
    interval : str, optional
        yfinance interval (e.g. "1d"). If None, mapped from config frequency.
    use_adj_close_for_returns : bool, default True
        Whether to compute returns using "AdjClose" instead of "Close".

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with OHLCV + returns, ready for backtesting.
    """
    # Step 1: download raw data
    raw_df = get_price_data(
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_save=True,  # keep saving a copy in data/raw
    )

    # Step 2: standardize structure
    clean_df = standardize_price_dataframe(raw_df, symbol=symbol)

    # Step 3: add returns
    clean_df = add_return_columns(clean_df, use_adj_close=use_adj_close_for_returns)

    return clean_df
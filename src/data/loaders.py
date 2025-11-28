# src/data/loaders.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf
from src.utils.config import get_config


def get_price_data(
    symbol: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: Optional[str] = None,
    auto_save: bool = True,
) -> pd.DataFrame:
    """
    Download historical OHLCV price data from Yahoo Finance.

    Parameters
    ----------
    symbol : str, optional
        Ticker symbol to download (e.g. "SPY").
        If None, uses the default symbol from the config file.
    start : str, optional
        Start date in "YYYY-MM-DD" format.
        If None, uses the default start_date from the config file.
    end : str, optional
        End date in "YYYY-MM-DD" format (exclusive in yfinance).
        If None, uses the default end_date from the config file.
    interval : str, optional
        Data frequency. Examples:
        "1d" (daily), "1h" (hourly), "1wk" (weekly).
        If None, uses the price_frequency from the config file
        (we will map "D" to "1d", "W" to "1wk", etc.).
    auto_save : bool, default True
        If True, save the downloaded data as a CSV file in data/raw/.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime with at least the columns:
        ["Open", "High", "Low", "Close", "Adj Close", "Volume"].

    Notes
    -----
    - This function is intentionally simple to make the data flow easy to follow.
    - We rely on Yahoo Finance via the `yfinance` library.
    """
    cfg = get_config()

    # ---- 1. Resolve parameters using the config defaults ----
    symbol = symbol or cfg["data"]["default_symbol"]
    start = start or cfg["data"]["start_date"]
    end = end or cfg["data"]["end_date"]

    # Config uses "D" for daily, but yfinance uses "1d", etc.
    default_freq = cfg["data"].get("price_frequency", "D")
    if interval is None:
        interval = _map_frequency_to_yf_interval(default_freq)

    # ---- 2. Call yfinance to download the data ----
    data = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,  # keep raw OHLC; we can adjust later in preprocessing
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No data returned from Yahoo Finance for symbol={symbol}, "
            f"start={start}, end={end}, interval={interval}."
        )

    # ---- 3. Basic cleaning: sort index and drop all-NaN rows ----
    data = data.sort_index()
    data = data.dropna(how="all")

    # Ensure the index is named consistently
    data.index.name = "Date"

    # ---- 4. Optionally save to CSV in data/raw/ ----
    if auto_save:
        _save_raw_data(data, symbol, interval)

    return data


def _map_frequency_to_yf_interval(freq: str) -> str:
    """
    Map our internal frequency notation (like 'D', 'W') to yfinance interval strings.

    Examples
    --------
    'D' -> '1d'
    'W' -> '1wk'
    'H' -> '1h'

    Parameters
    ----------
    freq : str
        Frequency string used in the config.

    Returns
    -------
    str
        Interval string compatible with yfinance.
    """
    freq = freq.upper()

    mapping = {
        "D": "1d",   # daily
        "W": "1wk",  # weekly
        "H": "1h",   # hourly
    }

    return mapping.get(freq, "1d")  # default to daily if unknown


def _save_raw_data(data: pd.DataFrame, symbol: str, interval: str) -> None:
    """
    Save the raw downloaded price data as a CSV file under data/raw/.

    The filename encodes the symbol and interval, e.g.:
    data/raw/SPY_1d.csv

    Parameters
    ----------
    data : pd.DataFrame
        The price data to save.
    symbol : str
        The ticker symbol.
    interval : str
        The yfinance interval string (e.g. "1d", "1wk").
    """
    cfg = get_config()
    raw_dir = Path(cfg["data"]["raw_dir"])

    raw_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol}_{interval}.csv"
    output_path = raw_dir / filename

    data.to_csv(output_path, index=True)
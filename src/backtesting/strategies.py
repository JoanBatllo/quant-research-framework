# src/backtesting/strategies.py

from __future__ import annotations

import pandas as pd


def always_long(data: pd.DataFrame) -> pd.Series:
    """
    Trivial baseline strategy: always fully invested in the asset.

    Parameters
    ----------
    data : pd.DataFrame
        Price/returns DataFrame with a DatetimeIndex (e.g. from prepare_price_data).

    Returns
    -------
    pd.Series
        Position series with value 1.0 for all dates, meaning:
        - 100% of the capital is invested in the asset at all times.
    """
    position = pd.Series(1.0, index=data.index, name="always_long")
    return position


def moving_average_crossover(data: pd.DataFrame, fast_window: int = 10, slow_window: int = 50) -> pd.Series:
    """
    Simple Moving Average Crossover strategy.
    
    Logic:
    - Long (1.0) when Fast MA > Slow MA
    - Cash (0.0) when Fast MA <= Slow MA
    
    Parameters
    ----------
    data : pd.DataFrame
        Must contain a "Close" column.
    fast_window : int
        Lookback period for the fast moving average.
    slow_window : int
        Lookback period for the slow moving average.
        
    Returns
    -------
    pd.Series
        Position series (1.0 or 0.0).
    """
    # Calculate Moving Averages
    fast_ma = data["Close"].rolling(window=fast_window).mean()
    slow_ma = data["Close"].rolling(window=slow_window).mean()
    
    # Generate Signal (Boolean -> Float)
    # True (1.0) when Fast > Slow, False (0.0) otherwise
    signal_values = (fast_ma > slow_ma).astype(float)
    
    # Create Series aligned with data index
    position = pd.Series(signal_values, index=data.index, name=f"ma_cross_{fast_window}_{slow_window}")
    
    return position
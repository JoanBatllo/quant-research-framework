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
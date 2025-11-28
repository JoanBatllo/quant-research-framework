# src/backtesting/engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.utils.config import get_config


@dataclass
class BacktestResult:
    """
    Container for backtest results for a single asset and strategy.

    Attributes
    ----------
    equity_curve : pd.Series
        Time series of portfolio value over time.
    returns : pd.Series
        Strategy daily returns after transaction costs.
    positions : pd.Series
        Position over time (e.g. 0 = out of market, 1 = fully invested).
    """
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series


def run_backtest(
    data: pd.DataFrame,
    signal: pd.Series,
    initial_cash: Optional[float] = None,
    trading_cost_bps: Optional[float] = None,
) -> BacktestResult:
    """
    Run a simple backtest for a single asset using a position signal.

    Assumptions
    -----------
    - Single asset.
    - 'data' contains a 'Return' column with asset returns per period.
    - 'signal' represents the fraction of capital invested in the asset:
        0.0 → fully in cash
        1.0 → fully invested in the asset
        (later we could allow leverage or shorts).

    Parameters
    ----------
    data : pd.DataFrame
        Price/returns DataFrame, typically the output of prepare_price_data().
        Must contain a 'Return' column.
    signal : pd.Series
        Desired position per date (same index as 'data' or broadcastable).
        Values usually in [0, 1].
    initial_cash : float, optional
        Starting capital. If None, uses config['backtest']['initial_cash'].
    trading_cost_bps : float, optional
        Trading cost in basis points (1 bps = 0.01%) applied to
        changes in position. If None, uses config['backtest']['trading_cost_bps'].

    Returns
    -------
    BacktestResult
        Object containing equity curve, strategy returns, and positions.
    """
    cfg = get_config()

    if initial_cash is None:
        initial_cash = cfg["backtest"]["initial_cash"]

    if trading_cost_bps is None:
        trading_cost_bps = cfg["backtest"]["trading_cost_bps"]

    if "Return" not in data.columns:
        raise ValueError("Input 'data' must contain a 'Return' column.")

    # ---- 1. Align signal to data index ----
    # Reindex the signal to match the data dates and forward-fill.
    signal = signal.reindex(data.index).ffill().fillna(0.0)
    signal.name = signal.name or "position"

    # ---- 2. Compute gross strategy returns (before costs) ----
    # We assume the return at time t depends on the position held at t-1.
    asset_returns = data["Return"].fillna(0.0)
    shifted_position = signal.shift(1).fillna(0.0)

    strategy_returns = shifted_position * asset_returns

    # ---- 3. Apply transaction costs on position changes ----
    # Cost is proportional to how much we change the position.
    position_change = signal.diff().abs().fillna(0.0)
    cost_rate = trading_cost_bps / 10_000.0  # bps → decimal (e.g. 10 bps = 0.001)

    cost = position_change * cost_rate

    net_returns = strategy_returns - cost

    # ---- 4. Build equity curve ----
    equity_curve = (1 + net_returns).cumprod() * initial_cash
    equity_curve.name = "equity"

    return BacktestResult(
        equity_curve=equity_curve,
        returns=net_returns,
        positions=signal,
    )
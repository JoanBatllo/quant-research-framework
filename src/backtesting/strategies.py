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


def ml_strategy(data: pd.DataFrame, model, feature_cols: list[str]) -> pd.Series:
    """
    Machine Learning Strategy.
    
    Uses a pre-trained classification model to predict market direction.
    
    Logic:
    - Long (1.0) if Model predicts Up (Class 1)
    - Cash (0.0) if Model predicts Down (Class 0)
    
    Parameters
    ----------
    data : pd.DataFrame
        Must contain the necessary feature columns used by the model.
    model : sklearn estimator
        Trained model with a .predict() method.
    feature_cols : list[str]
        List of column names expected by the model.
        
    Returns
    -------
    pd.Series
        Position series (1.0 or 0.0).
    """
    # Verify features exist
    missing_cols = [c for c in feature_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Data is missing features required by model: {missing_cols}")
        
    # Predict (returns 0 or 1)
    # We use .predict(X) directly on the feature subset
    X = data[feature_cols]
    
    # Handle NaNs (Model can't predict on NaNs). 
    # Strategy: Stay out of market (0) if data is missing.
    # We'll fill NaNs with 0 temporarily just to run predict without error, 
    # but then mask the result to 0 position for those rows.
    # A cleaner approach in production is to dropna, but backtest engine needs aligned index.
    
    # Better approach: Predict only on valid rows
    valid_mask = X.notna().all(axis=1)
    
    predictions = pd.Series(0.0, index=data.index)
    
    if valid_mask.any():
        X_valid = X[valid_mask]
        preds_valid = model.predict(X_valid)
        predictions[valid_mask] = preds_valid.astype(float)
    
    predictions.name = "ml_strategy_rf"
    return predictions


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
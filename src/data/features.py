import pandas as pd
import numpy as np

def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates technical features for Machine Learning models.
    
    Features created:
    - Returns: 1-day, 5-day returns.
    - Volatility: 5-day rolling standard deviation.
    - Momentum: Difference between Price and Moving Averages.
    - Lagged Returns: Returns from previous days (to predict today).
    
    Args:
        data (pd.DataFrame): DataFrame with 'Close' column.
        
    Returns:
        pd.DataFrame: Original data enriched with feature columns.
                      Rows with NaNs (due to rolling windows) are dropped.
    """
    df = data.copy()
    
    # Ensure we are working with 1D Series for calculations
    # This prevents "Per-column arrays must each be 1-dimensional" error
    # if the dataframe has MultiIndex columns or duplicate names.
    close_series = df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0] # Take first column if duplicate
    
    # Ensure we have returns
    if "Return" not in df.columns:
        df["Return"] = close_series.pct_change()
    
    # Use the safe series for lags and calcs
    ret_series = df["Return"]
        
    # --- 1. Momentum / Trends ---
    # Return Lags (The most important features: what happened yesterday?)
    df["Return_Lag1"] = ret_series.shift(1)
    df["Return_Lag2"] = ret_series.shift(2)
    df["Return_Lag5"] = ret_series.shift(5)
    
    # --- 2. Moving Averages Distances ---
    # Is the price above or below its average? (Normalized by price)
    ma_10 = close_series.rolling(window=10).mean()
    ma_50 = close_series.rolling(window=50).mean()
    
    df["Dist_MA10"] = (close_series - ma_10) / ma_10
    df["Dist_MA50"] = (close_series - ma_50) / ma_50
    
    # --- 3. Volatility ---
    # Rolling standard deviation of returns
    df["Vol_5d"] = ret_series.rolling(window=5).std()
    
    # --- 4. RSI (Relative Strength Index) ---
    # Standard 14-day RSI
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # --- 5. Relative Volume (RVOL) ---
    # Volume today / Average Volume of last 20 days
    # Avoid division by zero if volume is missing
    if "Volume" in df.columns:
        vol_series = df["Volume"]
        if isinstance(vol_series, pd.DataFrame):
             vol_series = vol_series.iloc[:, 0]
             
        vol_ma = vol_series.rolling(window=20).mean()
        df["RVOL"] = vol_series / vol_ma

    # --- 6. Target Variable (What we want to predict) ---
    # We want to predict if TOMORROW's return will be positive.
    # So we shift returns BACKWARDS by 1 day.
    # Target = 1 if Return(t+1) > 0, else 0
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
    
    # Remove rows with NaNs (created by lags/rolling windows)
    df = df.dropna()
    
    return df

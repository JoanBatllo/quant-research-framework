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
        # Avoid division by zero
        vol_ma = vol_ma.replace(0, np.nan) 
        df["RVOL"] = vol_series / vol_ma

    # --- 6. MACD (Trend) -> Changed to PPO (Percentage Price Oscillator) ---
    # Standard MACD uses absolute price diffs, which is bad for multi-asset training (BTC vs SPY).
    # PPO = ((12-day EMA - 26-day EMA) / 26-day EMA) * 100
    ema_12 = close_series.ewm(span=12, adjust=False).mean()
    ema_26 = close_series.ewm(span=26, adjust=False).mean()
    
    # Avoid zero division
    ema_26 = ema_26.replace(0, np.nan)
    
    ppo_line = ((ema_12 - ema_26) / ema_26) * 100
    signal_line = ppo_line.ewm(span=9, adjust=False).mean()
    
    # Feature: PPO Histogram (Distance between PPO and Signal)
    df["PPO_Hist"] = ppo_line - signal_line
    # We keep "MACD_Hist" name purely for backwards compatibility if needed, 
    # but strictly it is now PPO. Let's start using new names or alias.
    # ideally we rename column to PPO to be explicit.
    df["PPO_Line"] = ppo_line

    # --- 7. Bollinger Bands (Volatility & Mean Reversion) ---
    # 20-day SMA +/- 2 Std Dev
    bb_mid = close_series.rolling(window=20).mean()
    bb_std = close_series.rolling(window=20).std()
    
    bb_upper = bb_mid + (2 * bb_std)
    bb_lower = bb_mid - (2 * bb_std)
    
    # Feature: %B (Where is price relative to bands? 0=Lower, 1=Upper, >1=Overbought)
    # Avoid zero division
    bb_range = bb_upper - bb_lower
    bb_range = bb_range.replace(0, np.nan)
    df["BB_PctB"] = (close_series - bb_lower) / bb_range
    
    # Feature: Band Width (Volatility Squeeze)
    df["BB_Width"] = bb_range / bb_mid

    # --- 8. ATR (Average True Range) ---
    # Measures market "energy" or true volatility range
    if "High" in df.columns and "Low" in df.columns:
        high_s = df["High"]
        low_s = df["Low"]
        # Safety check for duplicates
        if isinstance(high_s, pd.DataFrame): high_s = high_s.iloc[:, 0]
        if isinstance(low_s, pd.DataFrame): low_s = low_s.iloc[:, 0]
        
        prev_close = close_series.shift(1)
        
        tr1 = high_s - low_s
        tr2 = (high_s - prev_close).abs()
        tr3 = (low_s - prev_close).abs()
        
        # True Range is max of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # Normalize ATR by price so it's comparable across time
        df["ATR_Rel"] = atr / close_series

    # --- 9. Target Variable (What we want to predict) ---
    # We want to predict if TOMORROW's return will be SIGNIFICANTLY positive.
    # Prediction: Will price rise > 0.1%? (Filter out noise)
    # Target = 1 if Return(t+1) > 0.001, else 0
    df["Target"] = (df["Return"].shift(-1) > 0.001).astype(int)
    
    # Remove rows with NaNs (created by lags/rolling windows)
    df = df.dropna()
    
    return df

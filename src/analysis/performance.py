import numpy as np
import pandas as pd

def calculate_total_return(equity_curve: pd.Series) -> float:
    """
    Calculates the total return percentage of the equity curve.
    
    Args:
        equity_curve (pd.Series): Series of equity values over time.
        
    Returns:
        float: Total return as a percentage (e.g., 20.5 for 20.5%).
    """
    if equity_curve.empty:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

def calculate_cagr(equity_curve: pd.Series) -> float:
    """
    Calculates the Compound Annual Growth Rate (CAGR).
    
    Args:
        equity_curve (pd.Series): Series of equity values with a DatetimeIndex.
        
    Returns:
        float: CAGR as a percentage.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]
    years = (end_date - start_date).days / 365.25
    
    if years == 0:
        return 0.0
        
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    cagr = (total_return ** (1 / years) - 1) * 100
    return cagr

def calculate_volatility(returns: pd.Series, annualization_factor: int = 252) -> float:
    """
    Calculates the annualized volatility.
    
    Args:
        returns (pd.Series): Series of periodic returns (e.g., daily returns).
        annualization_factor (int): Number of periods in a year (default 252 for daily).
        
    Returns:
        float: Annualized volatility as a percentage.
    """
    if returns.empty:
        return 0.0
    return returns.std() * np.sqrt(annualization_factor) * 100

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculates the annualized Sharpe Ratio.
    
    Args:
        returns (pd.Series): Series of periodic returns.
        risk_free_rate (float): Annualized risk-free rate (decimal, e.g., 0.02 for 2%).
        annualization_factor (int): Number of periods in a year.
        
    Returns:
        float: Sharpe Ratio.
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    
    # Adjust risk-free rate to the period of the returns (e.g., daily)
    # Simple approximation: rf_daily = rf_annual / 252
    rf_per_period = risk_free_rate / annualization_factor
    
    excess_returns = returns - rf_per_period
    mean_excess_return = excess_returns.mean()
    std_dev = returns.std()
    
    return (mean_excess_return / std_dev) * np.sqrt(annualization_factor)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculates the Maximum Drawdown.
    
    Args:
        equity_curve (pd.Series): Series of equity values.
        
    Returns:
        float: Maximum drawdown as a positive percentage (e.g., 15.5 for 15.5% drop).
    """
    if equity_curve.empty:
        return 0.0
        
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()  # This will be negative, e.g., -0.15
    
    return abs(max_drawdown) * 100

def summarize_performance(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Computes a summary of performance metrics.
    
    Args:
        equity_curve (pd.Series): Series of equity values with DatetimeIndex.
        risk_free_rate (float): Annualized risk-free rate.
        
    Returns:
        dict: Dictionary containing all performance metrics.
    """
    # Calculate returns from equity curve
    returns = equity_curve.pct_change().dropna()
    
    return {
        "Total Return (%)": calculate_total_return(equity_curve),
        "CAGR (%)": calculate_cagr(equity_curve),
        "Volatility (%)": calculate_volatility(returns),
        "Sharpe Ratio": calculate_sharpe_ratio(returns, risk_free_rate),
        "Max Drawdown (%)": calculate_max_drawdown(equity_curve)
    }

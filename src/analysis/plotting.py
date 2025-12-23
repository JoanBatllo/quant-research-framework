import matplotlib.pyplot as plt
import pandas as pd
from src.backtesting.engine import BacktestResult

def plot_performance(result: BacktestResult, title: str = "Backtest Results"):
    """
    Plots the equity curve and drawdown chart.
    
    Args:
        result (BacktestResult): The result object from run_backtest.
        title (str): Title of the plot.
    """
    equity = result.equity_curve
    
    # Calculate Drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100  # As percentage
    
    # Create figure with 2 subplots (Equity and Drawdown)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot Equity Curve
    ax1.plot(equity.index, equity.values, label="Equity", color="#1f77b4", linewidth=1.5)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Account Value ($)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Plot Drawdown
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.3, label="Drawdown %")
    ax2.plot(drawdown.index, drawdown.values, color="#d62728", linewidth=1)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.show()

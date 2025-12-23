# scripts/run_simple_backtest.py

from src.data.preprocessing import prepare_price_data
from src.backtesting.strategies import always_long
from src.backtesting.engine import run_backtest


def main():
    # 1) Get clean data with returns
    data = prepare_price_data()

    # 2) Generate strategy signal (always fully invested)
    signal = always_long(data)

    # 3) Run backtest
    result = run_backtest(data, signal)

    # 4) Calculate and print performance metrics
    from src.analysis.performance import summarize_performance
    
    print("=== Simple Backtest: Always Long ===")
    print("Start date:", data.index.min().date())
    print("End date:  ", data.index.max().date())
    
    # Calculate metrics
    metrics = summarize_performance(result.equity_curve)
    
    print("\n--- Performance Metrics ---")
    for metric, value in metrics.items():
        if "Ratio" in metric:
             print(f"{metric:<20}: {value:.2f}")
        else:
             print(f"{metric:<20}: {value:.2f}")
    print("\nSample of equity curve:")
    print(result.equity_curve.head())
    print("...")
    print(result.equity_curve.tail())

    # 5) Visualize results
    from src.analysis.plotting import plot_performance
    print("\n[INFO] Displaying performance plot...")
    plot_performance(result, title="Strategy: Always Long (SPY)")

if __name__ == "__main__":
    main()
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

    # 4) Print basic info
    print("=== Simple Backtest: Always Long ===")
    print("Start date:", data.index.min().date())
    print("End date:  ", data.index.max().date())
    print("Initial cash:", result.equity_curve.iloc[0])
    print("Final equity:", result.equity_curve.iloc[-1])
    total_return = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1
    print(f"Total return: {total_return:.2%}")
    print("\nSample of equity curve:")
    print(result.equity_curve.head())
    print("...")
    print(result.equity_curve.tail())


if __name__ == "__main__":
    main()
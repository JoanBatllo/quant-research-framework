# scripts/check_config.py

from src.utils.config import get_config


def main():
    cfg = get_config()

    print("Project name:", cfg["project"]["name"])
    print("Default symbol:", cfg["data"]["default_symbol"])
    print("Backtest initial cash:", cfg["backtest"]["initial_cash"])
    print("Risk-free rate:", cfg["risk"]["risk_free_rate_annual"])
    print("Raw data directory:", cfg["data"]["raw_dir"])


if __name__ == "__main__":
    main()
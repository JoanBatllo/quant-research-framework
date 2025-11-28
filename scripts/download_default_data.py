# scripts/download_default_data.py

from src.data.loaders import get_price_data


def main():
    """
    Simple script to download the default symbol's data using the
    configuration in config/settings.yaml and print a quick summary.

    This is mainly for manual testing and learning.
    """
    data = get_price_data()

    print("Downloaded data shape:", data.shape)
    print("First rows:")
    print(data.head())
    print("\nLast rows:")
    print(data.tail())


if __name__ == "__main__":
    main()
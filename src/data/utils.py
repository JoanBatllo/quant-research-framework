import pandas as pd
from src.data.loaders import get_price_data
from src.data.preprocessing import standardize_price_dataframe
from src.data.features import generate_features

def load_multi_asset_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads, processes, and stacks data for multiple assets.
    
    This creates a 'Global Dataset' where each row is a (Date, Ticker) observation.
    Features are generated PER ASSET to ensure statistics (RSI, Volatility) are 
    relative to that asset's history.
    
    Args:
        tickers: List of ticker symbols (e.g., ['SPY', 'QQQ', 'BTC-USD']).
        start_date: Start date string.
        end_date: End date string.
        
    Returns:
        pd.DataFrame: A single stacked DataFrame with a 'Ticker' column.
                      Index is DatetimeIndex (can have duplicate dates).
    """
    global_df = []
    
    for t in tickers:
        try:
            # 1. Fetch
            df = get_price_data(t, start_date, end_date)
            
            # 2. Standardize
            df = standardize_price_dataframe(df, symbol=t)
            
            # 3. Generate Features (Locally!)
            # Crucial: Z-Scores, RSI, etc are calculated here per asset
            df_feats = generate_features(df)
            
            # 4. Tag it
            df_feats['Ticker'] = t
            
            global_df.append(df_feats)
            
        except Exception as e:
            print(f"Warning: Failed to load data for {t}: {e}")
            continue
            
    if not global_df:
        raise ValueError("No data could be loaded for any of the provided tickers.")
        
    # Stack them vertically
    final_df = pd.concat(global_df, axis=0)
    
    # Sort by Date (helps TimeSeriesSplit later)
    final_df = final_df.sort_index()
    
    return final_df

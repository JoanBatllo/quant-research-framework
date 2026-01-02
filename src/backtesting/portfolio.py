import pandas as pd
import numpy as np

class VectorizedBacktester:
    """
    A fast, vectorized backtester for multi-asset strategies.
    
    Instead of iterating row-by-row (slow), we use matrix operations (fast).
    It takes a DataFrame of Signal Probabilities and returns a Portfolio Equity Curve.
    """
    
    def __init__(self, probabilities: pd.DataFrame, returns: pd.DataFrame):
        """
        Args:
            probabilities: DataFrame (Date x Ticker) with values 0..1 (Model Prediction).
            returns: DataFrame (Date x Ticker) with daily asset returns.
        """
        self.probs = probabilities
        self.returns = returns
        
        # Align data (intersection of dates and columns)
        common_idx = self.probs.index.intersection(self.returns.index)
        # Ensure we only use columns present in both
        common_cols = [c for c in self.probs.columns if c in self.returns.columns]
        
        self.probs = self.probs.loc[common_idx, common_cols]
        self.returns = self.returns.loc[common_idx, common_cols]

    def run_top_n_strategy(self, n: int = 3, transaction_cost_bps: float = 10.0) -> pd.DataFrame:
        """
        Simulates a strategy that buys the 'N' assets with highest probability each day.
        
        Args:
            n: Number of assets to hold.
            transaction_cost_bps: Cost per trade in basis points.
            
        Returns:
            pd.DataFrame: Portfolio metrics (Daily Return, Equity Curve).
        """
        if self.probs.empty:
             raise ValueError("No overlapping data found between probabilities and returns.")

        # 1. Rank Assets Daily (1 = Highest Probability, 2 = Second, etc.)
        # method='first' handles ties arbitrarily but consistently
        # ascending=False because Higher prob is better
        ranks = self.probs.rank(axis=1, ascending=False, method='first')
        
        # 2. Allocate Weights
        # 1/N for Top N, 0 otherwise
        target_weights = (ranks <= n).astype(float)
        
        # Normalize weights (if N > available assets, split evenly among available)
        row_sums = target_weights.sum(axis=1)
        # Avoid division by zero
        row_sums = row_sums.replace(0, 1) 
        target_weights = target_weights.div(row_sums, axis=0)
        
        # 3. Shift weights (We calculate signal at Close t, trade at Open t+1)
        # In this simple model, we assume we trade at Close t and earn Return t+1
        # So we align Position(t) with Return(t+1)
        # position(t) is determined by signal(t-1)? 
        # Actually, standard convention: Signal(t-1) -> Position(t) -> Return(t)
        # Our `generate_features` uses `shift(-1)` for target, meaning prediction at t is for t+1 return.
        # So, Probs(t) predicts Return(t+1).
        # Therefore, if we hold positions based on Probs(t), we capture Return(t+1).
        # We need to shift weights forward by 1 day to align with the returns they generate.
        
        # Weight at T captures Return at T (if we execute at Open).
        # If probs index is T, it predicts T+1. So we want Weight(T+1) to be based on Probs(T).
        portfolio_weights = target_weights.shift(1).fillna(0.0)
        
        # 4. Calculate Gross Returns
        # Element-wise multiplication: Weight(t, Asset) * Return(t, Asset)
        weighted_returns = portfolio_weights * self.returns
        portfolio_ret_gross = weighted_returns.sum(axis=1)
        
        # 5. Transaction Costs
        # Turnover = Sum of absolute change in weights per asset
        turnover = portfolio_weights.diff().abs().sum(axis=1).fillna(0.0)
        cost = turnover * (transaction_cost_bps / 10000.0)
        
        # 6. Net Returns
        portfolio_ret_net = portfolio_ret_gross - cost
        
        # 7. Equity Curve
        equity_curve = (1 + portfolio_ret_net).cumprod()
        
        # Start at 1.0 (or 100k, doesn't matter for curve shape)
        if not equity_curve.empty:
            equity_curve = equity_curve / equity_curve.iloc[0] 
        
        return pd.DataFrame({
            'Return': portfolio_ret_net,
            'Equity': equity_curve,
            'Turnover': turnover
        })

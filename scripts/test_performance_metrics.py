import unittest
import pandas as pd
import numpy as np
from src.analysis.performance import (
    calculate_total_return,
    calculate_cagr,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    summarize_performance
)

class TestPerformanceMetrics(unittest.TestCase):
    def setUp(self):
        # Create a simple equity curve: 100 -> 110 -> 121 (10% growth each step)
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        self.simple_curve = pd.Series([100.0, 110.0, 121.0], index=dates)
        
        # Create a volatile curve: 100 -> 150 -> 75
        dates_vol = pd.date_range(start="2023-01-01", periods=3, freq="D")
        self.volatile_curve = pd.Series([100.0, 150.0, 75.0], index=dates_vol)

    def test_total_return(self):
        # 100 -> 121 is 21% return
        self.assertAlmostEqual(calculate_total_return(self.simple_curve), 21.0)
        
        # 100 -> 75 is -25% return
        self.assertAlmostEqual(calculate_total_return(self.volatile_curve), -25.0)

    def test_max_drawdown(self):
        # Simple curve never drops, so DD is 0
        self.assertAlmostEqual(calculate_max_drawdown(self.simple_curve), 0.0)
        
        # Volatile curve: 150 -> 75 is a 50% drop
        self.assertAlmostEqual(calculate_max_drawdown(self.volatile_curve), 50.0)

    def test_cagr_simple(self):
        # 2 days is approx 0.00547 years
        # 21% return in 2 days is a huge CAGR
        # (1.21)^(1/0.00547) - 1
        # We just check it runs and is positive
        cagr = calculate_cagr(self.simple_curve)
        self.assertTrue(cagr > 0)

    def test_volatility(self):
        returns = pd.Series([0.01, -0.01, 0.01, -0.01]) # Low vol
        vol = calculate_volatility(returns)
        self.assertTrue(vol > 0)

    def test_sharpe_ratio(self):
        # Constant positive return -> Infinite Sharpe (std=0) -> handled as 0 in code or high?
        # Our code returns 0 if std is 0.
        const_returns = pd.Series([0.01, 0.01, 0.01])
        self.assertEqual(calculate_sharpe_ratio(const_returns), 0.0)
        
        # Normal returns
        returns = pd.Series([0.01, 0.02, -0.005, 0.015])
        sharpe = calculate_sharpe_ratio(returns)
        self.assertTrue(sharpe > 0)

    def test_summarize(self):
        summary = summarize_performance(self.simple_curve)
        self.assertIn("Total Return (%)", summary)
        self.assertIn("Sharpe Ratio", summary)
        self.assertAlmostEqual(summary["Total Return (%)"], 21.0)

if __name__ == '__main__':
    unittest.main()

# Glossary of Terms

A reference guide for the financial and technical terminology used in this project.

## 📈 Quantitative Finance

### **Backtesting**
The process of simulating a trading strategy using historical data to verify its profitability and risk profile before risking real capital.

### **CAGR (Compound Annual Growth Rate)**
The mean annual growth rate of an investment over a specified time period longer than one year. It represents the smooth trajectory of growth.

### **Drawdown**
The peak-to-trough decline during a specific period for an investment.
*   **Max Drawdown (MDD):** The largest percentage drop from a peak to a trough in the portfolio's history. A key measure of risk.

### **Equity Curve**
A graphical representation of the change in the value of a trading account over time.

### **Sharpe Ratio**
A measure of risk-adjusted return. It is calculated as `(Expected Return - Risk Free Rate) / Standard Deviation`.
*   *Interpretation:* >1 is good, >2 is very good, >3 is excellent.

### **Volatility**
A statistical measure of the dispersion of returns. In this project, we typically refer to annualized volatility (Std Dev of returns * sqrt(252)).

### **Transaction Costs**
Fees paid to execute a trade (commissions, spread, slippage). Ignoring these in backtests leads to overly optimistic results.

---

## 🤖 Machine Learning & Data

### **Feature Engineering**
The process of using domain knowledge to extract features (characteristics) from raw data.
*   *Example:* Creating a "Momentum" feature by subtracting the 50-day moving average from the current price.

### **Target (Label)**
The variable we want to predict. In our case, `Target = 1` if tomorrow's price is higher than today's, and `0` otherwise.

### **Train / Test Split**
Separating data into two sets: one to teach the model (Train), and one to evaluate it (Test). In time-series finance, this **must** be done chronologically (e.g., Train: 2010-2022, Test: 2023-2024) to avoid "Look-ahead Bias".

### **Look-ahead Bias**
A critical error in backtesting where the strategy uses information that would not have been available at the decision time (e.g., using tomorrow's closing price to decide today's trade).

### **Overfitting**
When a model learns the "noise" of the training data instead of the underlying pattern. It performs perfectly on past data but fails on new (test) data.

### **Confusion Matrix**
A table layout that allows visualization of the performance of a classification algorithm.
*   **True Positive:** Model predicted UP, and market went UP.
*   **False Positive:** Model predicted UP, but market went DOWN (Lost money).
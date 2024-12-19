import numpy as np
import pandas as pd

assets = ["Equity1", "Bond1"]

returns_df = pd.DataFrame({
    "Equity1": [0.01, 0.01, -0.01, 0.02],  
    "Bond1": [0.002, 0.002, 0.002, 0.003]
}, index=pd.date_range("2014-01-01", periods=4, freq='ME'))  # Example monthly returns


alpha = 0.95  # Confidence level for CVaR
SCR_threshold = 0.1  # Solvency Capital Requirement threshold

# Convert future returns DataFrame to numpy array
future_returns = returns_df.to_numpy()

n = len(assets)

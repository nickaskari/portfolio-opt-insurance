import cvxpy as cp
import numpy as np
import pandas as pd
from risk_measures import calculate_risk

# Example Data
assets = ["Equity1", "Bond1"]
returns_df = pd.DataFrame({
    "Equity1": [0.01, 0.02, -0.01, 0.03],  # Example future returns (01/01/2014 to 01/01/2015)
    "Bond1": [0.005, 0.004, 0.002, 0.003]
}, index=pd.date_range("2014-01-01", periods=4, freq='ME'))  # Example monthly returns

# Example parameters
alpha = 0.95  # Confidence level for CVaR
n = len(assets)  # Number of assets
future_returns = returns_df.to_numpy()  # Actual future return data (e.g., 01/01/2014 to 01/01/2015)

# Check shapes
print("Future Returns Shape:", future_returns.shape)
assert future_returns.shape[1] == n, "Number of assets does not match the returns data."

# Optimization variables (allocations to each asset)
allocations = cp.Variable(n)

# Calculate future returns from 01/01/2014 onward
risk_expression, risk_constraints = calculate_risk(allocations, future_returns, alpha, risk_measure="CVaR")

# Debug risk_expression
print("Risk Expression (before solving):", risk_expression)

# Solvency II constraint: Ensure CVaR or another risk measure stays below a solvency threshold
SCR_threshold = 0.05  # Solvency Capital Requirement (set as a threshold for the risk measure)
solvency_constraint = risk_expression <= SCR_threshold

# Objective: Maximize expected return while minimizing risk (here CVaR, but it could be any risk measure)
expected_return = future_returns.mean(axis=0) @ allocations
print("Expected Return (before solving):", expected_return)

# Set up the optimization problem with constraints
constraints = [
    cp.sum(allocations) == 1,  # Weights sum to 100%
    allocations >= 0           # No short-selling (non-negative weights)
] + risk_constraints + [solvency_constraint]

# Define the optimization problem (maximize expected return minus risk)
problem = cp.Problem(cp.Maximize(expected_return - risk_expression), constraints)

# Solve the problem
problem.solve()

# Check problem status and print results
print("Problem Status:", problem.status)
if problem.status == "optimal":
    optimal_allocations = allocations.value
    print("Optimal Allocations:", optimal_allocations)
else:
    print("No optimal solution found. Status:", problem.status)
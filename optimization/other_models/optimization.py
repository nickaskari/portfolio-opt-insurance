import cvxpy as cp
import numpy as np
import pandas as pd
from optimization.old_shit.risk_measures import calculate_risk
from optimization.old_shit.parameters import assets, future_returns, alpha, SCR_threshold, n 

# Optimization variables (allocations to each asset)
allocations = cp.Variable(n)

risk_expression, risk_constraints = calculate_risk(allocations, future_returns, alpha, risk_measure="CVaR")

# Solvency II constraint: Ensure CVaR or another risk measure stays below a solvency threshold
SCR_threshold = 0.1  
solvency_constraint = risk_expression <= SCR_threshold

expected_return = future_returns.mean(axis=0) @ allocations

constraints = [
    cp.sum(allocations) == 1,  # Weights sum to 100%
    allocations >= 0           # No short-selling (non-negative weights)
] + risk_constraints + [solvency_constraint]

problem = cp.Problem(cp.Maximize(expected_return - risk_expression), constraints)

problem.solve()

print("Problem Status:", problem.status)
if problem.status == "optimal":
    optimal_allocations = allocations.value
    print("Optimal Allocations:", optimal_allocations)

    losses = -future_returns @ optimal_allocations
    print("Losses (after solving):", losses)

    print(risk_expression.value)
else:
    print("No optimal solution found. Status:", problem.status)


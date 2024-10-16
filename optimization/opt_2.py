import pandas as pd
import numpy as np
from scipy.optimize import minimize
import json

# Load price data from asset_classes.csv
prices_df = pd.read_csv('../data/asset_classes.csv', index_col=0)
prices_df.index = pd.to_datetime(prices_df.index)

# Step 1: Calculate returns from price data
returns_df = prices_df.pct_change().dropna()

# Step 2: Calculate expected returns (mean) and covariance matrix
expected_returns = returns_df.mean()
cov_matrix = returns_df.cov()

# Number of assets
n_assets = len(expected_returns)
risk_measure = 'CVaR'

# Define liability (could be a constant or stochastic; for now assume constant)
liabilities = 100  # Example liability estimate

# Function to calculate portfolio value
def calculate_portfolio_value(weights, initial_investment=1000):
    return initial_investment * (1 + np.dot(returns_df, weights))

# Step 3: Calculate CVaR of Basic Own Funds (BOF)
def calculate_bof_cvar(weights, alpha=0.95):
    # Step 3.1: Calculate portfolio value based on weights
    portfolio_values = calculate_portfolio_value(weights)
    
    # Step 3.2: Calculate BOF (Assets - Liabilities)
    bof_values = portfolio_values - liabilities
    
    # Step 3.3: Calculate Value at Risk (VaR) at alpha confidence level
    var = np.percentile(bof_values, 100 * (1 - alpha))
    
    # Step 3.4: Calculate CVaR (average loss beyond VaR)
    cvar = np.mean(bof_values[bof_values <= var])
    
    # Return the negative of CVaR as a risk measure (since losses are negative)
    return -cvar

# Step 4: RoRAC objective function (maximize return / CVaR of BOF)
def roRAC_objective(weights, expected_returns):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_cvar = calculate_bof_cvar(weights)
    roRAC = portfolio_return / (-portfolio_cvar)  # Negative to maximize RoRAC
    
    return -roRAC  # Minimize the negative RoRAC to maximize it

# Step 5: Constraints
# Budget constraint (weights sum to 1)
def budget_constraint(weights):
    return np.sum(weights) - 1.0

# No short-selling constraint (weights >= 0)
bounds = [(0, 1) for _ in range(n_assets)]

# SCR constraint using CVaR of BOF
SCR_max = 50  # Set an arbitrary SCR_max for CVaR

def scr_constraint(weights):
    portfolio_cvar = calculate_bof_cvar(weights)
    return SCR_max - (-portfolio_cvar)  # Ensure CVaR <= SCR_max

# Step 6: Optimization
# Initial guess (equal weighting)
initial_weights = np.ones(n_assets) / n_assets

# Set up the constraints as a dictionary
constraints = (
    {'type': 'eq', 'fun': budget_constraint},  # Sum of weights = 1
    {'type': 'ineq', 'fun': scr_constraint}  # CVaR <= SCR_max
)

# Run the optimization
result = minimize(
    roRAC_objective, 
    initial_weights, 
    args=(expected_returns,), 
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

# Extract the optimal portfolio weights
optimal_weights = result.x

# Display the results
print("Optimal Portfolio Weights:", optimal_weights)
print("Expected Portfolio Return:", np.dot(optimal_weights, expected_returns))
print("Portfolio SCR (CVaR of BOF):", calculate_bof_cvar(optimal_weights))

results = {
    'optimization_run': 1,
    'optimal_weights': optimal_weights.tolist(), 
    'expected_return': np.dot(optimal_weights, expected_returns),  
    'SCR_risk': calculate_bof_cvar(optimal_weights),  
    'risk_measure' : risk_measure,

}

with open('optimization_results.json', 'w') as f:
    json.dump(results, f, indent=4)

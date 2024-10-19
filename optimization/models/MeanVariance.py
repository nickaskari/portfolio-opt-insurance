import numpy as np
import pandas as pd
from scipy.optimize import minimize

class MeanVarianceOptimizer:
    """
    Class for portfolio optimization using the Mean-Variance method.
    The goal is to maximize portfolio return subject to a variance constraint.
    """
    
    def __init__(self, Vmax, n_assets=None):

        self.prices_df, self.returns_df, self.expected_returns, self.covariance_matrix = self.init_df_values(n_assets)
        self.Vmax = Vmax
        self.n_assets = len(self.expected_returns)
    
    def fetch_data_df(self, n_assets):

        df = pd.read_csv('../../data/asset_classes.csv', index_col=0)
        df.index = pd.to_datetime(df.index)

        if n_assets:
            return df.iloc[:, :n_assets]  # Return the first n_assets columns

        return df
    
    def init_df_values(self, n_assets):

        prices_df = self.fetch_data_df(n_assets)
        returns_df = prices_df.pct_change().dropna()  # Calculate returns
        expected_returns = returns_df.mean()  # Mean returns for each asset
        covariance_matrix = returns_df.cov()  # Covariance matrix for the assets

        return prices_df, returns_df, expected_returns, covariance_matrix

    def portfolio_return(self, weights):

        return np.dot(weights, self.expected_returns)
    
    def portfolio_variance(self, weights):

        return np.dot(weights.T, np.dot(self.covariance_matrix, weights))
    
    def variance_constraint(self, weights):

        return self.Vmax - self.portfolio_variance(weights)
    
    def budget_constraint(self, weights):

        return np.sum(weights) - 1.0
    
    def optimize(self):
        initial_weights = np.ones(self.n_assets) / self.n_assets

        constraints = (
            {'type': 'eq', 'fun': self.budget_constraint},  # Sum of weights = 1
            {'type': 'ineq', 'fun': self.variance_constraint}  # Variance <= Vmax
        )

        bounds = [(0, 1) for _ in range(self.n_assets)]

        # Run the optimization
        result = minimize(
            lambda w: -self.portfolio_return(w),  # Maximize return (negative for minimization)
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )

        self.optimal_weights = result.x
        return result

# Example usage:
# Adjust the number of assets (n_assets) and maximum variance (Vmax) as needed
Vmax = 0.2  # Example maximum variance
optimizer = MeanVarianceOptimizer(Vmax, n_assets=None)  # Use None for all assets or set a number for partial

# Run the optimization
result = optimizer.optimize()

# Output the optimal weights and portfolio statistics
print("Optimal Portfolio Weights:", optimizer.optimal_weights)
print("Expected Portfolio Return:", optimizer.portfolio_return(optimizer.optimal_weights))
print("Portfolio Variance:", optimizer.portfolio_variance(optimizer.optimal_weights))

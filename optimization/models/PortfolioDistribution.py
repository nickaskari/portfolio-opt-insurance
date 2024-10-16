import numpy as np
import pandas as pd
from arch import arch_model

class PortfolioDistribution:
    def __init__(self, weights, returns_df, initial_portfolio_value, rescale_factor=10):
        """
        Initialize the PortfolioDistribution class with portfolio weights and asset returns data.

        Parameters:
        - weights: A NumPy array of portfolio weights.
        - returns_df: A pandas DataFrame containing historical returns for the assets.
        - initial_portfolio_value: Initial value of the portfolio
        """
        self.weights = weights
        self.returns_df = returns_df
        self.mean_returns = returns_df.mean()  # Mean of historical returns
        self.cov_matrix = returns_df.cov()     # Covariance matrix of returns
        self.initial_portfolio_value = initial_portfolio_value
        self.rescale_factor = rescale_factor  
    
    def historical_portfolio_returns(self):
        """
        Calculates the value of the portfolio based on the provided weights and the historical returns.
        
        Returns:
        - Portfolio value after applying the asset returns.
        """

        return self.initial_portfolio_value * (1 + np.dot(self.returns_df, self.weights))

    def simulate_monte_carlo(self, num_simulations=10000, time_horizon=1):
        """
        Simulate future portfolio values using Monte Carlo simulations.

        Parameters:
        - num_simulations: Number of Monte Carlo simulations (default is 10,000).
        - time_horizon: The number of periods to simulate (default is 1 period).

        Returns:
        - Simulated portfolio values (NumPy array).
        """
        # Simulate future returns using a multivariate normal distribution
        random_returns = np.random.multivariate_normal(self.mean_returns, self.cov_matrix, num_simulations)
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = np.dot(random_returns, self.weights)
        
        portfolio_values = self.initial_portfolio_value * np.exp(portfolio_returns * time_horizon)
        
        return portfolio_values

    def simulate_garch(self, num_simulations=5, time_horizon=1):
        """
        Simulate future portfolio values using a GARCH(1,1) model for each asset.

        Parameters:
        - num_simulations: Number of simulations (default is 5).
        - time_horizon: The number of periods to simulate (default is 1 period).

        Returns:
        - Simulated portfolio values (NumPy array).
        """
        simulated_returns = []
        
        for i in range(self.returns_df.shape[1]):
            asset_returns = self.returns_df.iloc[:, i]
            
            # Rescale the asset returns by the rescale_factor
            scaled_returns = asset_returns * self.rescale_factor
            
            # Fit a GARCH(1, 1) model to the rescaled returns
            garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
            garch_fit = garch_model.fit(disp='off')
            
            # Forecast future volatility and returns
            garch_forecast = garch_fit.forecast(horizon=time_horizon, start=None, reindex=False)
            
            # Get forecasted conditional volatility
            forecast_volatility = garch_forecast.variance.iloc[-1].values
            
            # Generate random shocks for each simulation and undo rescaling
            shocks = np.random.normal(0, np.sqrt(forecast_volatility), (num_simulations, time_horizon))
            sim_returns = (garch_fit.params['mu'] + shocks) / self.rescale_factor  # Undo rescaling
            
            simulated_returns.append(sim_returns)

        simulated_returns = np.array(simulated_returns).T  # Transpose to match dimensions
        portfolio_returns = np.dot(simulated_returns, self.weights)
        portfolio_values = self.initial_portfolio_value * np.exp(portfolio_returns)
        
        return portfolio_values
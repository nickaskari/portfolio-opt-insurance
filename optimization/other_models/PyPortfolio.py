import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

class PyPortfolioOptimizer:
    """
    Class for portfolio optimization using the PyPortfolioOpt library.
    This method uses mean-variance optimization with built-in tools for risk and return analysis.
    """
    
    def __init__(self, n_assets=None):
        """
        Initializes the optimizer with asset data.
        
        Parameters:
        - n_assets: Number of assets to include in the optimization.
        """
        self.prices_df, self.returns_df = self.init_df_values(n_assets)
        self.expected_returns = expected_returns.mean_historical_return(self.prices_df)
        self.cov_matrix = risk_models.sample_cov(self.prices_df)
    
    def fetch_data_df(self, n_assets):
        """
        Fetches the asset price data from the CSV file.
        """
        df = pd.read_csv('../../data/asset_classes.csv', index_col=0)
        df.index = pd.to_datetime(df.index)

        if n_assets:
            return df.iloc[:, :n_assets]  # Return the first n_assets columns
        return df

    def init_df_values(self, n_assets):
        """
        Initializes and calculates important data structures: asset prices and returns.
        """
        prices_df = self.fetch_data_df(n_assets)
        returns_df = prices_df.pct_change().dropna()  # Calculate returns
        return prices_df, returns_df

    def optimize(self):
        """
        Runs the portfolio optimization using mean-variance optimization.
        """
        ef = EfficientFrontier(self.expected_returns, self.cov_matrix)
        optimal_weights = ef.max_sharpe()  # Maximize the Sharpe ratio (return-to-risk)
        cleaned_weights = ef.clean_weights()

        # Display the portfolio performance
        performance = ef.portfolio_performance(verbose=True)
        
        return cleaned_weights, performance

# Example usage
optimizer = PyPortfolioOptimizer(n_assets=None)  # Use None for all assets or specify a number for fewer assets
weights, performance = optimizer.optimize()

# Output the optimal weights
print("Optimal Portfolio Weights:", weights)

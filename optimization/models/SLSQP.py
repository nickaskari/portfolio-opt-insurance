import pandas as pd
import numpy as np
from scipy.optimize import minimize
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import datetime
from PortfolioDistribution import PortfolioDistribution
from tqdm import tqdm

# We might have to normalize return as cleaning.
def normalize_returns(returns_df):
    return (returns_df - returns_df.min()) / (returns_df.max() - returns_df.min())


class SLSQP:
    """
    Class for portfolio optimization using Sequential Least Squares Programming (SLSQP).
    The optimization maximizes the Return on Risk-Adjusted Capital (RoRAC) subject to 
    constraints.

    MUST ADD CAPABILITY FOR MORE RISK MEASURES.
    """
    
    def __init__(self, risk_measure, liabilities_0, assets_0, eonia_rate, alpha, SCR_max, distribution_method='historical',
                 maxiter=100, n_assets=None):
        """
        Initializes the SLSQP optimizer with the given parameters.
        
        Parameters:
        - risk_measure: The type of risk measure (e.g., 'CVaR').
        - liabilities_0: Initial value of liabilities.
        - assets_0: Initial value of assets.
        - eonia_rate: Eonia rate or risk-free return used to adjust liabilities.
        - alpha: Confidence level for CVaR calculation (e.g., 0.95 for 95% confidence).
        - SCR_max: Maximum allowed SCR (Solvency Capital Requirement).
        - distribution_method: Specifies the method for portfolio distribution ('historical', 'montecarlo', 'garch').
        - n_assets: if 2, you are allowed to visualize.
        """
        
        self.prices_df, self.returns_df, self.expected_returns = self.init_df_values(n_assets)
        self.n_assets = len(self.expected_returns)
        self.returns_df = normalize_returns(self.returns_df)
        
        self.risk_measure = risk_measure 
        self.liabilities_0 = liabilities_0
        self.assets_0 = assets_0
        self.eonia_rate = eonia_rate
        self.alpha = alpha
        self.SCR_max = SCR_max
        self.distribution_method = distribution_method 
        self.pbar = tqdm(desc="Optimization Progress", total=maxiter)
        self.optimal_weights = None
        self.maxiter = maxiter
        
    def fetch_data_df(self, n_assets):
        """
        Fetches and returns the data from an asset prices CSV file.
        This data is used for calculating returns and optimizing the portfolio.
        
        Returns:
        - DataFrame containing the asset prices, with dates as the index.
        """
        df = pd.read_csv('../../data/asset_classes.csv', index_col=0)
        df.index = pd.to_datetime(df.index)

        if n_assets:
            return df.iloc[:, :2]

        return df
    
    def init_df_values(self, n_assets):
        """
        Initializes and calculates important data structures: asset prices, returns, and expected returns.
        
        Returns:
        - prices_df: DataFrame of asset prices.
        - returns_df: DataFrame of asset returns (percentage changes).
        - expected_returns: Series of mean returns for each asset.
        """
        prices_df = self.fetch_data_df(n_assets)
        returns_df = prices_df.pct_change().dropna()
        expected_returns = returns_df.mean()

        return prices_df, returns_df, expected_returns

    def calculate_dynamic_liabilities(self, initial_liability, eonia_rate, time_horizon=1):
        """
        Calculates the liabilities after a given time period, assuming they grow at the Eonia rate.
        
        Parameters:
        - initial_liability: Initial liability value (L_0).
        - eonia_rate: The Eonia rate or risk-free return for adjusting liabilities.
        - time_horizon: The time period for liability growth (default is 1).
        
        Returns:
        - Liabilities after the given time period.
        """

        return initial_liability * (1 + eonia_rate) ** time_horizon

    def calculate_scr(self, weights):
        """
        Calculates the Conditional Value at Risk (CVaR) of the portfolio's Basic Own Funds (BOF).
        BOF is defined as the difference between portfolio values and liabilities.

        Parameters:
        - weights: NumPy array of portfolio weights.
        
        Returns:
        - Negative CVaR of BOF, to be minimized by the optimizer.
        """

        #portfolio_values = self.calculate_portfolio_value(weights)
        liabilities_t = self.calculate_dynamic_liabilities(self.liabilities_0, self.eonia_rate)

        distributor = PortfolioDistribution(weights, self.returns_df, self.assets_0)

        if self.distribution_method == 'historical':
            portfolio_values = distributor.historical_portfolio_returns()
        elif self.distribution_method == 'montecarlo':
            portfolio_values = distributor.simulate_monte_carlo(num_simulations=100, time_horizon=1)
        elif self.distribution_method == 'garch':
            portfolio_values = distributor.simulate_garch(time_horizon=1)
        else:
            raise ValueError(f"Unknown distribution method: {self.distribution_method}")

        bof_value_0 = self.assets_0 - self.liabilities_0
        bof_values_t = portfolio_values - liabilities_t
        
        #self.plot_bof_distribution(portfolio_values)

        var = np.percentile(bof_values_t, 100 * (1 - self.alpha))
        cvar = np.mean(bof_values_t[bof_values_t <= var])
        
        return cvar

    def roRAC_objective(self, weights):

        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_cvar = self.calculate_scr(weights)
        roRAC = portfolio_return / portfolio_cvar
        
        return -roRAC  # Return negative RoRAC so the optimizer maximizes it
    
    def budget_constraint(self, weights):
        return np.sum(weights) - 1.0
    
    def scr_constraint(self, weights):
        portfolio_cvar = self.calculate_scr(weights)
        # Ensure CVaR <= SCR_max
        return self.SCR_max - portfolio_cvar

    def max_weight_constraint(self, weights):         
        return 5 - np.max(weights)

    def create_constraints(self):
        return (
            {'type': 'eq', 'fun': self.budget_constraint},  # Portfolio weights must sum to 1
            {'type': 'ineq', 'fun': self.scr_constraint}  # CVaR must be less than SCR_max
            #{'type': 'ineq', 'fun': self.max_weight_constraint}  # Max weight <= 0.5
        )
    
    def callback(self, xk):
        self.pbar.update(1)
    
    def optimize(self):
        # Bounds for weights: no short-selling (weights >= 0)
        bounds = [(0, 1) for _ in range(self.n_assets)]

        # Initial guess: equally weighted portfolio
        initial_weights = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            self.roRAC_objective, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=self.create_constraints(),
            callback=self.callback,
            options={'disp': True, 'maxiter': self.maxiter}
        )

        self.optimal_weights = result.x

        if not result.success:
            print("\nOptimization failed.")
            print(f"Status: {result.status}")
            print(f"Message: {result.message}")
        else:
            print("\nResult was a success!")

        print("\nOptimal Portfolio Weights:", self.optimal_weights.round(4), "sum", sum(self.optimal_weights))
        print("Optimal RORAC:", round(-self.roRAC_objective(self.optimal_weights) * 100, 5))
        print("Expected Portfolio Return:", round(np.dot(self.optimal_weights, self.expected_returns) * 100, 4), "%")
        print("Portfolio SCR (CVaR of BOF):", round(self.calculate_scr(self.optimal_weights), 2))
    
    def store_result(self):
        """
            Adds the results of the optimization to a dictionary in a JSON file.
            If the file does not exist, it creates a new file. Each result is keyed by a unique identifier.
        """
        results = {
            'optimal_weights': self.optimal_weights.tolist(), 
            'expected_return': np.dot(self.optimal_weights, self.expected_returns),  
            'SCR_risk': self.calculate_scr(self.optimal_weights),  
            'risk_measure': self.risk_measure,
        }

        file_path = '../optimization_results.json'
        unique_key = datetime.datetime.now().isoformat()

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        data = {}  
                except json.JSONDecodeError:
                    data = {}  
        else:
            data = {}

        data[unique_key] = results

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def plot_objective_landscape(self):
        """
        Plots the objective function landscape in 3D for two assets by varying their weights.
        """

        if self.n_assets != 2:
            print("Currently only supports visualization for 2 assets.")
            return
        
        w1 = np.linspace(0, 1, 100)
        w2 = np.linspace(0, 1, 100)
        W1, W2 = np.meshgrid(w1, w2)
        
        Z = np.zeros_like(W1)
        for i in range(W1.shape[0]):
            for j in range(W2.shape[1]):
                weights = np.array([W1[i, j], W2[i, j]])
                Z[i, j] = self.roRAC_objective(weights)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(W1, W2, Z, cmap='viridis')
        ax.set_xlabel('Weight 1')
        ax.set_ylabel('Weight 2')
        ax.set_zlabel('RoRAC')
        plt.title('RoRAC Objective Landscape for Two Assets')
        plt.show()
    
    def plot_bof_distribution(self, bof_values):
        """
        Plots the distribution of BOF values and highlights the VaR (Value at Risk) at a given confidence level.
        
        Parameters:
        - bof_values: The array of Basic Own Funds (BOF) values calculated in the optimization process.
        """
        var = np.percentile(bof_values, 100 * (1 - self.alpha))
        
        plt.figure(figsize=(10, 6))
        plt.hist(bof_values, bins=50, density=True, alpha=0.6, color='g', label='BOF Distribution')
        
        plt.axvline(var, color='r', linestyle='dashed', linewidth=2, label=f'VaR (alpha={self.alpha})')
        
        plt.xlabel('BOF Values')
        plt.ylabel('Density')
        plt.title(f'Distribution of BOF and VaR at {self.alpha * 100}% Confidence Level')
        plt.legend()
        plt.show()



optimizer = SLSQP(risk_measure='CVaR',
                  liabilities_0=900,
                  assets_0=1000,
                  eonia_rate=0.01,
                  alpha=0.995,
                  SCR_max=300, # Has to be defined with regards to the inital assets - liabilities
                  distribution_method='historical',
                  maxiter=100,
                  n_assets=None)


optimizer.optimize()

optimizer.store_result()

optimizer.plot_objective_landscape()


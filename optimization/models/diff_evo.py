import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

class DiffEvolution:

    def __init__(self, assets0, liabilities0, returns_df, alpha, risk_aversion = 1):
        self.assets0 = assets0
        self.liabilities0 = liabilities0
        self.returns_df = returns_df
        self.alpha = alpha
        self.liability_growth = 0.03 # Could be the Eonia rate for now
        self.mean_returns = returns_df.mean()
        self.cov_matrix = returns_df.cov()  
        self.risk_aversion = risk_aversion     

    def calculate_portfolio_return(self, weights):
        portfolio_returns = np.dot(self.returns_df, weights)
        return portfolio_returns

    def calculate_expected_return(self, simulated_cumulative_returns, weights):
        portfolio_values_t1 = self.assets0 * (1 + simulated_cumulative_returns)
        expected_return = (portfolio_values_t1.mean() - self.assets0) #/ self.assets0

        return expected_return

    def calculate_var(self, portfolio_returns):
        sorted_returns = np.sort(portfolio_returns)
        var_index = int((1 - self.alpha) * len(sorted_returns))
        var_value = sorted_returns[var_index]
        return var_value, sorted_returns  # Return both VaR and sorted returns
    
    def calculate_scr(self, weights):
        BOF_0 = self.assets0 - self.liabilities0  
        n_simulations = 500
        n_days = 252  # 1 year is typically 252 trading days

        simulated_daily_returns = np.random.multivariate_normal(self.mean_returns, self.cov_matrix, (n_simulations, n_days))

        portfolio_returns = np.dot(simulated_daily_returns, weights) 
        simulated_cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)[:, -1] - 1

        assets_t1 = self.assets0 * (1 + simulated_cumulative_returns)  # Assets after 1 year
        liabilities_t1 = self.liabilities0 * (1 + self.liability_growth)  # Liabilities after 1 year

        # Calculate Basic Own Funds (BOF) after 1 year
        bof_t1 = assets_t1 - liabilities_t1
        
        #bof_percentage = (bof_t1) / assets_t1 
        #scr_percentage = np.percentile(bof_percentage, 100 * (1 - self.alpha))

        bof_change = bof_t1 - BOF_0
        scr = np.percentile(bof_change, 100 * (1 - self.alpha))

        return scr, simulated_cumulative_returns  # Also return cumulative returns to use them for expected return calculation

    def penalty_sum_weights(self, weights):
        return 1e8 * abs(np.sum(weights) - 1)  # Penalize if weights deviate from 1

    def callback(self, xk, convergence):
        best_solution = xk
        best_convergence = convergence
        print(f"Current best solution: {np.round(best_solution, 4)}")
        print(f"Current convergence: {best_convergence}")
        return False  

    def objective(self, weights):
        scr, simulated_cumulative_returns = self.calculate_scr(weights)
        expected_return = self.calculate_expected_return(simulated_cumulative_returns, weights)  # Calculate expected return from 1-year cumulative returns
        penalty = self.penalty_sum_weights(weights)
        return -expected_return + self.risk_aversion * abs(scr) + penalty  # Combine objectives and add penalty

    def mean_var_optimization_global(self):
        N = self.returns_df.shape[1]

        bounds = [(0, 1) for _ in range(N)]

        result = differential_evolution(
            self.objective, 
            bounds=bounds,
            strategy='best1bin',
            maxiter=1000,
            popsize=4,
            tol=0.05,
            mutation=(0.2, 1),
            recombination=0.5,
            callback=self.callback,
            workers=-1,
            #updating="immediate"
        )

        return result

def fetch_data_df():
    df = pd.read_csv('../../data/asset_classes.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.iloc[:, :5]
    return df


if __name__ == "__main__":

    prices_df = fetch_data_df()
    returns_df = prices_df.pct_change(fill_method=None).dropna()

    start_time = time.time()

    optimizer = DiffEvolution(
        assets0=1000000,
        liabilities0=600000,
        returns_df=returns_df,
        alpha=0.995,
        risk_aversion=100
    )

    result = optimizer.mean_var_optimization_global()

    end_time = time.time()  
    print(f"Optimization took {end_time - start_time:.2f} seconds.")  

    if result.success:
        rounded_weights = np.round(result.x, 4)  
        rounded_objective_value = round(result.fun, 4) 
        
        asset_names = prices_df.columns.tolist()
        
        portfolio_allocation = {asset: weight for asset, weight in zip(asset_names, rounded_weights)}

        print("\nOptimal Portfolio Weights by Asset:")
        for asset, weight in portfolio_allocation.items():
            print(f"{asset}: {weight}")
        print("Sum of weights:", np.sum(result.x))

        print("\nCombined Objective Value (Expected Return + VaR):", rounded_objective_value)

        scr, simulated_cumulative_returns = optimizer.calculate_scr(result.x)
        expected_return = optimizer.calculate_expected_return(simulated_cumulative_returns, result.x)
        print("\nExpected return over 1 year:", round(expected_return * 100, 2))

        print("\nSolvency Capital Requirement (SCR):", abs(round(scr)))
        
    else:
        print("Optimization failed:", result.message)

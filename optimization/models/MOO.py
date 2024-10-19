import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

def calculate_portfolio_return(weights, returns_df):
    portfolio_returns = np.dot(returns_df, weights)
    return portfolio_returns

def calculate_expected_return(portfolio_returns):
    return np.mean(portfolio_returns)

def calculate_var(portfolio_returns, alpha=0.95):
    sorted_returns = np.sort(portfolio_returns)
    var_index = int((1 - alpha) * len(sorted_returns))
    var_value = sorted_returns[var_index]
    return var_value, sorted_returns  # Return both VaR and sorted returns

def penalty_sum_weights(weights):
    return 1e3 * abs(np.sum(weights) - 1)  # Penalize if weights deviate from 1

def callback(xk, convergence):
    best_solution = xk
    best_convergence = convergence
    print(f"Current best solution: {np.round(best_solution, 4)}")
    print(f"Current convergence: {best_convergence}")
    return False  

def objective(weights, returns_df, alpha=0.95):
    portfolio_returns = calculate_portfolio_return(weights, returns_df)
    expected_return = calculate_expected_return(portfolio_returns)
    var, _ = calculate_var(portfolio_returns, alpha)
    penalty = penalty_sum_weights(weights)
    return -expected_return + abs(var) + penalty  # Combine objectives and add penalty

def mean_var_optimization_global(returns_df, alpha=0.95):
    N = returns_df.shape[1]

    bounds = [(0, 1) for _ in range(N)]

    result = differential_evolution(
        objective, 
        bounds=bounds,
        args=(returns_df, alpha),  # Pass args for the objective function
        strategy='best1bin',
        maxiter=3000,
        popsize=150,
        tol=0.05,
        mutation=(0.2, 1),
        recombination=0.5,
        callback=callback,
        workers=-1
        #updating="immediate"
    )

    return result

def fetch_data_df():
    df = pd.read_csv('../../data/asset_classes.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.iloc[:, :12]
    return df

def plot_var_distribution(portfolio_returns, var_value, alpha):
    sorted_returns = np.sort(portfolio_returns)
    
    plt.figure(figsize=(10, 6))
    plt.hist(sorted_returns, bins=50, color='blue', alpha=0.7, label='Portfolio Returns')
    
    plt.axvline(x=var_value, color='red', linestyle='--', label=f'VaR ({int(alpha*100)}%) = {round(var_value, 4)}')
    
    plt.title(f'Portfolio Returns Distribution with VaR ({int(alpha*100)}%)')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    prices_df = fetch_data_df()
    returns_df = prices_df.pct_change(fill_method=None).dropna()

    alpha = 0.995

    result = mean_var_optimization_global(returns_df, alpha=alpha)

    if result.success:
        rounded_weights = np.round(result.x, 4)  
        rounded_objective_value = round(result.fun, 4) 
        
        asset_names = prices_df.columns.tolist()
        
        portfolio_allocation = {asset: weight for asset, weight in zip(asset_names, rounded_weights)}

        print("Optimal Portfolio Weights by Asset:")
        for asset, weight in portfolio_allocation.items():
            print(f"{asset}: {weight}")

        print("\nCombined Objective Value (Expected Return + VaR):", rounded_objective_value)

        portfolio_returns_over_time = calculate_portfolio_return(result.x, returns_df)
        expected_return = calculate_expected_return(portfolio_returns_over_time)
        print("\nExpected return:", round(expected_return, 8), "%")
        
        # Calculate portfolio returns using the optimal weights
        portfolio_returns = calculate_portfolio_return(result.x, returns_df)
        
        var_value, sorted_returns = calculate_var(portfolio_returns, alpha)
        
        plot_var_distribution(portfolio_returns, var_value, alpha)
        
    else:
        print("Optimization failed:", result.message)

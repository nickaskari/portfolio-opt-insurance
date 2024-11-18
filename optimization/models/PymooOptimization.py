import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from dotenv.main import load_dotenv
import os
load_dotenv(override=True)


class PortfolioOptimizationProblem(ElementwiseProblem):

    def __init__(self, assets0, liabilities0, returns_df, alpha, simulated_daily_returns, simulated_cumulative_returns, liability_growth=0.03, **kwargs):
        self.assets0 = assets0
        self.liabilities0 = liabilities0
        self.returns_df = returns_df
        self.alpha = alpha
        self.liability_growth = liability_growth
        self.mean_returns = returns_df.mean()
        self.cov_matrix = returns_df.cov()
        self.concentration_bounds = self.load_concentration_bounds()

        # Simulation setup
        self.n_simulations = int(os.getenv("N_SIMULATIONS"))
        self.n_days = int(os.getenv("N_DAYS"))

        self.simulated_daily_returns, self.n_simulatated_cumulative_returns = simulated_daily_returns, simulated_cumulative_returns

        self.risk_measure = os.getenv("RISK_MEASURE")
        self.duration_lower_bound, self.duration_upper_bound = float(os.getenv("DUR_LOWERBOUND")), float(os.getenv("DUR_UPPERBOUND"))

        yearly_returns = self.returns_df.resample('Y').apply(lambda x: np.prod(1 + x) - 1)
        self.mean_yearly_returns = yearly_returns.mean()

        num_concentration_constraints = len(returns_df.columns) * 2
        num_ieq_constraints = 3 + num_concentration_constraints

        # Set up the multi-objective problem
        super().__init__(n_var=self.returns_df.shape[1], 
                         n_obj=2,
                         n_eq_constr=1, 
                         n_ieq_constr=num_ieq_constraints,
                         xl=np.zeros(self.returns_df.shape[1]), 
                         xu=np.ones(self.returns_df.shape[1]))

    def _evaluate(self, weights, out, *args, **kwargs):
        scr = abs(self.calculate_scr(weights))
        expected_return = self.calculate_expected_return(weights)
        BOF_0 = self.assets0 - self.liabilities0
        weighted_duration = self.calculate_weighted_duration(weights)

        budget_constraint = np.sum(weights) - 1
        compliance = -BOF_0 + scr
        duration_left = self.duration_lower_bound - weighted_duration
        duration_right = weighted_duration - self.duration_upper_bound
        concentration = self.concentration_constraints(weights)
        
        
        out["F"] = [scr, -expected_return]  # Minimize SCR, maximize return (NSGA tries to minimize everything..)
        out["H"] = [budget_constraint]
        out["G"] = [compliance, duration_left, duration_right] + concentration # <= - constraint
        
        # Vectorize for speed

    def calculate_expected_return(self, weights):
        expected_return_percentage = np.dot(weights, self.mean_yearly_returns)
        expected_return_monetary = self.assets0 * expected_return_percentage
        return expected_return_monetary

    def calculate_scr(self, weights):
        BOF_0 = self.assets0 - self.liabilities0

        portfolio_returns = np.dot(self.simulated_daily_returns, weights)
        simulated_cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)[:, -1] - 1

        assets_t1 = self.assets0 * (1 + simulated_cumulative_returns)
        liabilities_t1 = self.liabilities0 * (1 + self.liability_growth)

        bof_t1 = assets_t1 - liabilities_t1
        bof_change = bof_t1 - BOF_0
        var = np.percentile(bof_change, 100 * (1 - self.alpha))

        if self.risk_measure == 'var':  
            scr = var
        elif self.risk_measure == 'cvar':
            cvar = np.mean(bof_change[bof_change <= var])
            scr = cvar
        else:
            raise SystemExit("Invalid risk measure:", self.risk_measure)

        return scr
    
    def calculate_weighted_duration(self, weights):
        weighted_duration = 0

        asset_columns = self.returns_df.columns

        for i, asset in enumerate(asset_columns):
            # Construct the environment variable key, e.g., DUR_GOV for asset "GOV"
            env_key = f"DUR_{asset.upper()}"
            bond_duration = float(os.getenv(env_key, 0))  # Default to 0 if not found
            weighted_duration += bond_duration * weights[i]

        return weighted_duration

    def load_concentration_bounds(self):
        bounds = {}

        asset_columns = self.returns_df.columns

        for asset in asset_columns:
            bound_str = os.getenv(f"CONCENTRATION_BOUNDS_{asset.upper()}")
            if bound_str:  # Only add if bounds are defined in the .env file
                lower_bound, upper_bound = map(float, bound_str.split(','))
                bounds[asset] = (lower_bound, upper_bound)
        
        return bounds
    
    def concentration_constraints(self, weights):
        concentration_constraints = []
        asset_columns = self.returns_df.columns

        for i, weight in enumerate(weights):
                asset_name = asset_columns[i]
                if asset_name in self.concentration_bounds:
                    lower_bound, upper_bound = self.concentration_bounds[asset_name]
                    
                    concentration_constraints.append(lower_bound - weight)  # weight >= lower_bound
                    concentration_constraints.append(weight - upper_bound)  # weight <= upper_bound
        
        return concentration_constraints
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

class PortfolioOptimizationProblem(ElementwiseProblem):

    def __init__(self, assets0, liabilities0, returns_df, alpha, liability_growth=0.03, distribution="normal"):
        ''' 
            distribution: 'normal' (deafult) or 'tstudent'
        '''
        self.assets0 = assets0
        self.liabilities0 = liabilities0
        self.returns_df = returns_df
        self.alpha = alpha
        self.liability_growth = liability_growth

        # Mean returns and covariance matrix
        self.mean_returns = returns_df.mean()
        self.cov_matrix = returns_df.cov()

        # Simulation setup
        self.n_simulations = 100000
        self.n_days = 252

        if distribution == 'normal':
            self.simulated_daily_returns, self.simulated_cumulative_returns = self.normal_distribution()
        elif distribution == 't-student':
            #FIX
            self.simulated_daily_returns, self.simulated_cumulative_returns = self.normal_distribution()

        yearly_returns = self.returns_df.resample('Y').apply(lambda x: np.prod(1 + x) - 1)
        self.mean_yearly_returns = yearly_returns.mean()

        # Set up the multi-objective problem
        super().__init__(n_var=self.returns_df.shape[1], 
                         n_obj=2,
                         n_eq_constr=1, 
                         xl=np.zeros(self.returns_df.shape[1]), 
                         xu=np.ones(self.returns_df.shape[1]))

    def _evaluate(self, weights, out, *args, **kwargs):
        scr = abs(self.calculate_scr(weights))
        expected_return = self.calculate_expected_return(weights)

        constraint = np.sum(weights) - 1
        
        out["F"] = [scr, -expected_return]  # Minimize SCR, maximize return (NSGA tries to minimize everything..)
        out["H"] = [constraint]

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
        scr = np.percentile(bof_change, 100 * (1 - self.alpha))
        return scr
    
    
    ## MAKE FUNCTION THAT SIMMUALTES NORMALSIMULATION
    def normal_distribution(self):
        simulated_daily_returns = np.random.multivariate_normal(
            self.mean_returns, self.cov_matrix, (self.n_simulations, self.n_days))
        simulated_cumulative_returns = np.cumprod(1 + self.simulated_daily_returns, axis=1)[:, -1] - 1

        return simulated_daily_returns, simulated_cumulative_returns


    ## MAKE FUNCTION THAT SIMMULATES T-STUDENT



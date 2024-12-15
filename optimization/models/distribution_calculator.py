from dotenv.main import load_dotenv
import os
import numpy as np
from scipy.stats import t
load_dotenv(override=True)


class DistributionCalculator:

    def __init__(self, returns_df):
        self.returns_df = returns_df
        self.n_simulations, self.n_days = int(os.getenv("N_SIMULATIONS")), int(os.getenv("N_DAYS"))

    # for daily returns
    def normal(self):
        mean_returns, cov_matrix = self.returns_df.mean(), self.returns_df.cov()

        simulated_daily_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, (self.n_simulations, self.n_days))
        simulated_cumulative_returns = np.cumprod(1 + simulated_daily_returns, axis=1)[:, -1] - 1

        return simulated_daily_returns, simulated_cumulative_returns

    # for daily returns
    def t_student(self):
        mean_returns, cov_matrix = self.returns_df.mean(), self.returns_df.cov()

        t_params = [t.fit(self.returns_df[col].dropna()) for col in self.returns_df.columns]
        df_degrees = [params[0] for params in t_params]  # Extract degrees of freedom for each asset

        # Initialize an array to store simulated daily returns for all assets
        simulated_daily_returns = np.zeros((self.n_simulations, self.n_days, len(mean_returns)))

        # Generate daily returns for each asset using the t-distribution
        for i, (mean, std, df) in enumerate(zip(mean_returns, np.sqrt(np.diag(cov_matrix)), df_degrees)):
            simulated_daily_returns[:, :, i] = t.rvs(df, loc=mean, scale=std, size=(self.n_simulations, self.n_days))

        # Calculate cumulative returns over 1 year for each simulation and asset
        simulated_cumulative_returns = np.cumprod(1 + simulated_daily_returns, axis=1)[:, -1, :] - 1

        return simulated_daily_returns, simulated_cumulative_returns
    
    
    def prefered_dist(self, custom=None):

        if not custom:
            preferance = os.getenv("DISTRIBUTION")
        else:
            preferance = custom

        if preferance == 'tstudent':
            return self.t_student()
        elif preferance == 'normal':
            return self.normal()
        
    # Returns only cumulative
    def prefered_liability_growth(self, custom = None):

        if not custom:
            preferance = os.getenv('DISTRIBUTION')
        else:
            preferance = custom

        if preferance == 'tstudent':
            t_params = t.fit(self.returns_df['EONIA'].dropna())
            df_degree = t_params[0]  # Degrees of freedom
            mean_eonia = t_params[1]  # Mean
            std_eonia = t_params[2]  # Scale (standard deviation)


            # Simulate daily returns for EONIA using the t-distribution
            simulated_lg = t.rvs(
                df_degree, loc=mean_eonia, scale=std_eonia, size=(self.n_simulations, self.n_days)
            )

            # Calculate cumulative returns over 1 year for each simulation
            simulated_cumulative_lg = (
                np.cumprod(1 + simulated_lg, axis=1)[:, -1] - 1
            )
            return simulated_cumulative_lg
        elif preferance == 'normal':
            mean_eonia_returns = self.returns_df['EONIA'].mean()
            variance_eonia_matrix = self.returns_df['EONIA'].var()

            simulated_lg = np.random.normal(mean_eonia_returns, np.sqrt(variance_eonia_matrix), (self.n_simulations, self.n_days))

            # Calculate cumulative returns for each asset class over 1 year
            simulated_cumulative_lg = np.cumprod(1 + simulated_lg, axis=1)[:, -1] - 1

            return simulated_cumulative_lg


        

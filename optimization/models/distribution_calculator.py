from dotenv.main import load_dotenv
import os
import numpy as np
from scipy.stats import t
load_dotenv(override=True)


class DistributionCalculator:

    def __init__(self, returns_df):
        self.returns_df = returns_df

    def normal(self):
        mean_returns, cov_matrix = self.returns_df.mean(), self.returns_df.cov()
        n_simulations, n_days = int(os.getenv("N_SIMULATIONS")), int(os.getenv("N_DAYS"))

        simulated_daily_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, (n_simulations, n_days))
        simulated_cumulative_returns = np.cumprod(1 + simulated_daily_returns, axis=1)[:, -1] - 1

        return simulated_daily_returns, simulated_cumulative_returns


    def t_student(self):
        mean_returns, cov_matrix = self.returns_df.mean(), self.returns_df.cov()
        n_simulations, n_days = int(os.getenv("N_SIMULATIONS")), int(os.getenv("N_DAYS"))

        t_params = [t.fit(self.returns_df[col].dropna()) for col in self.returns_df.columns]
        df_degrees = [params[0] for params in t_params]  # Extract degrees of freedom for each asset

        # Initialize an array to store simulated daily returns for all assets
        simulated_daily_returns = np.zeros((n_simulations, n_days, len(mean_returns)))

        # Generate daily returns for each asset using the t-distribution
        for i, (mean, std, df) in enumerate(zip(mean_returns, np.sqrt(np.diag(cov_matrix)), df_degrees)):
            simulated_daily_returns[:, :, i] = t.rvs(df, loc=mean, scale=std, size=(n_simulations, n_days))

        # Calculate cumulative returns over 1 year for each simulation and asset
        simulated_cumulative_returns = np.cumprod(1 + simulated_daily_returns, axis=1)[:, -1, :] - 1

        return simulated_daily_returns, simulated_cumulative_returns
    
    def prefered_dist(self):
        preferance = os.getenv("DISTRIBUTION")

        if preferance == 'tstudent':
            return self.t_student()
        elif preferance == 'normal':
            return self.normal()

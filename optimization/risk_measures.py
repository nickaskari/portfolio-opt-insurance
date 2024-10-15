import cvxpy as cp
import numpy as np

def calculate_risk(allocations, returns, alpha, risk_measure="CVaR"):
    
    """ 
    Calculates risk based on variables, return, alpha (confidence interval).
    Default risk measure is set to CVaR.
    Risk is calculated based on future returns.
    """

    if risk_measure == "CVaR":
        VaR = cp.Variable()  
        CVaR = cp.Variable()  
        losses = -returns @ allocations
        VaR_constraint = losses >= VaR
        CVaR_expression = VaR + (1 / (1 - alpha)) * cp.sum(cp.pos(losses - VaR)) / len(returns)
        return CVaR_expression, [VaR_constraint]

    elif risk_measure == "VaR":
        losses = -returns @ allocations
        sorted_losses = cp.sum(cp.pos(cp.sort(losses)))
        VaR = sorted_losses[int(len(losses) * (1 - alpha))]  # The alpha quantile of losses
        return VaR, []

    elif risk_measure == "variance":
        cov_matrix = np.cov(returns, rowvar=False)  # Covariance matrix based on returns
        portfolio_risk = cp.quad_form(allocations, cov_matrix)  # Variance (quadratic form)
        return portfolio_risk, []

    elif risk_measure == "standard_deviation":
        portfolio_variance, _ = calculate_risk(allocations, returns, alpha, "variance")
        portfolio_stddev = cp.sqrt(portfolio_variance)  
        return portfolio_stddev, []

    elif risk_measure == "max_drawdown":
        cumulative_returns = cp.cumsum(returns @ allocations) 
        peak = cp.Variable()  # Highest value at each point in time
        drawdown = cp.Variable()  # Drawdown at each point
        constraints = [peak >= cumulative_returns,  # Ensure peak tracks the highest value
                       drawdown >= peak - cumulative_returns,  # Drawdown is the difference between peak and cumulative return
                       drawdown >= 0]  # Drawdown must be non-negative
        max_drawdown = cp.max(drawdown)  # Max drawdown is the highest observed drawdown
        return max_drawdown, constraints

    elif risk_measure == "Sharpe_ratio":
        risk_free_rate = 0.02  
        portfolio_return = returns.mean(axis=0) @ allocations
        portfolio_stddev, _ = calculate_risk(allocations, returns, alpha, "standard_deviation")
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev  
        return -sharpe_ratio, []  

    else:
        raise ValueError("Unsupported risk measure")

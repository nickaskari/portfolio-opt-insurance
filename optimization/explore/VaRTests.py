from scipy.stats import chi2
import numpy as np

def christoffersen_test(violations, confidence_level=0.95):
    """
    Perform Christoffersen's Unconditional Coverage and Independence tests.
    
    Parameters:
        violations (np.array): A binary array where 1 indicates a VaR violation, 0 otherwise.
        confidence_level (float): The confidence level for VaR (e.g., 0.95 for a 95% VaR).
    
    Returns:
        dict: Test statistics and p-values for Unconditional Coverage, Independence, and Joint Test.
    """
    n = len(violations)
    n_violations = np.sum(violations)
    expected_violations = n * (1 - confidence_level)
    
    # Unconditional Coverage Test
    pi_hat = n_violations / n
    uc_stat = -2 * (n * np.log(1 - confidence_level) + n_violations * np.log(confidence_level / pi_hat))
    uc_p_value = 1 - chi2.cdf(uc_stat, df=1)
    
    # Independence Test
    n_00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
    n_01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
    n_10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
    n_11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))

    p_1 = n_10 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0
    p_2 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0

    ind_stat = -2 * (n_10 * np.log(p_1) + n_11 * np.log(1 - p_1) +
                     n_01 * np.log(p_2) + n_00 * np.log(1 - p_2))
    ind_p_value = 1 - chi2.cdf(ind_stat, df=1)
    
    # Joint Test (Unconditional Coverage + Independence)
    joint_stat = uc_stat + ind_stat
    joint_p_value = 1 - chi2.cdf(joint_stat, df=2)
    
    return {
        'Unconditional Coverage Statistic': uc_stat,
        'Unconditional Coverage p-value': uc_p_value,
        'Independence Statistic': ind_stat,
        'Independence p-value': ind_p_value,
        'Joint Test Statistic': joint_stat,
        'Joint Test p-value': joint_p_value
    }
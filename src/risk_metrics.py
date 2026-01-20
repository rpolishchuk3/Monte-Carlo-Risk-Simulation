import numpy as np

def calculate_metrics(initial_value, final_values):
    """
    Calculates risk metrics (VaR, CVaR) based on simulation results.
    """
    
    pnl = final_values - initial_value # calculate profit and loss (PnL) for each simulation
    
    # calculate Value at Risk (VaR) at 95% and 99% confidence levels
    # represents the cutoff for the worst 5% and 1% of scenarios
    var_95 = np.percentile(pnl, 5)
    var_99 = np.percentile(pnl, 1)
    
    # calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall
    # average loss for cases worse than the VaR threshold
    cvar_95 = pnl[pnl <= var_95].mean()
    cvar_99 = pnl[pnl <= var_99].mean()
    
    # stores results in a dictionary
    metrics = {
        "Mean Ending Value": np.mean(final_values),
        "Mean PnL": np.mean(pnl),
        "VaR 95%": var_95,
        "VaR 99%": var_99,
        "CVaR 95%": cvar_95,
        "CVaR 99%": cvar_99,
        "Std Dev PnL": np.std(pnl)
    }
    
    return metrics
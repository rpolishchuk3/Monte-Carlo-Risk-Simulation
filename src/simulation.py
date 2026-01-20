import numpy as np

def simulate_portfolio_paths(S0, mu, cov_matrix, weights, T, N_sims, dt=1/252):
    """
    Runs a Monte Carlo simulation to project future portfolio values.
    """
    
    n_assets = len(mu)

    L = np.linalg.cholesky(cov_matrix) # use Cholesky decomposition to handle correlations between assets

    portfolio_paths = np.zeros((T + 1, N_sims)) # format: (Days + 1, Number of Simulations)
    portfolio_paths[0] = S0                     # create a container for the simulation results 
    
    Z = np.random.normal(0, 1, (T, n_assets, N_sims)) # generate random shocks for the simulation in format: (Days, Assets, Simulations)
    Z_corr = np.einsum('ij, tjs -> tis', L, Z)        # adjust shocks using the Cholesky matrix to account for correlation
    
    drift = (mu - 0.5 * np.diag(cov_matrix)) * dt     # calculate drift (expected trend) for the assets
    drift_expanded = drift[np.newaxis, :, np.newaxis] # reshape drift to match the dimensions of our simulation array
    
    diffusion = Z_corr * np.sqrt(dt)                  # calculate diffusion (random movement)
    daily_log_returns = drift_expanded + diffusion    # combine drift and diffusion to get daily log returns
    daily_simple_returns = np.exp(daily_log_returns) - 1 # Convert log returns to simple returns
    
    current_val = np.full(N_sims, S0) # Track the current portfolio value across all simulations
    
    for t in range(T): # loop through each day in the simulation
        step_returns = np.dot(weights, daily_simple_returns[t]) # calculate the portfolio return for the day using weights
        
        current_val = current_val * (1 + step_returns)  # update portfolio value
        portfolio_paths[t+1] = current_val
        
    return portfolio_paths
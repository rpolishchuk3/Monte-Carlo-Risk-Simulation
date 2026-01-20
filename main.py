import numpy as np
import src.data_loader as dl
import src.simulation as sim
import src.risk_metrics as rm
import src.visualization as vis

def main():
    DATA_PATH = 'data/prices.csv'
    INITIAL_PORTFOLIO = 100_000     # Starting capital ($100k)
    TIME_HORIZON = 252              # 1 trading year
    N_SIMS = 5000                   # Number of simulations to run
    
    print("=== Starting Monte Carlo Risk Simulation ===")
    

    print(f"Loading data from {DATA_PATH}...")
    prices = dl.load_price_data(DATA_PATH)
    mu, cov_matrix = dl.get_return_stats(prices)
    
    
    num_assets = len(mu)
    weights = np.ones(num_assets) / num_assets # set portfolio weights (equal weight for each asset)
    
    print(f"Assets loaded: {list(prices.columns)}")
    print(f"Annualized Volatilities: {np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)}")
    
    # monte carlo simulation
    print(f"\nHorizon: {TIME_HORIZON} trading days")
    print(f"Simulating {N_SIMS} paths...")
    paths = sim.simulate_portfolio_paths(
        S0=INITIAL_PORTFOLIO,
        mu=mu,
        cov_matrix=cov_matrix,
        weights=weights,
        T=TIME_HORIZON,
        N_sims=N_SIMS
    )
    
    
    final_values = paths[-1]
    metrics = rm.calculate_metrics(INITIAL_PORTFOLIO, final_values)     # calculate risk metrics
    

    print("\n=== Risk Analysis Summary ===")
    print(f"Initial Investment:  ${INITIAL_PORTFOLIO:,.2f}")
    print(f"Exp. Ending Value:   ${metrics['Mean Ending Value']:,.2f}")
    print(f"Value at Risk (95%): ${metrics['VaR 95%']:,.2f}")
    print(f"Value at Risk (99%): ${metrics['VaR 99%']:,.2f}")
    print(f"CVaR (95%):          ${metrics['CVaR 95%']:,.2f}")
    print(f"Worst Simulated PnL: ${(np.min(final_values) - INITIAL_PORTFOLIO):,.2f}")
    

    print("\nGenerating plots...")
    vis.plot_simulation(paths, final_values, INITIAL_PORTFOLIO, metrics)

if __name__ == "__main__":
    main()
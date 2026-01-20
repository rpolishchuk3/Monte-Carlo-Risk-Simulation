import matplotlib.pyplot as plt
import numpy as np

def plot_simulation(paths, final_values, initial_value, metrics):
    """
    Plots the simulation paths and the distribution of final profits/losses.
    """

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # plot 1: Show the first 100 simulation paths
    # shows how the portfolio value changes over time
    axes[0].plot(paths[:, :100], alpha=0.4, linewidth=1)
    axes[0].set_title("Projected Portfolio Paths (First 100 Simulations)")
    axes[0].set_xlabel("Trading Days")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # plot 2: histogram of final Profit/Loss (PnL)
    pnl = final_values - initial_value
    axes[1].hist(pnl, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1].set_title("Distribution of Final PnL")
    axes[1].set_xlabel("Profit / Loss ($)")
    axes[1].set_ylabel("Frequency")
    
    # add vertical lines to show VaR and Mean PnL
    var_95 = metrics['VaR 95%']
    axes[1].axvline(var_95, color='r', linestyle='--', label=f'VaR 95%: ${var_95:,.0f}')
    axes[1].axvline(metrics['Mean PnL'], color='g', linestyle='-', label='Mean PnL')
    
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
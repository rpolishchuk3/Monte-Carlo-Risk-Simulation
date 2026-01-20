import pandas as pd
import numpy as np

def load_price_data(filepath):
    """
    Reads stock price data from a CSV file.
    """
    
    try:
        df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')  # load the file and set the Date column as the index
        df.sort_index(inplace=True) # sort by date to ensure the order is correct
        
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file at {filepath}")


def get_return_stats(price_df):
    """
    Calculates the mean returns and covariance matrix from price data.
    """
    
    log_returns = np.log(price_df / price_df.shift(1)) # calculate daily log returns
    log_returns.dropna(inplace=True) # remove the first row since it contains NaN values after shifting
    
    mean_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values # calculate average daily returns and the covariance matrix
    
    return mean_returns, cov_matrix
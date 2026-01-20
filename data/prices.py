import pandas as pd
from pandas_datareader import data as pdr


symbols = { # stores indecies of stocks
    "AAPL": "AAPL.US",
    "NVDA": "NVDA.US",
    "INTC": "INTC.US"
}

dfs = []    # stores dataframes of each stock

for name, symbol in symbols.items():
    df = pdr.DataReader(symbol, "stooq") # call stooq to get data
    df = df.sort_index()                 # reverse stock date
    dfs.append(df["Close"].rename(name)) # appends "Close" type stock to the list

prices = pd.concat(dfs, axis=1)          # write columns in the order: Date, AAPL, NVDA, INTC
prices = prices.loc["2025-01-01":"2025-12-31"] # get from 1st January 2025 up to 31st December 2025

prices.to_csv("data/prices.csv")
print(prices.head())

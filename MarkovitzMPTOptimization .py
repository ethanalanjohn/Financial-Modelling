from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
warnings.filterwarnings('ignore')

sp500 = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50")[1]

sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

symbols_list = sp500['Symbol'].unique().tolist()
symbols_list = [symbol + ".NS" for symbol in sp500['Symbol']]

end_date = '2024-12-27'

start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)

df = yf.download(tickers=symbols_list,
                 start=start_date,
                 end=end_date).stack()

df.index.names = ['date', 'ticker']


df.columns = df.columns.str.lower()
df.columns.name = None

df

# Reset index to ensure 'date' is a column
df = df.reset_index()


# Filter for rows where the day of the month is 28
df_28th = df[df['date'].dt.day == 28]

print(df_28th.head())  # Check output
df_28th = df_28th.reset_index(drop=True)

# Reset index to ensure 'date' is a column
df_28th = df_28th.reset_index(drop=True)

# Set 'date' as the index
df_28th = df_28th.set_index('date')

print(df_28th.head())  # Now the index will be 'date'

# Reset index and ensure 'date' is datetime
df = df.reset_index()
df['date'] = pd.to_datetime(df['date'])

# Filter for rows where the day of the month is 28
df_28th = df[df['date'].dt.day == 28]

# Set MultiIndex again
df_28th = df_28th.set_index(['date', 'ticker'])

print(df_28th.index.names)  # Should print ['date', 'ticker']

def process_each_symbol(df, symbols_list, threshold):
    """
    Computes the fraction of months where the monthly return of 'Adj Close' price is above a given threshold.

    Parameters:
    - df: MultiIndex DataFrame with index ['date', 'ticker']
    - symbols_list: List of ticker symbols
    - threshold: The return threshold to check against

    Returns:
    - Dictionary with symbol as key and fraction of months where return > threshold
    """
    results = {}

    for symbol in symbols_list:
        if symbol in df.index.get_level_values('ticker'):
            # Extract only 'Adj Close' prices for the symbol
            symbol_data = df.xs(symbol, level='ticker')[['close']].dropna()

            # Compute **monthly** returns
            monthly_returns = symbol_data['close'].pct_change().dropna()

            # Compute fraction of months above threshold
            above_threshold = (monthly_returns > threshold).sum()
            total_months = len(monthly_returns)

            # Store result as fraction of months
            results[symbol] = threshold*(above_threshold / total_months) if total_months > 0 else None

    return results

# Example usage: Check how often 'Adj Close' returns exceed 10% (0.1)
threshold = 0.08
results = process_each_symbol(df_28th, symbols_list, threshold)

# Print sample results
print(results)

import numpy as np

def compute_sd(df, symbols_list, average_values):
    """
    Computes the standard deviation of daily return deviations from the precomputed average.

    Parameters:
    - df: MultiIndex DataFrame with index ['date', 'ticker']
    - symbols_list: List of ticker symbols
    - average_values: Dictionary with symbol as key and precomputed average annual return

    Returns:
    - Dictionary with symbol as key and standard deviation of deviations from expected return
    """
    sd_results = {}

    for symbol in symbols_list:
        if symbol in df_28th.index.get_level_values('ticker'):
            # Extract 'Adj Close' prices for the symbol
            symbol_data = df_28th.xs(symbol, level='ticker')[['close']].dropna()

            # Compute daily log returns
            daily_returns = np.log(symbol_data['close'] / symbol_data['close'].shift(1)).dropna()

            # Convert expected annual return to daily equivalent (Assume 252 trading days)
            expected_daily_return = average_values[symbol] / (252 * 8)  # Dividing by total trading days

            # Compute deviations from expected daily return
            deviations = daily_returns - expected_daily_return

            # Compute standard deviation of these deviations
            sd_results[symbol] = np.std(deviations)

    return sd_results

# Compute standard deviations using previous results as the "expected annual return"
sd_results = compute_sd(df, symbols_list, results)

# Print standard deviation results
print(sd_results)

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_weights(profit, risk, max_allocation=0.15, capital_constraint=1.0):
    """
    Optimizes portfolio weights to maximize the Sharpe Ratio.
    
    Parameters:
    - profit: Dictionary of expected returns for each ticker.
    - risk: Dictionary of standard deviations (volatility) for each ticker.
    - max_allocation: Maximum weight allowed per stock (default 15%).
    - capital_constraint: Total capital constraint (default 100% = 1.0).
    
    Returns:
    - Dictionary with tickers as keys and optimized weights as values.
    """
    symbols = list(profit.keys())  # List of tickers
    expected_returns = np.array([profit[sym] for sym in symbols])  # Convert profit to array
    risk_values = np.array([risk[sym] for sym in symbols])  # Convert risk to array

    # Define Sharpe Ratio function (to maximize)
    def neg_sharpe(weights):
        portfolio_return = np.dot(weights, expected_returns)  # Weighted sum of returns
        portfolio_risk = np.sqrt(np.dot(weights**2, risk_values**2))  # Weighted risk (approx)
        return -portfolio_return / portfolio_risk  # Negative because we minimize in scipy

    # Constraints: Sum of weights must be 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - capital_constraint})

    # Bounds: Each stock gets between 0 and max_allocation
    bounds = [(0, max_allocation) for _ in symbols]

    # Initial guess (equal weight allocation)
    initial_weights = np.array([1/len(symbols)] * len(symbols))

    # Solve optimization problem
    result = minimize(neg_sharpe, initial_weights, bounds=bounds, constraints=constraints, method='SLSQP')

    # Convert optimized weights to dictionary
    optimized_weights = dict(zip(symbols, result.x))

    return optimized_weights

# Example Usage
optimized_weights = optimize_weights(profit=results, risk=sd_results)

# Print final allocation
print(optimized_weights)

# Convert final_weights dictionary to DataFrame
sorted_weights_df = pd.DataFrame(optimized_weights.items(), columns=['Ticker', 'Weight'])

# Sort by weight in descending order
sorted_weights_df = sorted_weights_df.sort_values(by='Weight', ascending=False)

# Print the sorted weights
print(sorted_weights_df)

# Convert optimized_weights dictionary to a DataFrame
weights_df = pd.DataFrame(optimized_weights.items(), columns=['Ticker', 'Weight'])

# Sort by weight in descending order
weights_df = weights_df.sort_values(by='Weight', ascending=False).reset_index(drop=True)

# Print the DataFrame
print(weights_df)

import datetime as dt
import yfinance as yf
import pandas as pd
# Download NIFTY 50 data (^NSEI) with monthly interval
nifty = yf.download(tickers="^NSEI",
                    start="2015-01-01",
                    end=dt.date.today(),
                    interval="1d")  # Daily data to get exact 28th dates

# Ensure the index is in datetime format
nifty.index = pd.to_datetime(nifty.index)

# Filter only the rows where the date is the 28th
nifty_28th = nifty[nifty.index.day == 28]

# Compute percentage change from previous month
first_close = nifty_28th.iloc[0]['Close']
nifty_28th['NIFTY 50 % Increase'] = ((nifty_28th['Close'] - first_close) / first_close) * 100
# Keep only relevant columns
nifty_28th = nifty_28th[['Close', 'NIFTY 50 % Increase']].dropna()
nifty_28th.columns = nifty_28th.columns.to_flat_index()  # Flatten MultiIndex
nifty_28th.columns = [col[0] if isinstance(col, tuple) else col for col in nifty_28th.columns]  # Extract first level

# Print final result
print(nifty_28th)


import datetime as dt
import yfinance as yf
import pandas as pd

def simulate_nifty_performance():
    """
    Computes the cumulative percentage increase of NIFTY 50 from the first available 28th date.
    
    Returns:
    - DataFrame containing cumulative percentage increase over time.
    """

    # Download NIFTY 50 daily data
    nifty = yf.download(tickers="^NSEI",
                        start="2015-01-01",
                        end=dt.date.today(),
                        interval="1d")

    # Ensure index is in datetime format
    nifty.index = pd.to_datetime(nifty.index)

    # Filter only the 28th of each month
    nifty_28th = nifty[nifty.index.day == 28].copy()

    # Check if data exists
    if nifty_28th.empty:
        raise ValueError("Error: No NIFTY 50 data available on the 28th of each month.")

    # Reference value (first 28th closing price)
    first_close = nifty_28th.iloc[0]['Close']

    # Compute percentage change from the first available 28th
    nifty_28th['NIFTY 50 % Increase'] = ((nifty_28th['Close'] - first_close) / first_close) * 100

    # Keep only relevant columns
    nifty_df = nifty_28th[['NIFTY 50 % Increase']]

    return nifty_df

def merge_portfolio_nifty(portfolio_df, nifty_df):
    """
    Merges the portfolio performance DataFrame with the NIFTY 50 performance DataFrame.
    
    Parameters:
    - portfolio_df: DataFrame containing portfolio cumulative percentage increase.
    - nifty_df: DataFrame containing NIFTY 50 cumulative percentage increase.

    Returns:
    - Combined DataFrame with both portfolio and NIFTY performance.
    """

    # Ensure both have datetime index
    portfolio_df.index = pd.to_datetime(portfolio_df.index)
    nifty_df.index = pd.to_datetime(nifty_df.index)

    # Merge both DataFrames on date
    combined_df = pd.merge(portfolio_df, nifty_df, left_index=True, right_index=True, how="inner")

    return combined_df

# Example usage
# Ensure both DataFrames have datetime index
portfolio_df.index = pd.to_datetime(portfolio_df.index)
nifty_28th.index = pd.to_datetime(nifty_28th.index)

# Keep only 'NIFTY 50 % Increase' for merging
nifty_28th = nifty_28th[['NIFTY 50 % Increase']]

# Merge on index
combined_df = pd.merge(portfolio_df, nifty_28th, left_index=True, right_index=True, how="inner")

# Print final result
print(combined_df.head())

combined_df.plot()

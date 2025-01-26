import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import yfinance as yf
from datetime import datetime,timedelta
import pyomo.environ as pyo
from pyomo.environ import SolverFactory

from pprint import pprint
from sklearn.preprocessing import StandardScaler
from pyopt.client import PriceHistory

# Set some display options for Pandas.
pd.set_option('display.max_colwidth', -1)
pd.set_option('expand_frame_repr', False)
import numpy as np
import pandas as pd

# Input Data
price_data_frame = {
    "STOCK NAME": ["DCBBANK.NS", "RELIANCE.NS", "FACT.NS", "ZOMATO.NS", "KSOLVES.NS",
                   "HAPPSTMNDS.NS", "ITC.NS", "PCBL.NS", "HDFC.NS", "HINDUNILVR.NS"],
    "CURRENT STOCK PRICE": [109.26, 1239.85, 893.45, 243.90, 916.45, 718.50, 439.05, 351.85, 351.85, 2591.70],
    "PREDICTED STOCK PRICE": [118.64, 1325.27, 1054.248782, 264.346153, 1083.01, 759.39, 463.61, 373.07, 387.63, 2591.70],
    "RISK": [0.1331, 0.0815, 0.27, 0.133328, 0.1823, 0.1311, 0.0704, 0.1521, 0.1521, 0.0633],
    "INDUSTRY": [2, 3, 1, 1, 4, 4, 7, 6, 2, 7]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(price_data_frame)

# Calculate Log Returns (using predicted vs current stock prices)
df["LOG RETURN"] = np.log(df["PREDICTED STOCK PRICE"] / df["CURRENT STOCK PRICE"])

# Generate Random Weights
number_of_symbols = len(df)
random_weights = np.random.random(number_of_symbols)

# Normalize weights (Rebalance Weights)
rebalance_weights = random_weights / np.sum(random_weights)
df["WEIGHTS"] = rebalance_weights

# Expected Returns (weighted average of log returns, annualized for 252 trading days)
exp_ret = np.sum(df["LOG RETURN"] * rebalance_weights) * 252

# Expected Volatility (using risk values provided)
cov_matrix = np.diag(df["RISK"])  # Assuming diagonal covariance matrix (independent risks)
exp_vol = np.sqrt(
    np.dot(rebalance_weights.T, np.dot(cov_matrix * 252, rebalance_weights))
)

# Sharpe Ratio
risk_free_rate = 0.02  # Assume a 2% risk-free rate
sharpe_ratio = (exp_ret - risk_free_rate) / exp_vol

# Display Weights and Metrics
print('='*80)
print("PORTFOLIO DATA:")
print(df)
print('-'*80)

metrics_df = pd.DataFrame({
    "Expected Portfolio Returns": [exp_ret],
    "Expected Portfolio Volatility": [exp_vol],
    "Portfolio Sharpe Ratio": [sharpe_ratio]
})

print('='*80)
print("PORTFOLIO METRICS:")
print(metrics_df)
print('-'*80)

import numpy as np
import pandas as pd

# Input Data
price_data_frame = {
    "STOCK NAME": ["DCBBANK.NS", "RELIANCE.NS", "FACT.NS", "ZOMATO.NS", "KSOLVES.NS",
                   "HAPPSTMNDS.NS", "ITC.NS", "PCBL.NS", "HDFC.NS", "HINDUNILVR.NS"],
    "CURRENT STOCK PRICE": [109.26, 1239.85, 893.45, 243.90, 916.45, 718.50, 439.05, 351.85, 351.85, 2591.70],
    "PREDICTED STOCK PRICE": [118.64, 1325.27, 1054.248782, 264.346153, 1083.01, 759.39, 463.61, 373.07, 387.63, 2591.70],
    "RISK": [0.1331, 0.0815, 0.27, 0.133328, 0.1823, 0.1311, 0.0704, 0.1521, 0.1521, 0.0633],
    "INDUSTRY": [2, 3, 1, 1, 4, 4, 7, 6, 2, 7]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(price_data_frame)

# Calculate Log Returns
df["LOG RETURN"] = np.log(df["PREDICTED STOCK PRICE"] / df["CURRENT STOCK PRICE"])

# Covariance matrix: Assume diagonal matrix using "RISK" column
cov_matrix = np.diag(df["RISK"])

# Number of symbols
number_of_symbols = len(df)

# Monte Carlo Simulation
num_of_portfolios = 100000

# Initialize arrays to store simulation results
all_weights = np.zeros((num_of_portfolios, number_of_symbols))
ret_arr = np.zeros(num_of_portfolios)
vol_arr = np.zeros(num_of_portfolios)
sharpe_arr = np.zeros(num_of_portfolios)

# Start the simulations
for ind in range(num_of_portfolios):
    # Generate random weights and normalize them
    weights = np.random.random(number_of_symbols)
    weights /= np.sum(weights)
    all_weights[ind, :] = weights

    # Calculate the expected returns (annualized)
    ret_arr[ind] = np.sum(df["LOG RETURN"] * weights)

    # Calculate the volatility (annualized)
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Calculate the Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

# Combine results into a DataFrame
simulations_df = pd.DataFrame({
    "Returns": ret_arr,
    "Volatility": vol_arr,
    "Sharpe Ratio": sharpe_arr
})

# Add the portfolio weights as a separate column
simulations_df["Portfolio Weights"] = list(all_weights)

# Print results
print('')
print('='*80)
print('SIMULATIONS RESULT:')
print('-'*80)
print(simulations_df.head())
print('-'*80)

# Get the optimal portfolio (highest Sharpe Ratio)
max_sharpe_idx = simulations_df["Sharpe Ratio"].idxmax()
optimal_portfolio = simulations_df.iloc[max_sharpe_idx]

print('')
print('='*80)
print('OPTIMAL PORTFOLIO:')
print('-'*80)
print(optimal_portfolio)
print('-'*80)
# Return the Max Sharpe Ratio from the run.
max_sharpe_ratio = simulations_df.loc[simulations_df['Sharpe Ratio'].idxmax()]

# Return the Min Volatility from the run.
min_volatility = simulations_df.loc[simulations_df['Volatility'].idxmin()]

print('')
print('='*80)
print('MIN VOLATILITY:')
print('-'*80)
print(min_volatility)
print('-'*80)


# Plot the data on a Scatter plot.
plt.scatter(
    y=simulations_df['Returns'],
    x=simulations_df['Volatility'],
    c=simulations_df['Sharpe Ratio'],
    cmap='RdYlBu'
)

# Give the Plot some labels, and titles.
plt.title('Portfolio Returns Vs. Risk')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')

# Access values by position using .iloc
plt.scatter(
    max_sharpe_ratio.iloc[1],  # x-axis (Volatility)
    max_sharpe_ratio.iloc[0],  # y-axis (Returns)
    marker=(5, 1, 0),
    color='r',
    s=600
)

plt.scatter(
    min_volatility.iloc[1],  # x-axis (Volatility)
    min_volatility.iloc[0],  # y-axis (Returns)
    marker=(5, 1, 0),
    color='b',
    s=600
)


# Finally, show the plot.
plt.show()

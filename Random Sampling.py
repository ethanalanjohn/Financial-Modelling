import numpy as np

# Provided price data
price_data_frame = {
    "STOCK NAME": ["DCBBANK.NS", "RELIANCE.NS", "FACT.NS", "ZOMATO.NS", "KSOLVES.NS",
                   "HAPPSTMNDS.NS", "ITC.NS", "PCBL.NS", "HDFC.NS", "HINDUNILVR.NS"],
    "CURRENT STOCK PRICE": [109.26, 1239.85, 893.45, 243.90, 916.45, 718.50, 439.05, 351.85, 351.85, 2591.70],
    "PREDICTED STOCK PRICE": [118.64, 1325.27, 1054.248782, 264.346153, 1083.01, 759.39, 463.61, 373.07, 387.63, 2591.70],
    "RISK": [0.1331, 0.0815, 0.27, 0.133328, 0.1823, 0.1311, 0.0704, 0.1521, 0.1521, 0.0633],
    "INDUSTRY": [2, 3, 1, 1, 4, 4, 7, 6, 2, 7]
}

# Convert data to arrays for processing
current_prices = np.array(price_data_frame["CURRENT STOCK PRICE"])
predicted_prices = np.array(price_data_frame["PREDICTED STOCK PRICE"])
risks = np.array(price_data_frame["RISK"])

# Calculate expected returns
expected_returns = (predicted_prices / current_prices) - 1

# Create a covariance matrix (diagonal matrix using risks as variances)
covariance_matrix = np.diag(risks**2)

# Number of symbols
number_of_symbols = len(expected_returns)

# Define a function to calculate the Sharpe Ratio
def calculate_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return sharpe_ratio

# Define random sampling parameters
num_samples = 100000  # Number of random weight sets to test
best_sharpe_ratio = -np.inf
best_weights = None
risk_free_rate = 0.0693  # Risk-free rate (e.g., government bond yield)

# Perform random sampling
for _ in range(num_samples):
    # Generate random weights that sum to 1
    weights = np.random.rand(number_of_symbols)
    weights /= np.sum(weights)
    
    # Calculate the Sharpe ratio for the weights
    sharpe_ratio = calculate_sharpe_ratio(weights, expected_returns, covariance_matrix, risk_free_rate)
    
    # Update best weights if Sharpe ratio is better
    if sharpe_ratio > best_sharpe_ratio:
        best_sharpe_ratio = sharpe_ratio
        best_weights = weights

# Print the results
print('')
print('=' * 80)
print('OPTIMIZED SHARPE RATIO:')
print('-' * 80)
print(f'Sharpe Ratio: {best_sharpe_ratio}')
print(f'Weights: {best_weights}')
print('-' * 80)
# Function to calculate portfolio metrics
def get_metrics(weights):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return {
        "Portfolio Return": portfolio_return,
        "Portfolio Volatility": portfolio_volatility,
        "Sharpe Ratio": sharpe_ratio
    }

# Grab the metrics using the best weights
optimized_metrics = get_metrics(weights=best_weights)

# Print the Optimized Weights
print('')
print('=' * 80)
print('OPTIMIZED WEIGHTS:')
print('-' * 80)
print(best_weights)
print('-' * 80)

# Print the Optimized Metrics
print('')
print('=' * 80)
print('OPTIMIZED METRICS:')
print('-' * 80)
for key, value in optimized_metrics.items():
    print(f'{key}: {value}')
print('-' * 80)

# Function to calculate portfolio volatility
def calculate_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Grid search for optimal weights minimizing portfolio volatility
best_volatility = np.inf
best_weights_volatility = None

# Iterate over random weights
iterations = 10000  # Number of random samples
for _ in range(iterations):
    weights = np.random.rand(number_of_symbols)  # Generate random weights
    weights = weights / np.sum(weights)  # Normalize weights to sum to 1
    volatility = calculate_volatility(weights, covariance_matrix)
    if volatility < best_volatility:  # Check if this is the minimum volatility
        best_volatility = volatility
        best_weights_volatility = weights

# Print the results
print('')
print('=' * 80)
print('OPTIMIZED VOLATILITY:')
print('-' * 80)
print(f'Volatility: {best_volatility}')
print(f'Weights: {best_weights_volatility}')
print('-' * 80)

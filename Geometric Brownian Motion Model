import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
# Risk Free Rate
mu = 0.069
# number of steps
n = 1000
# time in years
T = 1
# number of sims
M = 1000
# initial stock price
S0 =3614.5
# volatility
sigma = 0.0853

# calc each time step
dt = T/n

# simulation using numpy arrays
St = np.exp(
    (mu - sigma ** 2 / 2) * dt
    + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T
)

# include array of 1's
St = np.vstack([np.ones(M), St])

# multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
St = S0 * St.cumprod(axis=0)
# Define time interval correctly
time = np.linspace(0,T,n+1)

# Require numpy array that is the same shape as St
tt = np.full(shape=(M,n+1), fill_value=time).T
plt.plot(tt, St)
plt.xlabel("Years $(t)$")
plt.ylabel("Stock Price $(S_t)$")
plt.title(
    "Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(S0, mu, sigma)
)
plt.show()
St_max = np.max(St)
St_min = np.min(St)
St_avg = np.mean(St)


print(f"Maximum Stock Price: {St_max:.2f}")
print(f"Minimum Stock Price: {St_min:.2f}")
print(f"Average Stock Price: {St_avg:.2f}")
terminal_prices = St[-1, :]
probability_above_S0 = np.mean(terminal_prices > S0)
print(f"Probability that the stock price will be greater than S0: {probability_above_S0:.2%}")
median_price = np.median(terminal_prices)
print(f"Median Terminal Stock Price: {median_price:.2f}")
# Plot Histogram of Terminal Prices
plt.hist(terminal_prices, bins=50, color='skyblue', edgecolor='black')
plt.axvline(S0, color='red', linestyle='--', label=f'S0 = {S0}')
plt.xlabel('Terminal Stock Price')
plt.ylabel('Frequency')
plt.title('Distribution of Terminal Stock Prices')
plt.legend()
plt.show()

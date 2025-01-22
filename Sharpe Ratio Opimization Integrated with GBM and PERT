import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import yfinance as yf
from datetime import datetime,timedelta
import pyomo.environ as pyo
from pyomo.environ import SolverFactory
ticker = "ZOMATO.NS"
industry = 1
#agro =1
#BANKING=2
#ENERGY=3
#IT=4
#HOSPITALITY=5
#RAW=6
#FMCG=7
#HEALTHCARE=8
stock = yf.Ticker(ticker)
price = stock.info['currentPrice']
print(price)
 
start_date = "2010-01-01"
end_date = "2024-12-31"
monthly_data = yf.download(ticker, start="2014-01-01", end="2024-08-08", interval="1mo")
monthly_data["Percent Change"] = monthly_data["Close"].pct_change() * 100
std_dev = (monthly_data["Percent Change"].std())/100
print(std_dev)
# Risk Free Rate
mu = 0.069
# number of steps
n = 1000
# time in years
T = 1
# number of sims
M = 1000
# initial stock price
S0 =price
# volatility
sigma = std_dev
print(S0)
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
St_max = np.max(St)
St_min = np.min(St)
St_avg = np.mean(St)


print(f"Maximum Stock Price: {St_max:.2f}")
print(f"Minimum Stock Price: {St_min:.2f}")
print(f"Average Stock Price: {St_avg:.2f}")
pert=(4*St_avg+St_min+St_max)/6
print(pert)
import pandas as pd

# Define columns
columns = [
    "STOCK NAME",
    "NUMBER OF STOCKS",
    "WEIGHTS",
    "PREDICTED STOCK PRICE",
    "RISK",
    "CURRENT STOCK PRICE",
    "RISK WEIGHTS",
    "COSTS",
    "INDUSTRY"
]

# Initialize DataFrame with 0 values
# For text-based columns like "STOCK NAME" and "INDUSTRY", we'll use empty strings.
data = {
    "STOCK NAME": [""],  # Placeholder for stock names
    "NUMBER OF STOCKS": [0],
    "WEIGHTS": [0],
    "PREDICTED STOCK PRICE": [0],
    "RISK": [0],
    "CURRENT STOCK PRICE": [0],
    "RISK WEIGHTS": [0],
    "COSTS": [0],
    "INDUSTRY": ["0"]  # Placeholder for industry names
}

# Create the DataFrame
table = pd.DataFrame(data)

print(table)
new_row = {"STOCK NAME":ticker , "NUMBER OF STOCKS": 0,"WEIGHTS":0,"PREDICTED STOCK PRICE":pert,"RISK":std_dev,"CURRENT STOCK PRICE":price,"RISK WEIGHTS":0,"COSTS":0,"INDUSTRY":0}

table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)
print(table)
import numpy as np
from scipy.optimize import minimize

# Data for the stocks
stock_data = {
    "STOCK NAME": ["DCBBANK.NS", "RELIANCE.NS", "FACT.NS ", "ZOMATO.NS", "KSOLVES.NS",
                   "HAPPSTMNDS.NS", "ITC.NS", "PCBL.NS", "HDFC.NS", "HINDUNILVR.NS"],
    "CURRENT STOCK PRICE": [109.26, 1239.85,  893.45 , 243.90, 916.45, 718.50, 439.05, 351.85, 351.85, 2591.70],
    "PREDICTED STOCK PRICE": [118.64, 1325.27,1054.248782, 264.346153, 1083.01, 759.39, 463.61, 373.07, 387.63, 2591.70],
    "RISK": [0.1331, 0.0815, 0.27,0.133328, 0.1823, 0.1311, 0.0704, 0.1521, 0.1521, 0.0633],
    "INDUSTRY": [2, 3, 1, 1, 4, 4, 7, 6, 2, 7]
}

capital_available = 100000  # Total capital
risk_allowance = 0.10  # Maximum risk allowance
min_allocation_percentage = 0.1  # Minimum allocation of 0.1% per industry

# Convert stock data to numpy arrays for easier manipulation
current_prices = np.array(stock_data["CURRENT STOCK PRICE"])
predicted_prices = np.array(stock_data["PREDICTED STOCK PRICE"])
risks = np.array(stock_data["RISK"])
industries = np.array(stock_data["INDUSTRY"])

num_stocks = len(current_prices)

# Objective: Maximize profit = sum((predicted price - current price) * quantity)
def objective(x):
    return -np.sum((predicted_prices - current_prices) * x)

# Constraint: Total capital used <= available capital
def capital_constraint(x):
    return capital_available - np.sum(current_prices * x)

# Constraint: Weighted risk <= risk allowance
def risk_constraint(x):
    total_capital_used = np.sum(current_prices * x)
    if total_capital_used == 0:
        return risk_allowance
    weighted_risk = np.sum(risks * current_prices * x) / total_capital_used
    return risk_allowance - weighted_risk

# Constraint: Minimum allocation per industry
def industry_allocation_constraint(x, industry_id):
    industry_allocation = np.sum(current_prices * x * (industries == industry_id))
    return industry_allocation - (min_allocation_percentage * capital_available)

# Bounds: Non-negative integer values for the number of stocks
bounds = [(0, None) for _ in range(num_stocks)]

# Initial guess: Start with zero stocks
initial_guess = np.ones(num_stocks) * 10  # Start with 10 units of each stock

# Define constraints in a format suitable for SciPy
constraints = [
    {"type": "ineq", "fun": capital_constraint},  # Ensure capital constraint
    {"type": "ineq", "fun": risk_constraint},     # Ensure risk constraint
]

# Add industry allocation constraints
unique_industries = np.unique(industries)
for industry_id in unique_industries:
    constraints.append({
        "type": "ineq",
        "fun": lambda x, industry_id=industry_id: industry_allocation_constraint(x, industry_id)
    })

# Solve the optimization problem
result = minimize(
    objective,
    initial_guess,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"disp": True}
)

# Extract the optimal solution
optimal_quantities = np.round(result.x).astype(int)
total_profit = -result.fun

# Display results
print("Optimal Number of Stocks:")
for stock, qty in zip(stock_data["STOCK NAME"], optimal_quantities):
    print(f"{stock}: {qty}")

print(f"\nTotal Profit: {total_profit:.2f}")
print("Capital constraint:", capital_constraint(initial_guess))
print("Risk constraint:", risk_constraint(initial_guess))
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

# Create the problem
# Data for the stocks
stock_data = {
    "STOCK NAME": ["DCBBANK.NS", "RELIANCE.NS", "FACT.NS ", "COROMANDEL.NS", "KSOLVES.NS",
                   "HAPPSTMNDS.NS", "ITC.NS", "PCBL.NS", "HDFC.NS", "HINDUNILVR.NS"],
    "CURRENT STOCK PRICE": [109.26, 1239.85,  893.45 , 1808.00, 916.45, 718.50, 439.05, 351.85, 351.85, 2591.70],
    "PREDICTED STOCK PRICE": [118.64, 1325.27,1054.248782, 1914.74, 1083.01, 759.39, 463.61, 373.07, 387.63, 2591.70],
    "RISK": [0.1331, 0.0815, 0.100552   , 0.0742, 0.1823, 0.1311, 0.0704, 0.1521, 0.1521, 0.0633],
    "INDUSTRY": [2, 3, 1, 1, 4, 4, 7, 6, 2, 7]
}

capital_available = 100000  # Total capital
risk_allowance = 0.12  # Maximum risk allowance
min_allocation_percentage = 0.1  # Minimum allocation of 0.1% per industry

# Convert stock data to numpy arrays for easier manipulation
current_prices = np.array(stock_data["CURRENT STOCK PRICE"])
predicted_prices = np.array(stock_data["PREDICTED STOCK PRICE"])
risks = np.array(stock_data["RISK"])
industries = np.array(stock_data["INDUSTRY"])

from pulp import LpMaximize, LpProblem, LpVariable, lpSum

# Create the problem
problem = LpProblem("Maximize_Profit", LpMaximize)

# Define integer variables for stock quantities
quantities = [LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(num_stocks)]

# Objective function: Maximize profit
profit = lpSum((predicted_prices[i] - current_prices[i]) * quantities[i] for i in range(num_stocks))
problem += profit

# Constraint: Total capital used <= available capital
total_capital_used = lpSum(current_prices[i] * quantities[i] for i in range(num_stocks))
problem += total_capital_used <= capital_available

# Reformulated Constraint: Weighted risk <= risk allowance
risk_weighted_sum = lpSum(risks[i] * current_prices[i] * quantities[i] for i in range(num_stocks))
problem += risk_weighted_sum <= risk_allowance * total_capital_used

# Constraint: Minimum allocation per industry
for industry_id in unique_industries:
    problem += lpSum(
        current_prices[i] * quantities[i] for i in range(num_stocks) if industries[i] == industry_id
    ) >= min_allocation_percentage * capital_available

# Solve the problem
problem.solve()

# Extract the optimal solution
optimal_quantities = [int(quantities[i].varValue) for i in range(num_stocks)]
total_profit = sum((predicted_prices[i] - current_prices[i]) * optimal_quantities[i] for i in range(num_stocks))

# Display results
print("Optimal Number of Stocks:")
for stock, qty in zip(stock_data["STOCK NAME"], optimal_quantities):
    print(f"{stock}: {qty}")

print(f"\nTotal Profit: {total_profit:.2f}")


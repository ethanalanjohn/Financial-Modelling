# --- Imports & setup ---
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- 1) Universe: NIFTY 50 tickers ---
# Scrape tickers from Wikipedia (NIFTY 50 constituents)
nifty_tbls = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50")
# The table with tickers is typically index 1; safeguard fallback:
cand = [t for t in nifty_tbls if 'Symbol' in t.columns]
nifty_df = cand[0].copy()

# Normalize ticker symbols for Yahoo Finance
tickers = (nifty_df['Symbol']
           .astype(str)
           .str.replace('.', '-', regex=False)
           .apply(lambda s: s + ".NS")
           .unique()
           .tolist())

# --- 2) Download prices (Adj Close), build month-end returns ---
end_date = "2024-12-27"
start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=8)).strftime("%Y-%m-%d")

px = yf.download(tickers=tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
# Drop all-empty columns, forward-fill occasional missing, then drop rows still empty
px = px.dropna(axis=1, how='all').ffill().dropna(how='all')

# True month-end prices per ticker -> monthly returns
px_m = px.resample('M').last()
rets_m = px_m.pct_change().dropna(how='all')

# Align to tickers with enough history
min_months = 36  # require at least 3 years of monthly data
valid = rets_m.count()[rets_m.count() >= min_months].index.tolist()
rets_m = rets_m[valid].dropna()

# --- 3) Build expected returns (mu) and covariance (Sigma), annualized ---
mu_annual = rets_m.mean() * 12
Sigma_annual = rets_m.cov() * 12

# Safety: remove any columns with NaNs in mu/Sigma
ok = mu_annual.dropna().index.intersection(Sigma_annual.dropna(axis=0, how='all').dropna(axis=1, how='all').columns)
mu_annual = mu_annual.loc[ok]
Sigma_annual = Sigma_annual.loc[ok, ok]

# --- 4) Optimize portfolio: Max Sharpe with weight cap & fully invested ---
def neg_sharpe(w, mu, Sigma):
    r = float(w @ mu.values)
    v = float(np.sqrt(w @ Sigma.values @ w))
    # Add tiny epsilon to avoid division by zero
    return -(r / (v + 1e-12))

N = len(mu_annual)
max_allocation = 0.15  # 15% per name cap
bounds = [(0.0, max_allocation)] * N
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
w0 = np.repeat(1.0 / N, N)

opt = minimize(neg_sharpe, w0, args=(mu_annual, Sigma_annual),
               method='SLSQP', bounds=bounds, constraints=constraints,
               options={'maxiter': 1000, 'ftol': 1e-9})

if not opt.success:
    print("Optimization warning:", opt.message)

w_star = pd.Series(opt.x, index=mu_annual.index).sort_values(ascending=False)

# --- 5) Build portfolio performance (static weights over the sample) ---
port_rets_m = (rets_m[w_star.index] @ w_star).dropna()
port_cum = (1 + port_rets_m).cumprod()

# --- 6) NIFTY 50 benchmark (monthly) ---
nifty = yf.download("^NSEI", start=rets_m.index.min(), end=end_date, auto_adjust=False)['Adj Close']
nifty_m = nifty.resample('M').last().pct_change().dropna()
nifty_cum = (1 + nifty_m).cumprod()

# Align indices
idx = port_cum.index.intersection(nifty_cum.index)
combined = pd.DataFrame({
    'Portfolio': port_cum.reindex(idx),
    'NIFTY 50': nifty_cum.reindex(idx)
}).dropna()

# --- 7) Simple tear-sheet metrics ---
def cagr(series):
    if len(series) < 2:
        return np.nan
    years = (series.index[-1] - series.index[0]).days / 365.25
    return series.iloc[-1]**(1/years) - 1

def ann_vol(returns_m):
    return returns_m.std() * np.sqrt(12)

def sharpe(returns_m, rf_annual=0.0):
    rf_m = rf_annual / 12
    ex = returns_m - rf_m
    return ex.mean() / (ex.std() + 1e-12) * np.sqrt(12)

def max_drawdown(series):
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    return dd.min()

port_cagr = cagr(port_cum)
nifty_cagr = cagr(nifty_cum.reindex(idx))
port_vol = ann_vol(port_rets_m.reindex(idx))
nifty_vol = ann_vol(nifty_m.reindex(idx))
port_sharpe = sharpe(port_rets_m.reindex(idx))
nifty_sharpe = sharpe(nifty_m.reindex(idx))
port_mdd = max_drawdown(port_cum.reindex(idx))
nifty_mdd = max_drawdown(nifty_cum.reindex(idx))

metrics = pd.DataFrame({
    'CAGR': [port_cagr, nifty_cagr],
    'Ann.Vol': [port_vol, nifty_vol],
    'Sharpe': [port_sharpe, nifty_sharpe],
    'Max DD': [port_mdd, nifty_mdd]
}, index=['Portfolio', 'NIFTY 50'])

print("Top weights:")
print(w_star.head(15).round(4))
print("\nPerformance metrics (monthly backtest):")
print(metrics.round(3))

# --- 8) Plot cumulative performance ---
ax = combined.plot(figsize=(9,5), title="Portfolio vs NIFTY 50 (Monthly, Cum. Growth)")
ax.set_ylabel("Growth of 1")
ax.grid(True)
plt.show()

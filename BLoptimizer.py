import yfinance as yf
import numpy as np
import pandas as pd
import os
from datetime import timedelta

def get_nifty50_history(period="30y", interval="1d"):
    """Fetch 30 years of daily NIFTY 50 (India) index data from Yahoo Finance."""
    df = yf.download("^NSEI", period=period, interval=interval, progress=False).reset_index()
    df["Symbol"] = "NIFTY50"
    return df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Symbol"]]

nifty50 = get_nifty50_history()
print("30-year daily history loaded")
print(nifty50.head())

# Load (replace "..." with your file path/DataFrame source)
df = pd.read_csv("...")  # cleaned NSE history with columns: Date, Close, Symbol, ...

# Normalize & sort
df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Close", "Symbol"]).sort_values(["Symbol", "Date"])

symbols = ["a,"b","c"]
""" Enter tickers of assets in a list"""
missing = [s for s in symbols if s not in set(df["Symbol"])]
print("Requested:", symbols, "| Missing:", (missing or "None"))

cutoff = pd.Timestamp.today() - pd.DateOffset(years=10)
frames, no10y = [], []

for s in symbols:
    sdf = df.loc[df["Symbol"] == s]
    if sdf.empty: 
        continue
    sub = sdf.loc[sdf["Date"] >= cutoff]
    use = sub if len(sub) else sdf
    if sub.empty: 
        no10y.append(s)
    use = use.assign(**{"Daily % Change": use["Close"].pct_change() * 100})
    frames.append(use[["Date", "Symbol", "Daily % Change"]])

result = (
    pd.concat(frames, ignore_index=True)
      .sort_values(["Symbol", "Date"])
      .reset_index(drop=True)
    if frames else pd.DataFrame(columns=["Date", "Symbol", "Daily % Change"])
)

print("Included:", sorted(result["Symbol"].unique()))
if no10y: print("No last-10Y data (used full history):", no10y)
print(result.groupby("Symbol").head(3))

# --- Append new daily % changes from Yahoo Finance to your existing table ---

# pip install yfinance pandas openpyxl  (run once if needed)


PCT_FILE = "daily_percent_changes.csv"                 # your existing table
OUTPUT_FILE = "daily_percent_changes.csv"              # overwrite same file
SYMBOLS = ["HINDCOPPER", "FORTIS", "IEX", "WAAREEENER", "GANESHHOU"]

# If any Yahoo tickers differ from the pattern {SYMBOL}.NS, specify here:
# e.g., {"GANESHHOU": "GANESHHOUC.NS"} if you later confirm that mapping.
yahoo_overrides = {
    # "GANESHHOU": "GANESHHOUC.NS",
}

def sym_to_yahoo(sym: str) -> str:
    return yahoo_overrides.get(sym, f"{sym}.NS")

# --- Load existing percent-change table (or create empty) ---
if os.path.exists(PCT_FILE):
    existing = pd.read_csv(PCT_FILE)
    existing["Date"] = pd.to_datetime(existing["Date"], errors="coerce")
    existing["Symbol"] = existing["Symbol"].astype(str).str.strip().str.upper()
    existing = existing.dropna(subset=["Date","Symbol"]).sort_values(["Symbol","Date"]).reset_index(drop=True)
else:
    existing = pd.DataFrame(columns=["Date","Symbol","Daily % Change"])
    existing["Date"] = pd.to_datetime(existing["Date"])

print(f"Loaded existing rows: {len(existing)}")

# --- Determine start date per symbol (append from last known + 1 day) ---
last_dates = (
    existing.groupby("Symbol")["Date"].max().to_dict()
    if not existing.empty else {}
)

# --- Download, compute % change, and append ---
added_rows = 0
no_data = []      # tickers that returned no rows from Yahoo
not_found = []    # tickers that have never existed in your table AND returned no data
per_symbol_added = {}

frames = [existing[["Date","Symbol","Daily % Change"]]]

for sym in SYMBOLS:
    start_date = last_dates.get(sym, None)
    if start_date is None:
        # never seen before in the table -> pull plenty of history
        start_param = "30y"
        # yfinance needs either start=... or period=... ; we'll use period when unknown
        df = yf.download(sym_to_yahoo(sym), period="30y", interval="1d", auto_adjust=False, progress=False)
    else:
        # append from next day
        start = (start_date + timedelta(days=1)).date().isoformat()
        df = yf.download(sym_to_yahoo(sym), start=start, interval="1d", auto_adjust=False, progress=False)

    if df is None or df.empty:
        no_data.append(sym)
        if start_date is None:
            not_found.append(sym)
        continue

    df = df.reset_index().rename(columns={"Date":"Date"})
    # Ensure timezone-naive dates
    if pd.api.types.is_datetime64_any_dtype(df["Date"]):
        if getattr(df["Date"].dt, "tz", None) is not None:
            df["Date"] = df["Date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    # Compute daily % change from Close
    df["Daily % Change"] = df["Close"].pct_change() * 100

    # First new row after an append won’t have previous day’s close from the old table,
    # so its % change will be NaN — that’s expected and preserved.
    out = df[["Date","Daily % Change"]].copy()
    out["Symbol"] = sym

    frames.append(out)
    per_symbol_added[sym] = len(out)

# --- Concatenate, de-duplicate, sort, save ---
result = pd.concat(frames, ignore_index=True)
# Drop duplicate (Symbol, Date) keeping the existing row first
result = result.sort_values(["Symbol","Date"]).drop_duplicates(subset=["Symbol","Date"], keep="first").reset_index(drop=True)

# Enforce column order and types
result = result[["Date","Symbol","Daily % Change"]]
result["Date"] = pd.to_datetime(result["Date"], errors="coerce")

result.to_csv(OUTPUT_FILE, index=False)

print("\n--- Summary ---")
print(f"Symbols requested: {SYMBOLS}")
print("Yahoo tickers used:", {s: sym_to_yahoo(s) for s in SYMBOLS})
print("Rows now in file:", len(result))
print("Rows added per symbol (raw, before de-dup):", per_symbol_added)
if no_data:
    print("Tickers with no new data returned by Yahoo (could be mapping/listing/holiday issues):", no_data)
if not_found:
    print("Tickers that have never appeared in your table AND returned no data from Yahoo:", not_found)



# ---- paths ----
PCT_FILE   = "daily_percent_changes.csv"      # existing table
NIFTY_FILE = "nifty50_history_master.csv"     # NIFTY index history

# ---- load existing table (or create empty) ----
if os.path.exists(PCT_FILE):
    pct = pd.read_csv(PCT_FILE)
    pct["Date"] = pd.to_datetime(pct["Date"], errors="coerce")
    pct["Symbol"] = pct["Symbol"].astype(str).str.strip().str.upper()
    pct = pct.dropna(subset=["Date","Symbol"])
else:
    pct = pd.DataFrame(columns=["Date","Symbol","Daily % Change"])
    pct["Date"] = pd.to_datetime(pct["Date"], errors="coerce")

# ---- load NIFTY history ----
if NIFTY_FILE.endswith(".xlsx"):
    idx = pd.read_excel(NIFTY_FILE)
else:
    idx = pd.read_csv(NIFTY_FILE)

# Normalize columns
idx.rename(columns={"Date":"Date","Close":"Close"}, inplace=True)
idx["Date"] = pd.to_datetime(idx["Date"], errors="coerce")

# ✨ MAKE 'Close' NUMERIC (fix for your error)
# strip commas/whitespace, then coerce to numeric
idx["Close"] = (
    idx["Close"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
idx["Close"] = pd.to_numeric(idx["Close"], errors="coerce")

# Clean and sort
idx = idx.dropna(subset=["Date","Close"]).sort_values("Date")

# Compute daily % change
idx["Daily % Change"] = idx["Close"].pct_change() * 100
idx["Symbol"] = "NIFTY50"

idx_out = idx[["Date","Symbol","Daily % Change"]]

# ---- append, dedupe, sort, save ----
combined = pd.concat([pct, idx_out], ignore_index=True)
combined = (
    combined
    .drop_duplicates(subset=["Symbol","Date"], keep="last")
    .sort_values(["Symbol","Date"])
    .reset_index(drop=True)
)

combined.to_csv(PCT_FILE, index=False)

# ---- quick sanity report ----
print("Added NIFTY50 daily % changes.")
print("Rows now in file:", len(combined))
print("Symbols in file:", sorted(combined['Symbol'].unique()))
print("Sample NIFTY50 rows:")
print(combined[combined['Symbol'] == 'NIFTY50'].head())





# --- Load the data ---
FILE = "daily_percent_changes.csv"  

if FILE.endswith(".xlsx"):
    df = pd.read_excel(FILE)
else:
    df = pd.read_csv(FILE)

# ---  Clean & prepare ---
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
df["Daily % Change"] = pd.to_numeric(df["Daily % Change"], errors="coerce")

# Drop missing values
df = df.dropna(subset=["Date", "Daily % Change", "Symbol"])

# --- Calculate average daily return for each symbol ---
# Convert percent to decimal before averaging
daily_stats = (
    df.groupby("Symbol")["Daily % Change"]
    .agg(["mean", "std", "count"])
    .rename(columns={"mean": "Avg Daily %", "std": "Daily Std %", "count": "Days"})
)

# --- Convert daily avg return to annual expected return ---
# Expected Annual Return ≈ (1 + avg_daily_return)^252 - 1
daily_stats["Expected Annual Return %"] = (
    (1 + daily_stats["Avg Daily %"] / 100) ** 252 - 1
) * 100

# --- 5️⃣ Optional: Add annualized volatility too ---
daily_stats["Annualized Volatility %"] = daily_stats["Daily Std %"] * np.sqrt(252)

# --- 6️⃣ Sort & preview ---
daily_stats = daily_stats.sort_values("Expected Annual Return %", ascending=False).reset_index()

# --- 7️⃣ Save & display ---
out_path = "annual_expected_returns.csv"
daily_stats.to_csv(out_path, index=False)

print("Annual expected returns calculated and saved to:", out_path)
print("\nSample output:")
print(daily_stats.head(10))





PCT_FILE = "daily_percent_changes.csv"   # your combined daily returns (%)
BETA_FILE = "asset_betas.csv"            # previously saved betas (optional)
MARKET_SYM = "NIFTY50"
TOL = 1e-6   # numerical tolerance for equality checks

def _beta_cov(rs, rm):
    cov = np.cov(rs, rm, ddof=1)[0,1]
    var_m = np.var(rm, ddof=1)
    return np.nan if var_m == 0 else cov / var_m

def _beta_reg(rs, rm):
    # slope of stock ~ beta*market + alpha via least squares
    # polyfit returns slope, intercept; we only need slope
    slope, intercept = np.polyfit(rm, rs, 1)
    return slope

def _beta_corr(rs, rm):
    corr = np.corrcoef(rs, rm)[0,1]
    std_s = np.std(rs, ddof=1)
    std_m = np.std(rm, ddof=1)
    if std_m == 0:
        return np.nan
    return corr * (std_s / std_m)

def _safe_num(series):
    return pd.to_numeric(series, errors="coerce")

# 1) Load & clean
assert os.path.exists(PCT_FILE), f"Missing file: {PCT_FILE}"
df = pd.read_csv(PCT_FILE)

# normalize dtypes
df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
df["Daily % Change"] = _safe_num(df["Daily % Change"])
df = df.dropna(subset=["Date","Symbol","Daily % Change"]).sort_values(["Symbol","Date"]).reset_index(drop=True)

print(" Loaded rows:", len(df))
print(" Symbols:", sorted(df["Symbol"].unique().tolist()))
assert MARKET_SYM in df["Symbol"].unique(), f"Market symbol '{MARKET_SYM}' not found."

# 2) Split market & stocks
mkt = (df[df["Symbol"] == MARKET_SYM]
       .loc[:, ["Date", "Daily % Change"]]
       .rename(columns={"Daily % Change":"MktPct"})
       .drop_duplicates(subset="Date")
       .sort_values("Date"))

stocks = (df[df["Symbol"] != MARKET_SYM]
          .loc[:, ["Date","Symbol","Daily % Change"]]
          .rename(columns={"Daily % Change":"StockPct"}))

assert not mkt.empty, "Market series is empty after cleaning."

# diagnostics for the raw data
dups = stocks.duplicated(subset=["Symbol","Date"]).sum()
if dups:
    print(f"⚠️ Found {dups} duplicate (Symbol, Date) rows in stocks. Keeping first in merges.")
# basic outlier check (not an error; just info)
outlier_cut = 50  # %
outliers = (stocks["StockPct"].abs() > outlier_cut).sum()
if outliers:
    print(f"⚠️ Found {outliers} daily returns with abs(change) > {outlier_cut}% in stocks (may be real gap-ups/downs).")

# 3) function to compute & cross-check betas for one symbol
def check_one_symbol(sym):
    sdf = stocks[stocks["Symbol"] == sym].drop_duplicates(subset="Date").sort_values("Date")
    if sdf.empty:
        return {"Symbol": sym, "Status":"NO DATA"}

    # strict join: only exact same dates
    strict = pd.merge(sdf, mkt, on="Date", how="inner").dropna(subset=["StockPct","MktPct"])

    # tolerant join: asof backward (use latest prior market close)
    asof = pd.merge_asof(sdf.sort_values("Date"),
                         mkt.sort_values("Date"),
                         on="Date", direction="backward").dropna(subset=["StockPct","MktPct"])

    results = {"Symbol": sym}

    for label, merged in [("STRICT", strict), ("ASOF", asof)]:
        n = len(merged)
        results[f"{label}_points"] = n
        if n >= 2:
            rs = merged["StockPct"]/100.0
            rm = merged["MktPct"]/100.0

            beta_cov = _beta_cov(rs, rm)
            beta_reg = _beta_reg(rs, rm)
            beta_crr = _beta_corr(rs, rm)
            r2 = (np.corrcoef(rs, rm)[0,1]**2) if n > 1 else np.nan

            # internal consistency checks
            agree = (np.allclose(beta_cov, beta_reg, atol=TOL, equal_nan=True)
                     and np.allclose(beta_cov, beta_crr, atol=TOL, equal_nan=True))
            results[f"{label}_beta_cov"] = beta_cov
            results[f"{label}_beta_reg"] = beta_reg
            results[f"{label}_beta_corr"] = beta_crr
            results[f"{label}_R2"] = r2
            results[f"{label}_agree"] = bool(agree)
            results[f"{label}_date_min"] = merged["Date"].min().date()
            results[f"{label}_date_max"] = merged["Date"].max().date()
        else:
            results[f"{label}_beta_cov"] = np.nan
            results[f"{label}_beta_reg"] = np.nan
            results[f"{label}_beta_corr"] = np.nan
            results[f"{label}_R2"] = np.nan
            results[f"{label}_agree"] = False
            results[f"{label}_date_min"] = None
            results[f"{label}_date_max"] = None

    return results

# 4) run checks
symbols = sorted(stocks["Symbol"].unique().tolist())
rows = [check_one_symbol(s) for s in symbols]
chk = pd.DataFrame(rows)

# 5) Show summary table
cols_order = ["Symbol",
              "STRICT_points","STRICT_beta_cov","STRICT_beta_reg","STRICT_beta_corr","STRICT_R2","STRICT_agree","STRICT_date_min","STRICT_date_max",
              "ASOF_points","ASOF_beta_cov","ASOF_beta_reg","ASOF_beta_corr","ASOF_R2","ASOF_agree","ASOF_date_min","ASOF_date_max"]
existing_cols = [c for c in cols_order if c in chk.columns]
chk = chk[existing_cols].sort_values("ASOF_points", ascending=False)

print("\n===== CHECK SUMMARY (per symbol) =====")
print(chk.to_string(index=False))

# 6) Compare with previously saved betas (if available)
if os.path.exists(BETA_FILE):
    saved = pd.read_csv(BETA_FILE)
    saved["Symbol"] = saved["Symbol"].astype(str).str.strip().str.upper()
    # we compare against ASOF_beta_cov (our primary)
    comp = (saved.merge(chk[["Symbol","ASOF_beta_cov","ASOF_R2","ASOF_points"]],
                        on="Symbol", how="left")
                 .rename(columns={"ASOF_beta_cov":"Recomp_Beta",
                                  "ASOF_R2":"Recomp_R2",
                                  "ASOF_points":"Recomp_Points"}))
    comp["Delta_Beta"] = comp["Beta"] - comp["Recomp_Beta"]
    print("\n===== COMPARISON vs asset_betas.csv =====")
    print(comp.to_string(index=False))
else:
    print("\n(No existing asset_betas.csv found to compare against.)")

print("\n Checker finished. Read the CHECK SUMMARY to verify:")
print("- STRICT_points: exact same trading days with index")
print("- ASOF_points: tolerant alignment (backward) — should be >= STRICT")
print("- *_agree: all three beta methods match numerically")
print("- *_R2: goodness of fit")
print("- date_min/max: window used for each method")




RETURNS_FILE = "annual_expected_returns.csv"   # From earlier step
BETAS_FILE = "asset_betas.csv"                 # From beta calculation
RISK_FREE_RATE = 6.5                           # enter country risk-free rate

# --- 1️⃣ Load datasets ---
returns = pd.read_csv(RETURNS_FILE)
betas = pd.read_csv(BETAS_FILE)

# Normalize column names
returns["Symbol"] = returns["Symbol"].astype(str).str.strip().str.upper()
betas["Symbol"] = betas["Symbol"].astype(str).str.strip().str.upper()

# ---  Get the market's (NIFTY50) expected annual return ---
market_return = returns.loc[returns["Symbol"] == "NIFTY50", "Expected Annual Return %"].values
if len(market_return) == 0:
    raise ValueError(" NIFTY50 expected return not found in annual_expected_returns.csv")
market_return = float(market_return[0])
print(f" Market (NIFTY50) Expected Annual Return = {market_return:.2f}%")

# ---  Merge betas with returns ---
merged = pd.merge(betas, returns[["Symbol", "Expected Annual Return %"]], on="Symbol", how="left")

# ---  CAPM expected return ---
merged["CAPM Expected Return %"] = RISK_FREE_RATE + merged["Beta"] * (market_return - RISK_FREE_RATE)

# ---  Alpha (actual minus CAPM expected) ---
merged["Alpha %"] = merged["Expected Annual Return %"] - merged["CAPM Expected Return %"]

# ---  Organize and save ---
cols = [
    "Symbol", "Beta", "Expected Annual Return %", "CAPM Expected Return %",
    "Alpha %", "R²", "Data Points"
]
final = merged[cols]
final = final.sort_values("Alpha %", ascending=False).reset_index(drop=True)

final.to_csv("capm_results.csv", index=False)

print("\n CAPM results saved to capm_results.csv")
print(final)


# ---  Load & clean data ---
FILE = "daily_percent_changes.csv"

df = pd.read_csv(FILE)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
df["Daily % Change"] = pd.to_numeric(df["Daily % Change"], errors="coerce")
df = df.dropna(subset=["Date","Symbol","Daily % Change"])

# Convert % → decimal
df["Daily Return"] = df["Daily % Change"] / 100

# Pivot into matrix (Date × Symbol)
pivot = df.pivot(index="Date", columns="Symbol", values="Daily Return")

# Drop NIFTY50 if you want only stock relationships
if "NIFTY50" in pivot.columns:
    pivot = pivot.drop(columns=["NIFTY50"])

# ---  Compute covariance & correlation ---
cov_daily = pivot.cov()
cov_annual = cov_daily * 252
corr = pivot.corr()

# --- Compute inverse covariance matrix ---
# Handle singularity (if matrix not invertible)
try:
    cov_inv = np.linalg.inv(cov_annual.values)
    cov_inv_df = pd.DataFrame(cov_inv, index=cov_annual.index, columns=cov_annual.columns)
except np.linalg.LinAlgError:
    print(" Covariance matrix is singular — using pseudo-inverse instead.")
    cov_inv = np.linalg.pinv(cov_annual.values)
    cov_inv_df = pd.DataFrame(cov_inv, index=cov_annual.index, columns=cov_annual.columns)

# ---  Save all results ---
cov_annual.to_csv("covariance_matrix_annual.csv")
corr.to_csv("correlation_matrix.csv")
cov_inv_df.to_csv("inverse_covariance_matrix.csv")

print(" Annualized covariance matrix saved as 'covariance_matrix_annual.csv'")
print(" Correlation matrix saved as 'correlation_matrix.csv'")
print(" Inverse covariance matrix saved as 'inverse_covariance_matrix.csv'\n")

print("Covariance matrix (annualized):")
print(cov_annual.round(6), "\n")

print("Inverse covariance matrix:")
print(cov_inv_df.round(4))




# ----------------------------
# CONFIG
# ----------------------------
DAILY_FILE   = "daily_percent_changes.csv"      # Date, Symbol, Daily % Change
ANNUAL_FILE  = "annual_expected_returns.csv"    # Symbol, Expected Annual Return %
EXCLUDE      = {"NIFTY50"}         # exclude from analysis
TRADING_DAYS = 252

# ----------------------------
# LOAD & CLEAN DAILY RETURNS
# ----------------------------
df = pd.read_csv(DAILY_FILE)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
df["Daily % Change"] = pd.to_numeric(df["Daily % Change"], errors="coerce")
df = df.dropna(subset=["Date","Symbol","Daily % Change"])

# Convert % → decimal
df["Daily Return"] = df["Daily % Change"] / 100.0

# Filter stocks (exclude unwanted)
keep_syms = sorted([s for s in df["Symbol"].unique() if s not in EXCLUDE])
pivot = (
    df[df["Symbol"].isin(keep_syms)]
    .pivot(index="Date", columns="Symbol", values="Daily Return")
    .dropna(how="all")
)

# ----------------------------
# CORRELATION & INVERSE CORRELATION
# ----------------------------
corr = pivot.corr(min_periods=2)
try:
    inv_corr_vals = np.linalg.inv(corr.values)
except np.linalg.LinAlgError:
    inv_corr_vals = np.linalg.pinv(corr.values)
inv_corr = pd.DataFrame(inv_corr_vals, index=corr.index, columns=corr.columns)

# ----------------------------
# COVARIANCE (DAILY & ANNUALIZED) & INVERSE COVARIANCE
# ----------------------------
cov_daily = pivot.cov(min_periods=2)
cov_annual = cov_daily * TRADING_DAYS

try:
    inv_cov_vals = np.linalg.inv(cov_annual.values)
except np.linalg.LinAlgError:
    inv_cov_vals = np.linalg.pinv(cov_annual.values)
inv_cov = pd.DataFrame(inv_cov_vals, index=cov_annual.index, columns=cov_annual.columns)

# ----------------------------
# EXPECTED RETURNS VECTOR
# ----------------------------
annual = pd.read_csv(ANNUAL_FILE)
annual["Symbol"] = annual["Symbol"].astype(str).str.strip().str.upper()
annual = annual[annual["Symbol"].isin(keep_syms)].set_index("Symbol").reindex(keep_syms)

if "Expected Annual Return %" not in annual.columns:
    raise ValueError("Column 'Expected Annual Return %' not found in annual_expected_returns.csv")

r = (annual["Expected Annual Return %"].astype(float) / 100.0).values.reshape(-1, 1)

# ----------------------------
# VECTOR MULTIPLICATIONS (BOTH UNIT VECTOR & EXPECTED RETURN)
# ----------------------------
ones = np.ones((len(keep_syms), 1))

A_corr = inv_corr.values
A_cov  = inv_cov.values

# CORRELATION-based
v_corr_ones = A_corr @ ones            # inverse correlation × unit vector
v_corr_rets = A_corr @ r               # inverse correlation × expected return vector

# COVARIANCE-based
v_cov_ones  = A_cov @ ones             # inverse covariance × unit vector
v_cov_rets  = A_cov @ r                # inverse covariance × expected return vector

# Normalization (so each sums to 1)
def _normalize(vec):
    s = vec.sum()
    return vec / s if s != 0 else vec / np.sum(np.abs(vec))

norm_corr_ones = _normalize(v_corr_ones)
norm_corr_rets = _normalize(v_corr_rets)
norm_cov_ones  = _normalize(v_cov_ones)
norm_cov_rets  = _normalize(v_cov_rets)

# ----------------------------
#  COMBINE INTO DATAFRAMES
# ----------------------------
# Raw (non-normalized)
raw_df = pd.DataFrame({
    "Symbol": keep_syms,
    "invCORR * 1":  v_corr_ones.flatten(),
    "invCORR * ER": v_corr_rets.flatten(),
    "invCOV * 1":   v_cov_ones.flatten(),
    "invCOV * ER":  v_cov_rets.flatten(),
})

# Normalized
norm_df = pd.DataFrame({
    "Symbol": keep_syms,
    "Norm(invCORR * 1)":  norm_corr_ones.flatten(),
    "Norm(invCORR * ER)": norm_corr_rets.flatten(),
    "Norm(invCOV * 1)":   norm_cov_ones.flatten(),
    "Norm(invCOV * ER)":  norm_cov_rets.flatten(),
})

# ----------------------------
#  SAVE OUTPUTS
# ----------------------------
corr.to_csv("correlation_matrix_no_ganeshhou.csv")
inv_corr.to_csv("inverse_correlation_matrix_no_ganeshhou.csv")
cov_daily.to_csv("covariance_matrix_daily_no_ganeshhou.csv")
cov_annual.to_csv("covariance_matrix_annual_no_ganeshhou.csv")
inv_cov.to_csv("inverse_covariance_matrix_no_ganeshhou.csv")

raw_df.to_csv("inverse_times_vectors_raw_no_ganeshhou.csv", index=False)
norm_df.to_csv("inverse_times_vectors_normalized_no_ganeshhou.csv", index=False)

# ----------------------------
#  PRINT SUMMARIES
# ----------------------------
print(" Saved files:")
print(" - correlation_matrix_no_ganeshhou.csv")
print(" - inverse_correlation_matrix_no_ganeshhou.csv")
print(" - covariance_matrix_daily_no_ganeshhou.csv")
print(" - covariance_matrix_annual_no_ganeshhou.csv")
print(" - inverse_covariance_matrix_no_ganeshhou.csv")
print(" - inverse_times_vectors_raw_no_ganeshhou.csv")
print(" - inverse_times_vectors_normalized_no_ganeshhou.csv\n")

print("Correlation Matrix (rounded):\n", corr.round(3), "\n")
print("Inverse Correlation Matrix (rounded):\n", inv_corr.round(3), "\n")
print("Annualized Covariance Matrix (rounded):\n", cov_annual.round(6), "\n")
print("Inverse Covariance Matrix (rounded):\n", inv_cov.round(6), "\n")

print("Raw Multiplications (non-normalized):\n", raw_df.round(6), "\n")
print("Normalized Results (sum to 1):\n", norm_df.round(6), "\n")

# sanity check
print("Sum checks (should be close to 1):")
print(" Norm(invCORR * 1):", float(norm_corr_ones.sum()))
print(" Norm(invCORR * ER):", float(norm_corr_rets.sum()))
print(" Norm(invCOV * 1):", float(norm_cov_ones.sum()))
print(" Norm(invCOV * ER):", float(norm_cov_rets.sum()))

# ----------------------------
#COMPUTE (1,1,1,1) * INVERSE COVARIANCE MATRIX
# ----------------------------
ones_row = np.ones((1, len(keep_syms)))  # row vector [1,1,1,1]
inv_cov_row_mult = ones_row @ inv_cov.values  # (1×N) x (N×N) = 1×N

# Make it a DataFrame
inv_cov_row_df = pd.DataFrame({
    "Symbol": keep_syms,
    "(1,1,1,1) * invCOV": inv_cov_row_mult.flatten()
})

# Optional normalized version (sum to 1)
inv_cov_row_df["Normalized (sum=1)"] = inv_cov_row_df["(1,1,1,1) * invCOV"] / inv_cov_row_df["(1,1,1,1) * invCOV"].sum()

# Save
inv_cov_row_df.to_csv("unit_row_times_inverse_covariance.csv", index=False)

print("\n Added: (1,1,1,1) × Inverse Covariance Matrix")
print(inv_cov_row_df.round(6))

# ----------------------------
# COMPUTE ALPHA (α)
# ----------------------------
# Alpha = e^T C^{-1} e
e = np.ones((len(keep_syms), 1))  # column vector of ones
C_inv = inv_cov.values            # inverse covariance matrix

alpha = (e.T @ C_inv @ e).item()

print("\n Alpha (α) Computed Successfully")
print(f"α (eᵀ·C⁻¹·e) = {alpha:.6f}")
# ============================
# 11) Markowitz scalars: β, γ, Δ
# ============================
# Shapes:
# e : (N×1)   ones column
# r : (N×1)   expected annual returns (decimals)
# C_inv : (N×N) inverse covariance (annualized)

beta_m = (e.T @ C_inv @ r).item()          # β = eᵀ C⁻¹ r
gamma_m = (r.T @ C_inv @ r).item()         # γ = rᵀ C⁻¹ r
delta_m = alpha * gamma_m - beta_m**2      # Δ = αγ − β²

print("\n Markowitz scalars")
print(f"α = {alpha:.6f}")
print(f"β = {beta_m:.6f}")
print(f"γ = {gamma_m:.6f}")
print(f"Δ = {delta_m:.6f}")

# ============================
# 12) Tangency portfolio (with risk-free rate)
# ============================
RF = 0.065  # 6.5% risk-free
# Tangency portfolio expected return (scalar):
r_tan = (gamma_m - beta_m*RF) / (beta_m - alpha*RF)

# Tangency portfolio risky-asset weights (sum to 1):
# w_tan ∝ C⁻¹ (r − Rf·e)
k_vec = r - RF * e
w_tan = (C_inv @ k_vec)
w_tan = w_tan / w_tan.sum()                # normalize to sum to 1

# Minimum-variance portfolio weights (sum to 1):
w_mv = (C_inv @ e) / (e.T @ C_inv @ e)

# Portfolio metrics
w_tan_vec = w_tan.flatten()
w_mv_vec  = w_mv.flatten()

# expected returns
ER_tan = float((w_tan.T @ r).item())
ER_mv  = float((w_mv.T  @ r).item())

# vols: sqrt(w' Σ w)   (use annualized Σ)
Sigma = cov_annual.values
vol_tan = float(np.sqrt(w_tan_vec @ Sigma @ w_tan_vec))
vol_mv  = float(np.sqrt(w_mv_vec  @ Sigma @ w_mv_vec))

# Sharpe (excess over RF)
sharpe_tan = (ER_tan - RF) / vol_tan
sharpe_mv  = (ER_mv  - RF) / vol_mv

print("\n Tangency portfolio (risky assets only)")
print(f"E[R_tan]  = {ER_tan*100:.2f}%")
print(f"Vol_tan   = {vol_tan*100:.2f}%")
print(f"Sharpe_tan= {sharpe_tan:.3f}")
print("\n Minimum-variance portfolio")
print(f"E[R_mv]   = {ER_mv*100:.2f}%")
print(f"Vol_mv    = {vol_mv*100:.2f}%")
print(f"Sharpe_mv = {sharpe_mv:.3f}")

# Show weights with symbols
tan_weights = pd.DataFrame({"Symbol": keep_syms, "w_tan": w_tan_vec})
mv_weights  = pd.DataFrame({"Symbol": keep_syms, "w_minvar": w_mv_vec})

display_cols = ["Symbol", "w_tan"]
print("\nTangency weights (sum≈1):\n", tan_weights.round(6).to_string(index=False))
print("\nMin-variance weights (sum≈1):\n", mv_weights.round(6).to_string(index=False))

# ============================
# Lagrange multipliers for a target return R*
# ============================
# For any target return R_star, the frontier multipliers are:
# λ = (γ − β R*) / Δ,   μ = (α R* − β) / Δ
R_star = ER_tan  # you can set any target; using tangency return by default
lambda_star = (gamma_m - beta_m * R_star) / delta_m
mu_star     = (alpha * R_star - beta_m) / delta_m
print(f"\nλ(R*) = {lambda_star:.6f},  μ(R*) = {mu_star:.6f}  (with R* = {R_star*100:.2f}%)")

import matplotlib.pyplot as plt

# ----------------------------
# Plot Efficient Frontier
# ----------------------------
# Define a range of expected returns (between min and max of your assets)
R_min, R_max = r.min(), r.max()
R_range = np.linspace(R_min, R_max, 100)

# Efficient frontier (portfolio volatility)
sigma_range = np.sqrt((alpha * R_range**2 - 2 * beta_m * R_range + gamma_m) / delta_m)

# Tangency & min variance points
tan_point = (vol_tan, ER_tan)
mv_point  = (vol_mv, ER_mv)

# Plot
plt.figure(figsize=(9,6))
plt.plot(sigma_range * 100, R_range * 100, label='Efficient Frontier', color='navy', linewidth=2)
plt.scatter(vol_tan * 100, ER_tan * 100, color='orange', marker='*', s=200, label='Tangency Portfolio')
plt.scatter(vol_mv * 100, ER_mv * 100, color='green', marker='o', s=100, label='Min-Variance Portfolio')

# Add RF line
plt.axhline(y=RF*100, color='gray', linestyle='--', linewidth=1)
plt.text(0, RF*100 + 0.2, f"Risk-Free ({RF*100:.1f}%)", color='gray')

# Labels
plt.title('Efficient Frontier (Markowitz Portfolio Theory)', fontsize=14, pad=15)
plt.xlabel('Portfolio Volatility (%)', fontsize=12)
plt.ylabel('Expected Annual Return (%)', fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# ----------------------------
# Add Capital Market Line (CML)
# ----------------------------
# Capital Market Line equation
sigma_cml = np.linspace(0, max(sigma_range)*1.1, 100)
R_cml = RF + ((ER_tan - RF) / vol_tan) * sigma_cml

# Plot again (overlay on the existing figure)
plt.figure(figsize=(9,6))
plt.plot(sigma_range * 100, R_range * 100, label='Efficient Frontier', color='navy', linewidth=2)
plt.plot(sigma_cml * 100, R_cml * 100, label='Capital Market Line (CML)', color='red', linestyle='--', linewidth=2)

# Mark tangency portfolio
plt.scatter(vol_tan * 100, ER_tan * 100, color='orange', marker='*', s=200, label='Tangency Portfolio')
plt.scatter(vol_mv * 100, ER_mv * 100, color='green', marker='o', s=100, label='Min-Variance Portfolio')

# Add labels and line for risk-free rate
plt.axhline(y=RF*100, color='gray', linestyle='--', linewidth=1)
plt.text(0, RF*100 + 0.3, f"Risk-Free ({RF*100:.1f}%)", color='gray')

# Titles and labels
plt.title('Efficient Frontier & Capital Market Line', fontsize=14, pad=15)
plt.xlabel('Portfolio Volatility (%)', fontsize=12)
plt.ylabel('Expected Annual Return (%)', fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ----------------------------
# μ (mu) and portfolio weights
# ----------------------------

# 1. h and g vectors
h_vec = C_inv @ e     # h = C⁻¹ e
g_vec = C_inv @ r     # g = C⁻¹ r

# 2. Choose a target expected return (you can change this)
R_target = ER_tan  # example: use tangency portfolio expected return

# 3. Compute lambda and mu using Markowitz formulas
lambda_m = (gamma_m - beta_m * R_target) / delta_m
mu_m = (alpha * R_target - beta_m) / delta_m

print(f"\n Lagrange multipliers:")
print(f"λ = {lambda_m:.6f}")
print(f"μ = {mu_m:.6f}")

# 4. Compute portfolio weights
w_star = lambda_m * h_vec + mu_m * g_vec
w_star = w_star / w_star.sum()  # normalize weights to sum to 1

# 5. Show results
portfolio_df = pd.DataFrame({
    "Symbol": keep_syms,
    "Weight (w*)": w_star.flatten()
})

print("\nOptimal Portfolio Weights (w* = λh + μg):")
print(portfolio_df.round(6).to_string(index=False))

# Optional: save to file
portfolio_df.to_csv("optimal_portfolio_weights.csv", index=False)




# --- Market Cap Weighted Portfolio ---
#input your stocks and their market caps here
market_caps = {
    "a": 1,
    "b": 1,
    "c": 1,
    "d": 1
}

# total and normalized market cap weights
total_cap = sum(market_caps.values())
mcap_weights = {k: v / total_cap for k, v in market_caps.items()}

# --- Convert both to DataFrames for comparison ---
w_calc = pd.DataFrame({
    "Symbol": keep_syms,
    "Calculated Weight": w_star.flatten()
})

w_mcap = pd.DataFrame({
    "Symbol": list(mcap_weights.keys()),
    "Market Cap Weight": list(mcap_weights.values())
})

# --- Merge and compute difference ---
compare_df = pd.merge(w_calc, w_mcap, on="Symbol", how="inner")
compare_df["Difference"] = compare_df["Calculated Weight"] - compare_df["Market Cap Weight"]

# --- Display and save ---
print("\nComparison: Calculated vs Market Cap Weights")
print(compare_df.round(6).to_string(index=False))

compare_df.to_csv("weight_difference_comparison.csv", index=False)
print("\n Saved as 'weight_difference_comparison.csv'")

from scipy.optimize import minimize
import numpy as np
import pandas as pd

# --- constants from your model ---
C_inv = inv_cov.values
e = np.ones((len(keep_syms), 1))
mcap_w = np.array([mcap_weights[s] for s in keep_syms]).reshape(-1, 1)
alpha = (e.T @ C_inv @ e).item()
beta_m = (e.T @ C_inv @ r).item()
gamma_m = (r.T @ C_inv @ r).item()
delta_m = alpha * gamma_m - beta_m**2

# --- objective function: minimize difference from market cap weights ---
def objective(r_vec_flat):
    r_vec = np.array(r_vec_flat).reshape(-1, 1)

    # recompute beta, gamma, delta for new returns
    beta_ = (e.T @ C_inv @ r_vec).item()
    gamma_ = (r_vec.T @ C_inv @ r_vec).item()
    delta_ = alpha * gamma_ - beta_**2

    # pick target return = weighted average of r_vec
    R_target = float((mcap_w.T @ r_vec).item())

    # compute lambda, mu
    lam = (gamma_ - beta_ * R_target) / delta_
    mu  = (alpha * R_target - beta_) / delta_

    # compute optimal weights
    w_star = lam * (C_inv @ e) + mu * (C_inv @ r_vec)
    w_star = w_star / w_star.sum()

    # objective = sum of squared differences
    diff = w_star - mcap_w
    return float(np.sum(diff**2))

# --- initial guess: your current expected returns ---
x0 = r.flatten()

# --- run optimization ---
result = minimize(objective, x0, method='BFGS', options={'disp': True})
r_opt = result.x.reshape(-1, 1)

# --- recompute final weights ---
beta_opt = (e.T @ C_inv @ r_opt).item()
gamma_opt = (r_opt.T @ C_inv @ r_opt).item()
delta_opt = alpha * gamma_opt - beta_opt**2
R_target_opt = float((mcap_w.T @ r_opt).item())
lam_opt = (gamma_opt - beta_opt * R_target_opt) / delta_opt
mu_opt  = (alpha * R_target_opt - beta_opt) / delta_opt
w_opt = lam_opt * (C_inv @ e) + mu_opt * (C_inv @ r_opt)
w_opt = w_opt / w_opt.sum()

# --- show results ---
final_df = pd.DataFrame({
    "Symbol": keep_syms,
    "Market Cap Weight": mcap_w.flatten(),
    "Optimized Weight": w_opt.flatten(),
    "Difference": (w_opt - mcap_w).flatten(),
    "Original ER": r.flatten(),
    "Optimized ER": r_opt.flatten()
})

print("\n Reverse Optimization Results (Expected Returns adjusted to match Market Cap Weights):")
print(final_df.round(6).to_string(index=False))



from scipy.optimize import minimize

# ----------------------------
# Inputs you edit: your view *deltas* in percentage points
# (i.e., +1.5 means "I think this stock will earn 1.5% more than the implied return")
# ----------------------------
view_delta_pp = {
    "a":  0.0,
    "b": -10.0,
    "c":  3.0,
    "d": 2.0,
} 

# ----------------------------
# Config / files
# ----------------------------
TRADING_DAYS = 252
EXCLUDE = {"GANESHHOU", "NIFTY50"}
COV_FILE = "covariance_matrix_annual_no_ganeshhou.csv"
ANNUAL_FILE = "annual_expected_returns.csv"          # used as initial guess for optimizer
RF = 0.065

# Your market caps (₹ Cr) for market-portfolio target
mcap_dict = {
    "HINDCOPPER": 33448,
    "WAAREEENER": 108086,
    "IEX":        12407,
    "FORTIS":     82887,
}

# ----------------------------
# Load covariance and align symbols
# ----------------------------
cov_annual = pd.read_csv(COV_FILE, index_col=0)
symbols = cov_annual.columns.str.upper().tolist()
Sigma = cov_annual.values
C_inv = np.linalg.pinv(Sigma)  # stable inverse

# Build market-cap weights in this symbol order
mcap_w = np.array([mcap_dict[s] for s in symbols], dtype=float)
mcap_w = (mcap_w / mcap_w.sum()).reshape(-1, 1)

# Ones column
e = np.ones((len(symbols), 1))

# ----------------------------
# Helper: Markowitz weights for a given expected-returns vector (n×1, decimals)
# Using the standard unconstrained mean-variance: w ∝ Σ^{-1} μ
# ----------------------------
def weights_from_mu(Sigma, mu):
    invS = np.linalg.pinv(Sigma)
    w_raw = invS @ mu
    w = w_raw / w_raw.sum()
    return w

# ----------------------------
# Reverse-optimize to get IMPLIED (market-consistent) returns if not present:
# We choose μ to minimize || w(μ) - w_mkt ||^2
# ----------------------------
def implied_returns_from_mkt(Sigma, w_target, r_init):
    invS = np.linalg.pinv(Sigma)
    evec = np.ones((Sigma.shape[0], 1))

    def objective(mu_flat):
        mu = mu_flat.reshape(-1,1)
        w = weights_from_mu(Sigma, mu)
        return float(np.sum((w - w_target)**2))

    res = minimize(objective, r_init.flatten(), method="BFGS", options={"maxiter": 200})
    mu_star = res.x.reshape(-1,1)
    return mu_star

# ----------------------------
# Get an initial guess for returns (decimals) for the optimizer
# ----------------------------
ann = pd.read_csv(ANNUAL_FILE)
ann["Symbol"] = ann["Symbol"].str.strip().str.upper()
ann = ann[ann["Symbol"].isin(symbols)].set_index("Symbol").reindex(symbols)
r_init = (ann["Expected Annual Return %"].astype(float).values.reshape(-1,1)) / 100.0

# ----------------------------
# Establish baseline implied returns μ_implied (market-consistent)
# If you already computed r_opt earlier in your notebook, you can reuse it.
# Here we compute (or recompute) them robustly.
# ----------------------------
mu_implied = implied_returns_from_mkt(Sigma, mcap_w, r_init)

# ----------------------------
# Apply your view deltas (percentage points) to the implied returns
# New μ = μ_implied + delta
# ----------------------------
delta_vec = np.array([view_delta_pp.get(s, 0.0) for s in symbols], dtype=float).reshape(-1,1) / 100.0
mu_new = mu_implied + delta_vec

# ----------------------------
# Recompute optimal weights with the *new* expected returns
# ----------------------------
w_new = weights_from_mu(Sigma, mu_new)
w_new = w_new.reshape(-1,1)

# Portfolio stats
ER_new  = float((w_new.T @ mu_new).item())                   # expected return
VOL_new = float(np.sqrt((w_new.flatten() @ Sigma @ w_new.flatten())))  # vol (annual)
Sharpe  = (ER_new - RF) / VOL_new if VOL_new > 0 else np.nan

# ----------------------------
# Output & save
# ----------------------------
out = pd.DataFrame({
    "Symbol": symbols,
    "Implied μ (dec)": mu_implied.flatten(),
    "View Δ (pp)":     (delta_vec*100).flatten(),
    "New μ (dec)":     mu_new.flatten(),
    "MktCap w":        mcap_w.flatten(),
    "New Weight":      w_new.flatten(),
    "Diff (New − Mkt)": (w_new - mcap_w).flatten()
})

print("=== Black–Litterman: Non-Interactive Sensitivity (via view deltas) ===")
print(out.round(6).to_string(index=False))

print(f"\nPortfolio (with new μ):  E[R]={ER_new*100:.2f}%,  Vol={VOL_new*100:.2f}%,  Sharpe (Rf={RF*100:.1f}%): {Sharpe:.3f}")

out.to_csv("bl_sensitivity_new_weights.csv", index=False)
print("\n Saved: bl_sensitivity_new_weights.csv")

# ==== Compare BL-adjusted portfolio vs Tangency on the Efficient Frontier ====

# --- sanity: need Sigma (annual cov), C_inv, e (ones), mu_new (decimals), w_new (sum=1), RF ---
assert 'Sigma' in globals() and isinstance(Sigma, np.ndarray), "Sigma (annual covariance matrix) not found."
assert 'C_inv' in globals() and isinstance(C_inv, np.ndarray), "C_inv (inverse covariance) not found."
assert 'e' in globals(), "e (ones column) not found."
assert 'mu_new' in globals(), "mu_new (new expected returns) not found."
assert 'w_new' in globals(), "w_new (weights for new μ) not found."
assert 'RF' in globals(), "RF (risk-free) not set."

# --- Markowitz scalars for current (BL-adjusted) returns μ = mu_new ---
alpha_bl = (e.T @ C_inv @ e).item()
beta_bl  = (e.T @ C_inv @ mu_new).item()
gamma_bl = (mu_new.T @ C_inv @ mu_new).item()
delta_bl = alpha_bl * gamma_bl - beta_bl**2

# --- Efficient frontier parametric curve: σ(R) for a grid of returns ---
R_min, R_max = float(mu_new.min()), float(mu_new.max())
R_grid = np.linspace(R_min, R_max, 200)
sigma_grid = np.sqrt((alpha_bl * R_grid**2 - 2*beta_bl * R_grid + gamma_bl) / max(delta_bl, 1e-12))

# --- Tangency portfolio for current μ: w ∝ Σ⁻¹(μ − Rf·e) ---
k_vec = mu_new - RF * e
w_tan = (C_inv @ k_vec)
w_tan = w_tan / w_tan.sum()
ER_tan = float((w_tan.T @ mu_new).item())
VOL_tan = float(np.sqrt(w_tan.flatten() @ Sigma @ w_tan.flatten()))
Sharpe_tan = (ER_tan - RF) / VOL_tan if VOL_tan > 0 else np.nan

# --- Your BL-adjusted portfolio point (already computed as w_new) ---
ER_new  = float((w_new.T @ mu_new).item())
VOL_new = float(np.sqrt(w_new.flatten() @ Sigma @ w_new.flatten()))
Sharpe_new = (ER_new - RF) / VOL_new if VOL_new > 0 else np.nan

# --- Minimum-variance portfolio (optional) ---
w_mv = (C_inv @ e) / (e.T @ C_inv @ e)
ER_mv = float((w_mv.T @ mu_new).item())
VOL_mv = float(np.sqrt(w_mv.flatten() @ Sigma @ w_mv.flatten()))

# --- Plot (single chart) ---
plt.figure(figsize=(9,6))
plt.plot(sigma_grid*100, R_grid*100, label="Efficient Frontier", linewidth=2)
plt.scatter(VOL_tan*100, ER_tan*100, marker='*', s=200, label="Tangency (BL μ)")
plt.scatter(VOL_new*100, ER_new*100, marker='o', s=120, label="Your BL Portfolio")
plt.scatter(VOL_mv*100,  ER_mv*100,  marker='s', s=100, label="Min-Variance")

# Risk-free line for reference
plt.axhline(y=RF*100, linestyle='--', linewidth=1, label=f"Risk-Free ({RF*100:.1f}%)")

plt.title("Efficient Frontier vs Your BL Portfolio and Tangency")
plt.xlabel("Volatility (%)")
plt.ylabel("Expected Annual Return (%)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# --- Print a compact comparison ---
print("=== Comparison ===")
print(f"Tangency (BL μ):   ER={ER_tan*100:.2f}%,  Vol={VOL_tan*100:.2f}%,  Sharpe={Sharpe_tan:.3f}")
print(f"Your BL portfolio: ER={ER_new*100:.2f}%,  Vol={VOL_new*100:.2f}%,  Sharpe={Sharpe_new:.3f}")
print(f"Min-Variance:      ER={ER_mv*100:.2f}%,  Vol={VOL_mv*100:.2f}%")

# --- Compare weights: Tangency vs Your BL vs Minimum-Variance (no saving) ---
weights_compare = pd.DataFrame({
    "Symbol": [s for s in symbols],
    "Tangency w": w_tan.flatten(),
    "Your BL w":  w_new.flatten(),
    "Min-Var w":  w_mv.flatten()
})

# Normalize to ensure clean comparison
weights_compare[["Tangency w","Your BL w","Min-Var w"]] = (
    weights_compare[["Tangency w","Your BL w","Min-Var w"]]
    .div(weights_compare[["Tangency w","Your BL w","Min-Var w"]].sum(axis=0), axis=1)
)

print("=== Portfolio Weights Comparison ===")
print(weights_compare.round(6).to_string(index=False))












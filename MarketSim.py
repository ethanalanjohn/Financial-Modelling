import pandas as pd
import numpy as np
from datetime import datetime
import os

DATA_FILE = ""   # File with data set of all stocks in the concerned stock exchange. I put mine together using yahoo finances API
DATE_COL = "Date"
ADJ_COL = "Adj Close"    # fallback to "Close" if not present
N_PATHS = 200
HORIZON = 252
SEED = 42

# ---------------- 1) Load & pivot ----------------
df_raw = pd.read_csv(DATA_FILE, parse_dates=[DATE_COL])
if ADJ_COL not in df_raw.columns:
    if "Close" in df_raw.columns:
        ADJ_COL = "Close"
    else:
        raise ValueError(f"Neither '{ADJ_COL}' nor 'Close' found in {DATA_FILE}")

df_prices = df_raw.pivot(index=DATE_COL, columns="Symbol", values=ADJ_COL).sort_index()
# normalize index to date-level (midnight) to avoid tz mismatch
df_prices.index = pd.DatetimeIndex(df_prices.index).normalize()
df_prices = df_prices.ffill().bfill()
returns = df_prices.pct_change().dropna(how="all")
print("Loaded price panel:", df_prices.shape, "returns:", returns.shape)

# ---------------- 2) Factor model pieces ----------------
fm = FactorModel()
clusters = fm.build_sector_clusters(returns, n_clusters=8, random_state=SEED)
print("Clusters counts (top 8):\n", clusters.value_counts().head(8))

factor_returns = fm.compute_factor_returns(returns, market_returns=None)
print("Factor names:", factor_returns.columns.tolist())

# estimate betas (use last 500 factor rows if available)
betas = fm.estimate_betas(returns, factor_returns, window=500)
print("betas shape:", getattr(betas, "shape", None))

# compute residuals aligned to factor_returns
residuals = fm.compute_residuals(returns, factor_returns, betas=betas)
print("residuals shape:", getattr(residuals, "shape", None))

# ---------------- 3) Fit GARCH engine ----------------
ge = None
try:
    ge = GarchEngine(dist='normal')  # your class
    params = ge.fit_all(factor_returns)   # fit each factor
    print("GARCH fit: sample factors params keys:", list(params.keys())[:5])
except Exception as e:
    print("Warning: GARCH fit failed or not available:", e)
    ge = None

# ---------------- 4) Fit Copula engine (Phase-B2) ----------------
ce = None
try:
    ce = CopulaEngine(shrinkage=True, winsorize_z=8.0)
    corr = ce.fit(residuals, min_obs=30)
    print("Copula fit OK. corr shape:", corr.shape)
except Exception as e:
    print("Warning: Copula fit failed:", e)
    ce = None

# ---------------- 5) Build simulator and run ----------------
rm = RegimeModel()  # optional/placeholder
sim = PortfolioSimulator(factor_model=fm, regime_model=rm, garch_engine=ge, copula_engine=ce)

# equal weights across all available symbols in the returns panel
symbols = list(returns.columns)
if len(symbols) == 0:
    raise ValueError("No symbols found in returns panel.")
weights = {s: 1.0/len(symbols) for s in symbols}

# align betas for simulator (ensure factors match)
if betas is not None and set(betas.columns) >= set(factor_returns.columns):
    sim_betas = betas.reindex(columns=factor_returns.columns).fillna(0.0)
else:
    sim_betas = pd.DataFrame(np.ones((len(symbols), len(factor_returns.columns))),
                             index=symbols, columns=factor_returns.columns)

alphas = pd.Series(0.0, index=sim_betas.index)

# Try to run the engine-driven simulation. PortfolioSimulator may accept different return shapes;
# handle common variants robustly.
try:
    # Many implementations return either (final_vals, stats) or (paths, final_vals, stats).
    out = sim.simulate_portfolio(
        weights=weights,
        betas=sim_betas,
        alphas=alphas,
        n_paths=N_PATHS,
        horizon=HORIZON,
        regime_series=None,
        seed=SEED,
        verbose=True
    )
    # normalize possible outputs
    if isinstance(out, tuple) and len(out) == 3:
        portfolio_paths, final_returns, stats = out
    elif isinstance(out, tuple) and len(out) == 2:
        final_returns, stats = out
        portfolio_paths = None
    elif isinstance(out, dict):
        # some implementations may return dict
        final_returns = np.asarray(out.get('final_returns') or out.get('final_vals') or out.get('returns'))
        stats = out.get('stats', {})
        portfolio_paths = out.get('paths', None)
    else:
        # unexpected: try to coerce
        try:
            final_returns = np.asarray(out)
            stats = {}
            portfolio_paths = None
        except Exception:
            raise RuntimeError("Simulator returned unexpected structure. Inspect sim.simulate_portfolio output.")
except Exception as e_sim:
    # fallback: try to call a simpler simulate(...) API, or fallback Phase-A
    print("Engine-driven simulation failed:", e_sim)
    try:
        final_returns = sim.simulate(weights=weights, n_paths=N_PATHS, horizon=HORIZON)
        stats = {
            "mean": float(np.nanmean(final_returns)),
            "std": float(np.nanstd(final_returns, ddof=1)),
            "sharpe": float(np.nanmean(final_returns) / np.nanstd(final_returns, ddof=1)) if np.nanstd(final_returns, ddof=1) > 0 else np.nan
        }
        portfolio_paths = None
        print("Fallback simple simulate(...) succeeded.")
    except Exception as e2:
        raise RuntimeError(f"Simulator calls failed. Engine error: {e_sim}; fallback error: {e2}")

# ---------------- 6) Print & save summary ----------------
print("\nSIM STATS:", stats)
# ensure final_returns is numpy array
final_returns = np.asarray(final_returns, dtype=float)
print("Sample of simulated final returns (first 10):", final_returns[:10])
print("Mean of simulated final returns:", np.mean(final_returns))
print("Std of simulated final returns:", np.std(final_returns, ddof=1))

# optional: save small summary row to results CSV
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
results_path = os.path.join(RESULTS_DIR, "simulation_summary.csv")
row = {
    "timestamp": datetime.now().isoformat(),
    "data_file": DATA_FILE,
    "n_symbols": len(symbols),
    "n_paths": int(N_PATHS),
    "horizon": int(HORIZON),
    "mean_sim": float(stats.get("mean", np.nan)),
    "std_sim": float(stats.get("std", np.nan)),
    "sharpe_sim": float(stats.get("sharpe", np.nan))
}
import csv
write_header = not os.path.exists(results_path)
with open(results_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)

print("Summary appended to:", results_path)






# ========== Visualization dashboard for simulation outputs ==========
# Paste and run after running the simulation. Produces many plots and saves them to results/figs/.


import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

OUT_DIR = "results/figs"
os.makedirs(OUT_DIR, exist_ok=True)

# Helper checks
have_portfolio = 'portfolio_paths' in globals() or 'port_daily' in globals()
# prefer portfolio_paths variable name used in latest code
portfolio_paths = globals().get('portfolio_paths', globals().get('port_daily', None))
final_returns = globals().get('final_returns', globals().get('final_vals', globals().get('final_vals', None)))
stats = globals().get('stats', globals().get('sim_stats', None))
factor_returns = globals().get('factor_returns', globals().get('fm', None))
residuals = globals().get('residuals', globals().get('fm_residuals', globals().get('residuals', None)))
betas = globals().get('betas', None)
copula_engine = globals().get('ce', globals().get('copula_engine', None))
garch_engine = globals().get('ge', globals().get('garch_engine', None))
returns = globals().get('returns', None)
df_prices = globals().get('df_prices', None)

# Some fallbacks / shape normalizers
if portfolio_paths is not None:
    portfolio_paths = np.asarray(portfolio_paths)
    # If shape is (n_paths, horizon, n_symbols) try to reduce to (n_paths, horizon)
    if portfolio_paths.ndim == 3:
        # already aggregated? If third axis present, assume it's per-symbol returns and we don't know weights here
        # Try to detect if it's aggregated by values being near zero mean; else aggregate by mean across symbols
        portfolio_paths = portfolio_paths.sum(axis=2) if True else portfolio_paths.mean(axis=2)
elif 'port_daily' in globals():
    portfolio_paths = globals()['port_daily']

if final_returns is None and portfolio_paths is not None:
    final_returns = np.prod(1.0 + portfolio_paths, axis=1) - 1.0

# Utility: compute per-path Sharpe using daily returns
def per_path_sharpe(port_daily, rf_annual=0.06):
    rf_daily = rf_annual / 252.0
    means = np.nanmean(port_daily, axis=1)
    sds = np.nanstd(port_daily, axis=1, ddof=1)
    sharpe = (means - rf_daily) / sds
    return sharpe

# Utility: per-path max drawdown
def per_path_mdd(port_daily):
    n_paths, horizon = port_daily.shape
    mdds = np.zeros(n_paths)
    for i in range(n_paths):
        cum = np.cumprod(1.0 + port_daily[i])
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        mdds[i] = np.nanmin(dd)
    return mdds

# ---------- 1) Histogram of final returns ----------
if final_returns is not None:
    plt.figure(figsize=(8,4))
    plt.hist(final_returns, bins=50, edgecolor='k', alpha=0.8)
    plt.axvline(np.nanmean(final_returns), color='k', linestyle='--', label=f"mean {np.nanmean(final_returns):.2%}")
    plt.title("Histogram of simulated final returns (per-path)")
    plt.xlabel("Total return over horizon")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "hist_final_returns.png"))
    plt.show()

# ---------- 2) Empirical CDF & percentiles ----------
if final_returns is not None:
    arr = np.sort(final_returns)
    p = np.linspace(0,100,len(arr))
    plt.figure(figsize=(6,4))
    plt.plot(p, arr, lw=1.5)
    for q in [1,5,10,25,50,75,90,95,99]:
        val = np.nanpercentile(arr, q)
        plt.scatter(q, val, s=20)
        plt.text(q, val, f"{q}p:{val:.2%}", fontsize=8, va='bottom')
    plt.xlabel("Percentile")
    plt.ylabel("Final return")
    plt.title("Empirical percentiles of simulated final returns")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cdf_final_returns.png"))
    plt.show()

# ---------- 3) Sample simulated portfolio paths (random 20) ----------
if portfolio_paths is not None:
    n_paths, horizon = portfolio_paths.shape
    n_plot = min(30, n_paths)
    idx = np.random.RandomState(0).choice(n_paths, size=n_plot, replace=False)
    plt.figure(figsize=(10,5))
    for i in idx:
        plt.plot(np.cumprod(1+portfolio_paths[i,:]), alpha=0.6)
    plt.title(f"Sample of {n_plot} simulated cumulative portfolio paths")
    plt.xlabel("Day")
    plt.ylabel("Cumulative return (1+x)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sample_portfolio_paths.png"))
    plt.show()

# ---------- 4) Per-path stats distributions: Sharpe & MDD ----------
if portfolio_paths is not None:
    sharpe_arr = per_path_sharpe(portfolio_paths)
    mdd_arr = per_path_mdd(portfolio_paths)
    fig, ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].hist(sharpe_arr[np.isfinite(sharpe_arr)], bins=40, edgecolor='k')
    ax[0].set_title("Per-path Sharpe distribution")
    ax[0].set_xlabel("Sharpe (daily series -> unannualized)")
    ax[1].hist(mdd_arr, bins=40, edgecolor='k')
    ax[1].set_title("Per-path max drawdown distribution")
    ax[1].set_xlabel("Max drawdown (fraction negative)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "per_path_sharpe_mdd.png"))
    plt.show()

# ---------- 5) Compare mean simulated daily vol vs empirical vol ----------
if portfolio_paths is not None and returns is not None:
    # portfolio simulated daily series mean across paths
    avg_sim_daily = np.nanmean(portfolio_paths, axis=0)
    sim_vol = np.nanstd(portfolio_paths, axis=0, ddof=1)
    # build realized portfolio from equal-weighted current returns if possible
    common_symbols = returns.columns
    if len(common_symbols) > 0:
        equal_w = np.ones(len(common_symbols))/len(common_symbols)
        realized_port = returns.fillna(0).values.dot(equal_w)
        real_vol_rolling = pd.Series(realized_port).rolling(21).std().to_numpy()  # 1-month rolling
        # Plot
        plt.figure(figsize=(10,4))
        plt.plot(sim_vol, label="sim daily vol (per day)", alpha=0.7)
        plt.plot(pd.Series(real_vol_rolling).fillna(method='bfill'), label="realized port 21d rolling vol", alpha=0.7)
        plt.title("Simulated vs realized portfolio volatility (daily)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "sim_vs_real_vol.png"))
        plt.show()

# ---------- 6) Factor diagnostics (if factor_returns exists) ----------
if isinstance(factor_returns, pd.DataFrame):
    fr = factor_returns.copy()
    # plot correlation heatmap (small sample if many factors)
    fac = fr.columns.tolist()
    if len(fac) > 0:
        corr = fr.corr()
        plt.figure(figsize=(6,5))
        plt.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Factor correlation matrix")
        xt = range(len(fac))
        plt.xticks(xt, fac, rotation=45, ha='right', fontsize=8)
        plt.yticks(xt, fac, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "factor_corr.png"))
        plt.show()
    # EWMA vol tail
    try:
        ewma = fr.apply(lambda s: s.ewm(span=60, adjust=False).std()).tail(20)
        plt.figure(figsize=(10,4))
        for c in ewma.columns:
            plt.plot(ewma.index, ewma[c], alpha=0.6)
        plt.title("Factor EWMA vol (tail)")
        plt.xlabel("Date")
        plt.ylabel("EWMA sigma")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "factor_ewma_tail.png"))
        plt.show()
    except Exception:
        pass

# ---------- 7) Residuals diagnostics & Copula checks ----------
if isinstance(residuals, pd.DataFrame):
    R = residuals.copy()
    # correlation heatmap sample (first 40 symbols)
    sample_cols = R.columns[:40]
    corr_r = R[sample_cols].corr()
    plt.figure(figsize=(8,6))
    plt.imshow(corr_r, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Residuals correlation heatmap (first 40 symbols)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "residuals_corr_sample.png"))
    plt.show()

    # QQ-plot for a few sample assets to check tails
    import math
    def qq_plot(series, ax=None, npt=200):
        s = series.dropna().values
        s = s[np.isfinite(s)]
        if s.size < 10:
            return
        s = np.sort(s)
        p = (np.arange(1, len(s)+1)-0.5)/len(s)
        q = np.quantile(np.random.normal(size=100000), p)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(q, s, marker='.', linestyle='none')
        ax.plot([q.min(), q.max()], [q.min(), q.max()], 'r--')
    plt.figure(figsize=(10,6))
    cols = list(R.columns[:6])
    for i, c in enumerate(cols):
        ax = plt.subplot(2,3,i+1)
        qq_plot(R[c], ax=ax)
        ax.set_title(c)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "residuals_qq_sample.png"))
    plt.show()

# ---------- 8) Copula correlation matrix (if available) ----------
if copula_engine is not None and getattr(copula_engine, "corr_", None) is not None:
    corr_df = copula_engine.corr_
    # show condition number & top pairs
    try:
        vals = np.linalg.eigvalsh(corr_df.values)
        cond = np.nanmax(vals)/np.nanmin(vals[np.where(vals>0)])
    except Exception:
        cond = None
    print("Copula corr shape:", corr_df.shape, "cond approx:", cond)
    # top correlated pairs
    cd = corr_df.where(~np.eye(corr_df.shape[0],dtype=bool))
    flat = cd.unstack().dropna()
    flat_sorted = flat.abs().sort_values(ascending=False).head(20)
    print("Top correlation abs pairs (sample):")
    print(flat_sorted.head(10))

    # heatmap for first 40 symbols
    sample_cols = corr_df.columns[:40]
    plt.figure(figsize=(8,8))
    plt.imshow(corr_df.loc[sample_cols, sample_cols], cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Copula correlation (first 40 symbols)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "copula_corr_40.png"))
    plt.show()

# ---------- 9) Betas heatmap (if betas present) ----------
if isinstance(betas, pd.DataFrame):
    # reduce to first 100 symbols if too many
    B = betas.copy()
    if B.shape[0] > 200:
        B = B.iloc[:200, :]
    plt.figure(figsize=(10, max(4, 0.15*B.shape[0])))
    plt.imshow(B.values, aspect='auto', cmap='bwr', vmin=-np.nanmax(np.abs(B.values)), vmax=np.nanmax(np.abs(B.values)))
    plt.colorbar()
    plt.yticks(range(B.shape[0]), B.index)
    plt.xticks(range(B.shape[1]), B.columns, rotation=45, ha='right')
    plt.title("Betas heatmap (symbols x factors)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "betas_heatmap.png"))
    plt.show()

# ---------- 10) Save numeric summaries to CSV ----------
summary_out = os.path.join(OUT_DIR, "summary_numbers.csv")
rows = []
if final_returns is not None:
    rows.append({
        "name": "final_returns_mean", "value": float(np.nanmean(final_returns))
    })
if stats is not None:
    for k,v in stats.items():
        rows.append({"name": f"sim_{k}", "value": float(v) if np.isscalar(v) else str(v)})
pd.DataFrame(rows).to_csv(summary_out, index=False)

print("All visualizations saved to", OUT_DIR)




# CopulaEngine
import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.stats import norm

class CopulaEngine:
    """
    Gaussian copula engine for residuals (Phase-B2).
    - Fit on standardized residuals (z = (r - mean)/std)
    - Store per-symbol mean & std for later rescaling
    - Store empirical sorted arrays of z for quantile inversion
    - Estimate correlation on transformed z and (optionally) apply Ledoit-Wolf
    - sample(...) returns standardized z samples (dict: symbol -> (n_paths, horizon))
      so the caller (PortfolioSimulator) can rescale using per-asset sigma/time series.
    """

    def __init__(self, shrinkage: bool = True, winsorize_z: float = 8.0):
        self.shrinkage = bool(shrinkage)
        # set winsorize threshold (None or positive float)
        self.winsorize_z = None if winsorize_z is None else float(winsorize_z)
        self.corr_ = None                    # DataFrame (symbols x symbols)
        self._resid_means_ = None            # Series
        self._resid_stds_ = None             # Series (no zeros)
        self._empirical_z_ = dict()          # symbol -> sorted array of historical z (standardized residuals)

    def fit(self, residuals_df: pd.DataFrame, min_obs: int = 30):
        """
        Fit copula using historical residuals (Date x Symbol).
        - residuals_df: DataFrame, may contain NaNs; columns are symbols.
        - min_obs: minimum non-NaN observations required per column to include it.
        Returns correlation DataFrame.
        """

        R = residuals_df.copy()
        # drop columns with too few observations
        valid_cols = [c for c in R.columns if R[c].dropna().shape[0] >= min_obs]
        if len(valid_cols) == 0:
            raise ValueError("No symbols have enough observations to fit copula (min_obs=%d)." % min_obs)
        R = R[valid_cols].dropna(how='all')

        # compute per-symbol mean/std (use ddof=1)
        means = R.mean(axis=0)
        stds = R.std(axis=0, ddof=1).replace(0.0, np.nan)
        # If any std is zero or NaN, replace with global median std to avoid division by zero
        median_std = stds.dropna().median() if stds.dropna().size > 0 else 1.0
        stds = stds.fillna(median_std).replace(0.0, median_std)

        self._resid_means_ = means
        self._resid_stds_ = stds

        # z-score (standardize) and winsorize extremes (if configured)
        Z = (R - means) / stds
        if self.winsorize_z is not None:
            up = self.winsorize_z
            Z = Z.clip(lower=-up, upper=up)

        # rank -> uniform (using average ranks)
        ranks = Z.rank(axis=0, method='average', pct=False)
        n = len(Z)
        u = (ranks - 0.5) / n
        # guard bounds
        eps = 1e-12
        u = u.clip(eps, 1 - eps)

        # gaussianize
        try:
            zvals = u.apply(lambda col: norm.ppf(col)).values  # shape (T, M)
        except Exception:
            # fallback elementwise (slower) if necessary
            zvals = np.zeros_like(u.values, dtype=float)
            for j, col in enumerate(u.columns):
                zvals[:, j] = norm.ppf(u[col].values)

        # estimate correlation of Gaussianized z
        if self.shrinkage:
            lw = LedoitWolf().fit(zvals)
            cov = lw.covariance_
            d = np.sqrt(np.diag(cov))
            corr = cov / np.outer(d, d)
        else:
            corr = np.corrcoef(zvals, rowvar=False)

        self.corr_ = pd.DataFrame(corr, index=Z.columns, columns=Z.columns)

        # store empirical standardized residual arrays (sorted z per symbol) for inversion
        self._empirical_z_ = {}
        for c in Z.columns:
            arr = np.sort(Z[c].dropna().values)
            # if arr is empty fallback to truncated normal support
            if arr.size == 0:
                arr = np.array([-self.winsorize_z if self.winsorize_z is not None else -8.0,
                                0.0,
                                self.winsorize_z if self.winsorize_z is not None else 8.0])
            self._empirical_z_[c] = arr

        return self.corr_

    def sample(self, horizon: int, n_paths: int = 1000, seed: int = 42, randomize_seed: bool = True, clip_z: float = None):
        """
        Sample standardized residual z-values from the fitted copula.

        Returns dict: {symbol: array shape (n_paths, horizon)} containing z-values (standardized).
        clip_z: optional per-call clip to apply to sampled z; if None uses self.winsorize_z.
        """

        if self.corr_ is None:
            raise RuntimeError("CopulaEngine not fitted. Call fit(...) before sample().")

        rng = np.random.default_rng(seed if not randomize_seed else None)

        symbols = list(self.corr_.columns)
        M = len(symbols)
        cov = self.corr_.values

        # ensure PD for cholesky
        eps = 1e-10
        try:
            L = np.linalg.cholesky(cov + eps * np.eye(M))
        except np.linalg.LinAlgError:
            # regularize by eigenvalue clipping
            w, v = np.linalg.eigh(cov)
            w_clipped = np.clip(w, a_min=1e-8, a_max=None)
            cov = (v * w_clipped) @ v.T
            L = np.linalg.cholesky(cov + eps * np.eye(M))

        # determine clip threshold
        clip_threshold = self.winsorize_z if clip_z is None else clip_z

        # prepare output container (standardized z-values)
        sampled_z = {col: np.zeros((n_paths, horizon), dtype=float) for col in symbols}

        # sample path-by-path to avoid giant memory usage for huge M
        for p in range(n_paths):
            # draw independent standard normals (horizon x M)
            Z = rng.normal(size=(horizon, M))
            # apply correlation
            correlated = Z @ L.T  # (horizon, M)
            # convert to uniforms
            U = norm.cdf(correlated)  # (horizon, M)
            # clip U away from 0/1
            U = np.clip(U, 1e-10, 1 - 1e-10)

           
            for j, col in enumerate(symbols):
                arr = self._empirical_z_[col]
                
                try:
                    sampled_vals = np.quantile(arr, U[:, j], method='linear')
                except TypeError:
                   
                    sampled_vals = np.quantile(arr, U[:, j], interpolation='linear')

                # apply clipping to sampled z-values if requested
                if clip_threshold is not None:
                    sampled_vals = np.clip(sampled_vals, -clip_threshold, clip_threshold)

                # store standardized z
                sampled_z[col][p, :] = sampled_vals

        # return standardized z-samples; caller must rescale to residual scale using stored stds & means
        return sampled_z

    # convenience: sample and immediately return rescaled residuals (original scale)
    def sample_residuals(self, horizon: int, n_paths: int = 1000, seed: int = 42, clip_z: float = None):
        """
        Convenience: sample standardized z-values then rescale to original residual scale:
          resid = z * std + mean
        Returns dict: {symbol: array (n_paths, horizon)} in original residual units.
        (Use carefully; recommended approach is to get standardized z and apply per-asset
         time-varying sigma in PortfolioSimulator.)
        """
        sampled_z = self.sample(horizon=horizon, n_paths=n_paths, seed=seed, clip_z=clip_z)
        sampled_resids = {}
        for col, arr in sampled_z.items():
            mean = float(self._resid_means_[col]) if self._resid_means_ is not None else 0.0
            std = float(self._resid_stds_[col]) if self._resid_stds_ is not None else 1.0
            # arr is standardized z; rescale
            sampled_resids[col] = arr * std + mean
        return sampled_resids



# GarchEngine 
class GarchEngine:
    """
    Fit & simulate GARCH(1,1) for factors with explicit scaling control.

    Usage:
      ge = GarchEngine(p=1, q=1, dist='normal', scale=100.0)
      params = ge.fit_all(factor_returns_df)
      paths = ge.simulate_all_factors(['market','sector_0'], n_paths=200, horizon=252, seed=42)

    Notes on scaling:
      - If your input factor returns are in decimal (e.g. 0.01 = 1%), `scale=100.0`
        will turn them into percent-like values (1.0) for arch fitting (helps optimizer).
      - The arch models' fitted params are for the scaled series. Simulation
        is performed in scaled units and then converted back to decimals by dividing
        by `scale` before returning.
    """

    def __init__(self, p=1, q=1, dist='normal', scale: float = 100.0):
        self.p = int(p)
        self.q = int(q)
        self.dist = dist
        self.scale = float(scale) if scale is not None else 1.0

        self.models = {}             # per-factor fitted arch results (statsmodels-like object)
        self.params = {}             # store params dict (scaled units)
        self.fitted_vol_scaled = {}  # conditional vol history in *scaled* units (same units as fit)
        self.fitted_vol = {}         # conditional vol history in decimal units (scaled / scale)

    def fit_all(self, factor_returns_df, by_regime=None, disp=False):
        """
        Fit GARCH models for each factor column.

        factor_returns_df: DataFrame [Date x factor columns] (decimal returns expected)
        by_regime: not implemented in detail here; placeholder for future regime-specific fits
        disp: pass to arch_model.fit(disp=...) to control printing
        Returns: dict of params per factor (scaled units)
        """
        try:
            from arch import arch_model
        except Exception as e:
            raise ImportError("arch package required (pip install arch)") from e

        for col in factor_returns_df.columns:
            series = factor_returns_df[col].dropna()
            if series.shape[0] < 10:
                # too few observations to fit reliably
                continue

            # scale to arch-friendly units (e.g., percent)
            y_scaled = series * self.scale

            # build model: pass rescale=False to avoid internal automatic rescaling
            # (some arch versions accept rescale, others ignore; we wrap in try)
            try:
                am = arch_model(y_scaled, vol='Garch', p=self.p, q=self.q, dist=self.dist, rescale=False)
            except TypeError:
                # arch version might not accept rescale kwarg
                am = arch_model(y_scaled, vol='Garch', p=self.p, q=self.q, dist=self.dist)

            # fit quietly (disp flag)
            res = am.fit(disp=disp)
            # store
            self.models[col] = res
            # params as provided by arch (these are for the scaled series)
            self.params[col] = res.params.to_dict()
            # conditional_volatility from arch is on same units as y_scaled (i.e. scaled units)
            sigma_scaled = res.conditional_volatility  # Series in scaled units
            # keep both representations: scaled and decimal (divide by scale)
            self.fitted_vol_scaled[col] = sigma_scaled
            self.fitted_vol[col] = sigma_scaled / self.scale

        return self.params

    def simulate_factor_paths(self, n_paths, horizon, seed=42, start_sigma=None, factor_name=None):
        """
        Simulate future factor returns using fitted params (basic GARCH(1,1) recursion).
        Returns array shape (n_paths, horizon) in *decimal* units (same as input returns).
        If model for factor_name not found, raises RuntimeError.
        """
        rng = np.random.default_rng(seed)
        if factor_name is None:
            raise ValueError("factor_name must be provided for simulate_factor_paths.")

        if factor_name not in self.models:
            raise RuntimeError(f"simulate_factor_paths: requested factor '{factor_name}' not fitted.")

        res = self.models[factor_name]
        params = res.params

        # params are for scaled series; read them as-is
        # param names can vary; try common keys
        omega = float(params.get('omega', params.get('omega[1]', 1e-6)))
        # alpha and beta may be named 'alpha[1]' or 'alpha' depending on arch version
        alpha = float(params.get('alpha[1]', params.get('alpha', 0.05)))
        beta = float(params.get('beta[1]', params.get('beta', 0.9)))
        mu = float(params.get('mu', params.get('constant', 0.0)))

        # last sigma in scaled units if available, else use provided start_sigma (assumed decimal)
        if factor_name in self.fitted_vol_scaled:
            last_sigma_scaled = float(self.fitted_vol_scaled[factor_name].iloc[-1])
        else:
            # if only start_sigma (decimal) provided, convert to scaled
            last_sigma_scaled = (float(start_sigma) * self.scale) if (start_sigma is not None) else 1.0

        # clamp params to safe ranges
        alpha = max(min(alpha, 0.9999), 0.0)
        beta = max(min(beta, 0.9999), 0.0)
        omega = max(omega, 1e-12)

        paths = np.zeros((n_paths, horizon), dtype=float)

        # simulate in scaled units, then divide by scale to return decimal units
        for p in range(n_paths):
            sigma = last_sigma_scaled
            for t in range(horizon):
                z = rng.normal()
                eps_scaled = sigma * z            # innovation in scaled units
                ret_scaled = mu + eps_scaled      # scaled return
                # store decimal return
                paths[p, t] = ret_scaled / self.scale

                # update GARCH variance in scaled units
                sigma2 = omega + alpha * (eps_scaled ** 2) + beta * (sigma ** 2)
                sigma = np.sqrt(max(sigma2, 1e-12))

        return paths

    def simulate_all_factors(self, factor_names, n_paths, horizon, seed=42):
        """Simulate each factor and return dict factor -> (n_paths, horizon) array (decimal units)."""
        out = {}
        for f in factor_names:
            out[f] = self.simulate_factor_paths(n_paths=n_paths, horizon=horizon, seed=seed, factor_name=f)
        return out




#portfolio simulator
# Try to import user metrics if available; otherwise provide safe fallbacks
try:
    from metrics import sharpe_ratio, sortino_ratio, max_drawdown
except Exception:
    def sharpe_ratio(returns, rf=0.06):
        arr = np.asarray(returns, dtype=float)
        mean = np.nanmean(arr)
        sd = np.nanstd(arr, ddof=1)
        return (mean - rf/252.0) / sd if sd and not np.isnan(sd) else np.nan

    def sortino_ratio(returns, rf=0.06):
        arr = np.asarray(returns, dtype=float)
        mean = np.nanmean(arr)
        downside = np.nanstd(np.minimum(0, arr), ddof=1)
        return (mean - rf/252.0) / downside if downside and not np.isnan(downside) else np.nan

    def max_drawdown(cum_series):
        arr = np.asarray(cum_series, dtype=float)
        if arr.size == 0:
            return np.nan
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / peak
        return float(np.nanmin(dd))


class PortfolioSimulator:
    """
    Engine-driven Portfolio Simulator (Phase-B ready).

    Integration assumptions:
      - copula_engine.sample(horizon, n_paths, seed) returns standardized z-values
        dict: {symbol: ndarray(n_paths, horizon)}.
      - copula_engine optionally exposes _resid_means_ and _resid_stds_ (used only if present).
      - factor_model optionally exposes residuals_ DataFrame (Date x Symbol) for EWMA per-asset sigma.
      - garch_engine should implement simulate_all_factors(factor_names, n_paths, horizon, seed)
        or simulate_factor_paths(factor_name, n_paths, horizon, seed).
    """

    def __init__(self, factor_model=None, regime_model=None, garch_engine=None, copula_engine=None,
                 ewma_span: int = 60, z_clip: float = 8.0, ret_clip: float = 0.2):
        self.factor_model = factor_model
        self.regime_model = regime_model
        self.garch_engine = garch_engine
        self.copula_engine = copula_engine
        self.ewma_span = int(ewma_span)
        self.z_clip = float(z_clip)
        self.ret_clip = float(ret_clip)

    def simulate_portfolio(self, weights: dict, betas: pd.DataFrame,
                           alphas: pd.Series = None,
                           n_paths: int = 1000, horizon: int = 252,
                           regime_series: pd.Series = None, seed: int = 42,
                           verbose: bool = False):
        """
        Public entrypoint.

        Returns:
          (portfolio_paths, final_returns, stats)
          - portfolio_paths: ndarray (n_paths, horizon) of daily portfolio returns
          - final_returns: ndarray (n_paths,) of total return per path (prod(1+daily)-1)
          - stats: dict with mean/std/sharpe/sortino/mdd/n_paths/horizon
        """
        # validate weights
        if not isinstance(weights, dict) or len(weights) == 0:
            raise ValueError("weights must be a non-empty dict mapping symbol -> weight")

        # normalize weights
        total_w = sum(float(v) for v in weights.values())
        if total_w <= 0:
            raise ValueError("weights sum to zero or negative")
        weights = {k: float(v) / total_w for k, v in weights.items()}
        symbols = list(weights.keys())

        # ensure betas is a DataFrame and has rows for all symbols
        betas = pd.DataFrame(betas).copy() if betas is not None else pd.DataFrame()
        for s in symbols:
            if s not in betas.index:
                betas.loc[s] = 0.0
        # Reindex betas to exactly the symbol order for later alignment
        betas = betas.reindex(index=symbols)

        # alphas vector
        if alphas is None:
            alphas = pd.Series(0.0, index=symbols)
        else:
            alphas = pd.Series(alphas).reindex(symbols).fillna(0.0)

        factor_names = list(betas.columns)

        # RNG
        rng = np.random.default_rng(seed)

        # If we have both engines and factor_names, run engine-driven sim
        if (self.garch_engine is not None) and (self.copula_engine is not None) and len(factor_names) > 0:
            if verbose:
                print("PortfolioSimulator: running engine-driven simulation (GARCH + Copula).")
            return self._simulate_with_engines(
                weights=weights, betas=betas, alphas=alphas,
                n_paths=n_paths, horizon=horizon, factor_names=factor_names,
                regime_series=regime_series, seed=seed, rng=rng, verbose=verbose
            )

        # Fallback: Phase-A iid noise behaviour
        if verbose:
            print("PortfolioSimulator: engines missing or factors empty — using iid-normal fallback.")
        w_arr = np.array([weights[s] for s in symbols], dtype=float)
        noise = rng.normal(loc=0.0, scale=0.01, size=(n_paths, horizon, len(symbols)))
        port_daily = (noise * w_arr.reshape(1, 1, -1)).sum(axis=2)
        final_vals = np.prod(1.0 + port_daily, axis=1) - 1.0
        stats = self._compute_stats(port_daily, final_vals)
        return port_daily, final_vals, stats

    def _simulate_with_engines(self, weights, betas, alphas,
                               n_paths, horizon, factor_names,
                               regime_series, seed, rng, verbose=False):
        """
        Engine-driven simulation. Steps:
          1) simulate factor paths F (n_paths, horizon, n_factors)
          2) sample standardized residual z-values from copula (dict symbol -> (n_paths, horizon))
          3) compute per-asset EWMA sigma series from factor_model.residuals_ (if available)
          4) build asset returns: R_total = F @ B + alpha + z * sigma_ts
          5) compute weighted portfolio returns
        """
        # ---------- 1) factor simulations ----------
        factor_paths = {}
        if hasattr(self.garch_engine, "simulate_all_factors"):
            if verbose: print("Calling garch_engine.simulate_all_factors(...)")
            factor_paths = self.garch_engine.simulate_all_factors(factor_names, n_paths=n_paths, horizon=horizon, seed=seed)
        else:
            # fallback per-factor call
            for f in factor_names:
                if hasattr(self.garch_engine, "simulate_factor_paths"):
                    factor_paths[f] = self.garch_engine.simulate_factor_paths(n_paths=n_paths, horizon=horizon, seed=seed, factor_name=f)
                else:
                    raise RuntimeError("garch_engine must implement simulate_all_factors or simulate_factor_paths")

        # validate and stack factor outputs
        F_list = []
        for f in factor_names:
            arr = np.asarray(factor_paths.get(f))
            if arr is None:
                raise RuntimeError(f"Missing factor simulation output for '{f}'")
            # try to coerce to shape (n_paths, horizon)
            if arr.shape != (n_paths, horizon):
                try:
                    arr = arr.reshape((n_paths, horizon))
                except Exception as e:
                    raise RuntimeError(f"Factor '{f}' produced unexpected shape {arr.shape}") from e
            F_list.append(arr)
        F = np.stack(F_list, axis=2)  # shape: (n_paths, horizon, n_factors)

        # ---------- 2) copula standardized-z samples ----------
        if verbose: print("Calling copula_engine.sample(...)")
        if not hasattr(self.copula_engine, "sample"):
            raise RuntimeError("copula_engine must implement sample(horizon, n_paths, seed)")
        z_samples = self.copula_engine.sample(horizon=horizon, n_paths=n_paths, seed=seed)
        # z_samples: dict symbol -> (n_paths, horizon) of standardized z-values

        # ---------- 3) per-asset sigma time-series (EWMA) ----------
        sigma_map = {}  # symbol -> ndarray length horizon
        # prefer factor_model.residuals_ if present
        resid_df = None
        if (self.factor_model is not None) and hasattr(self.factor_model, "residuals_") and (self.factor_model.residuals_ is not None):
            resid_df = self.factor_model.residuals_.copy()
            # normalize dates to midnight if possible to avoid tz mismatch
            try:
                resid_df.index = pd.DatetimeIndex(resid_df.index).normalize()
            except Exception:
                pass

        if resid_df is not None:
            for s in weights.keys():
                if s in resid_df.columns:
                    series = resid_df[s].dropna()
                    if series.shape[0] == 0:
                        sigma_map[s] = np.repeat(series.std(ddof=1) if series.size>0 else 0.01, horizon)
                    else:
                        ewma_var = (series ** 2).ewm(span=self.ewma_span, adjust=False).mean()
                        ewma_sigma = np.sqrt(ewma_var)
                        # take most recent values; pad at the left if shorter than horizon
                        tail = ewma_sigma.values[-horizon:] if ewma_sigma.shape[0] >= horizon else ewma_sigma.values
                        if tail.shape[0] < horizon:
                            pad_len = horizon - tail.shape[0]
                            pad_val = tail[0] if tail.size>0 else (series.std(ddof=1) if series.size>0 else 0.01)
                            tail = np.concatenate([np.repeat(pad_val, pad_len), tail])
                        sigma_map[s] = tail
                else:
                    sigma_map[s] = np.repeat(0.01, horizon)
        else:
            # fallback: use copula-engine stored stds if available, else uniform small sigma
            base_stds = getattr(self.copula_engine, "_resid_stds_", None)
            for s in weights.keys():
                std = float(base_stds.get(s, 0.01)) if base_stds is not None else 0.01
                sigma_map[s] = np.repeat(std, horizon)

        # ---------- 4) combine factor + residual components ----------
        symbols = list(weights.keys())
        n_symbols = len(symbols)
        # B matrix: (n_factors, n_symbols) aligning factor order
        B = betas.reindex(columns=factor_names).fillna(0.0).T.values  # (n_factors, n_symbols)
        alpha_vals = alphas.reindex(symbols).fillna(0.0).values  # (n_symbols,)
        w_arr = np.array([weights[s] for s in symbols], dtype=float)

        portfolio_paths = np.zeros((n_paths, horizon), dtype=float)

        # optional copula mean usage if available
        copula_means = getattr(self.copula_engine, "_resid_means_", None)

        # Loop per path to keep memory bounded
        for p in range(n_paths):
            # systematic returns for this path: (horizon x n_symbols)
            R_sys = F[p].dot(B)  # (horizon, n_symbols)
            R_sys = R_sys + alpha_vals.reshape(1, -1)

            # residuals for this path
            R_resid = np.zeros_like(R_sys)
            for j, s in enumerate(symbols):
                z_arr = z_samples.get(s)
                if z_arr is None:
                    R_resid[:, j] = 0.0
                    continue
                z_row = z_arr[p, :]
                # clip standardized z
                z_row = np.clip(z_row, -self.z_clip, self.z_clip)
                sigma_ts = sigma_map.get(s, np.repeat(0.01, horizon))
                if copula_means is not None and s in copula_means.index:
                    mean_s = float(copula_means[s])
                    R_resid[:, j] = z_row * sigma_ts + mean_s
                else:
                    R_resid[:, j] = z_row * sigma_ts

            # total returns and safe clipping
            R_total = R_sys + R_resid
            R_total = np.clip(R_total, -self.ret_clip, self.ret_clip)

            # portfolio daily returns for this path
            port_ts = R_total.dot(w_arr)
            portfolio_paths[p, :] = port_ts

        # final returns per path
        final_returns = np.prod(1.0 + portfolio_paths, axis=1) - 1.0
        stats = self._compute_stats(portfolio_paths, final_returns)
        return portfolio_paths, final_returns, stats

    def _compute_stats(self, port_daily_array, final_vals):
        mean_ret = float(np.nanmean(final_vals))
        std_ret = float(np.nanstd(final_vals, ddof=1))
        try:
            sr = sharpe_ratio(final_vals)
        except Exception:
            sr = np.nan
        try:
            sor = sortino_ratio(final_vals)
        except Exception:
            sor = np.nan
        avg_daily = np.nanmean(port_daily_array, axis=0)
        cum = np.cumprod(1.0 + avg_daily)
        try:
            mdd = float(max_drawdown(cum))
        except Exception:
            mdd = np.nan

        return {
            "mean": mean_ret,
            "std": std_ret,
            "sharpe": sr,
            "sortino": sor,
            "mdd": mdd,
            "n_paths": int(port_daily_array.shape[0]),
            "horizon": int(port_daily_array.shape[1])
        }


"""
metrics.py
------------
Portfolio and simulation evaluation metrics.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, ks_2samp

def sharpe_ratio(returns, rf=0.06):
    mean, std = np.nanmean(returns), np.nanstd(returns)
    return (mean - rf/252) / std if std > 0 else np.nan

def sortino_ratio(returns, rf=0.06):
    downside = np.nanstd(np.minimum(0, returns))
    return (np.nanmean(returns) - rf/252) / downside if downside > 0 else np.nan

def max_drawdown(cum_returns):
    peak = np.maximum.accumulate(cum_returns)
    dd = (cum_returns - peak) / peak
    return dd.min()

def value_at_risk(returns, alpha=0.05):
    return np.nanpercentile(returns, 100 * alpha)

def cvar(returns, alpha=0.05):
    cutoff = value_at_risk(returns, alpha)
    return returns[returns <= cutoff].mean()

def coverage(real, simulated, p_low=10, p_high=90):
    low, high = np.nanpercentile(simulated, [p_low, p_high])
    return (low <= real <= high)

def ks_similarity(real_series, simulated_series):
    return ks_2samp(real_series, simulated_series).pvalue





# Robust FactorModel: Ledoit-Wolf + EWMA + robust beta estimation

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf

class FactorModel:
    """
    FactorModel vB1 (robust)
      - build pseudo-sector clusters (KMeans on correlation)
      - compute simple factor returns (market, sector means, style)
      - estimate betas (robust per-symbol OLS with index normalization)
      - compute residuals aligned to factors
      - Ledoit-Wolf shrunk covariance and correlation
      - EWMA vol helpers
    """
    def __init__(self, n_clusters: int = 10, ewma_span: int = 60, min_obs: int = 250):
        self.n_clusters = int(n_clusters)
        self.ewma_span = int(ewma_span)
        self.min_obs = int(min_obs)
        self.cluster_labels_ = None
        self.factor_returns_ = None
        self.betas_ = None
        self.residuals_ = None
        self.shrunk_cov_ = None

    # ------------------ internal helpers ------------------
    @staticmethod
    def _normalize_index_to_date(idx: pd.Index) -> pd.DatetimeIndex:
        """
        Convert index to timezone-naive date-normalized DatetimeIndex (midnight).
        This removes timezone/time-of-day mismatches for daily data alignment.
        """
        di = pd.DatetimeIndex(pd.to_datetime(idx))
        # drop tz-awareness safely
        if di.tz is not None:
            try:
                di = di.tz_convert(None)
            except Exception:
                di = di.tz_localize(None)
        return di.normalize()

    # ------------------ clustering ------------------
    def build_sector_clusters(self, returns_df: pd.DataFrame, n_clusters: int = None, random_state: int = 42):
        """
        Cluster stocks into pseudo-sectors using correlation matrix rows.
        returns_df: Date x Symbol returns (DataFrame)
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        # correlation on columns (symbols)
        corr = returns_df.corr().fillna(0.0)
        km = KMeans(n_clusters=n_clusters, random_state=random_state)
        km.fit(corr.values)
        labels = pd.Series(km.labels_, index=corr.index, name="sector_id")
        self.cluster_labels_ = labels
        return labels

    # ------------------ factor construction ------------------
    def compute_factor_returns(self, stock_returns: pd.DataFrame, market_returns: pd.Series = None):
        """
        Build factor returns DataFrame indexed by Date.
        Factors: market, sector_<id> (if clusters present), style_mom30
        """
        # start with stock_returns index (do not normalize here; we'll normalize later in regressions)
        df = pd.DataFrame(index=stock_returns.index)
        # market factor
        if market_returns is None:
            df["market"] = stock_returns.mean(axis=1)
        else:
            df["market"] = market_returns.reindex(stock_returns.index).ffill()

        # sector factors (mean of cluster members)
        if self.cluster_labels_ is not None:
            for cid in sorted(self.cluster_labels_.unique()):
                members = [s for s, lab in self.cluster_labels_.items() if lab == cid and s in stock_returns.columns]
                if len(members) == 0:
                    continue
                df[f"sector_{cid}"] = stock_returns[members].mean(axis=1)

        # style: 30-day average of returns' cross-section mean (simple)
        df["style_mom30"] = stock_returns.rolling(30).mean().mean(axis=1)

        # drop rows that are fully NA
        df = df.dropna(how='all')
        self.factor_returns_ = df
        return df

    # ------------------ robust beta estimation ------------------
    def estimate_betas(self, stock_returns: pd.DataFrame, factor_returns: pd.DataFrame, window: int = None):
        """
        Robust per-symbol OLS beta estimation.
        Aligns on overlapping non-NaN dates per symbol and normalizes dates to avoid tz/time mismatches.
        If `window` is provided, uses the last `window` factor rows as the candidate period
        and intersects each symbol's available dates with that window.
        Returns DataFrame index=symbol, columns=factor names
        """
        if stock_returns is None or factor_returns is None:
            raise ValueError("stock_returns and factor_returns must be provided")

        # Copy to avoid side-effects and sort index
        sr = stock_returns.copy().sort_index()
        fr = factor_returns.copy().sort_index()

        # Normalize both indexes to date-level timezone-naive midnight
        sr.index = self._normalize_index_to_date(sr.index)
        fr.index = self._normalize_index_to_date(fr.index)

        factor_cols = list(fr.columns)
        betas = {}

        # Candidate factor window to use for rolling mode
        if window is not None and window > 0:
            if fr.shape[0] < window:
                fr_window = fr.copy()
            else:
                fr_window = fr.iloc[-window:]
        else:
            fr_window = fr  # full factor frame

        # perform symbol-by-symbol OLS using only overlapping dates
        lr = LinearRegression()
        for sym in sr.columns:
            # dates where symbol has data
            idx_sym = sr[sym].dropna().index
            # intersect with factor window index
            idx_common = idx_sym.intersection(fr_window.index)
            if idx_common.size == 0:
                betas[sym] = np.full(len(factor_cols), np.nan)
                continue
            y = sr.loc[idx_common, sym].values
            Xloc = fr_window.loc[idx_common].values
            if y.size < max(10, self.min_obs // 5):
                betas[sym] = np.full(len(factor_cols), np.nan)
                continue
            try:
                lr.fit(Xloc, y)
                betas[sym] = lr.coef_.astype(float)
            except Exception:
                betas[sym] = np.full(len(factor_cols), np.nan)

        df_betas = pd.DataFrame(betas, index=factor_cols).T
        self.betas_ = df_betas
        return df_betas

    # ------------------ residuals ------------------
    def compute_residuals(self, stock_returns: pd.DataFrame, factor_returns: pd.DataFrame, betas: pd.DataFrame = None):
        """
        Compute residuals eps_{i,t} = r_{i,t} - beta_i . f_t
        Aligns with normalized date indices and returns DataFrame (Date x Symbols)
        """
        if betas is None:
            betas = self.betas_
        if betas is None:
            raise ValueError("No betas available. Call estimate_betas() or pass betas argument.")

        sr = stock_returns.copy().sort_index()
        fr = factor_returns.copy().sort_index()
        # Normalize indexes
        sr.index = self._normalize_index_to_date(sr.index)
        fr.index = self._normalize_index_to_date(fr.index)

        # Intersect times
        common_idx = fr.index.intersection(sr.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping dates between stock_returns and factor_returns after normalization.")

        # Subset to betas symbols
        syms = [s for s in betas.index if s in sr.columns]
        if len(syms) == 0:
            raise ValueError("No symbols from betas found in stock_returns columns.")

        R = sr.loc[common_idx, syms].fillna(0.0)  # T x n_symbols
        F = fr.loc[common_idx].fillna(0.0)        # T x k
        # ensure ordering alignment: betas rows correspond to syms order
        B = betas.loc[syms].values                 # n_symbols x k

        pred = F.values @ B.T                      # T x n_symbols
        resid = R.values - pred
        resid_df = pd.DataFrame(resid, index=common_idx, columns=syms)
        self.residuals_ = resid_df
        return resid_df

    # ------------------ Ledoit-Wolf shrinkage ------------------
    def ledoit_wolf_cov(self, returns_df: pd.DataFrame):
        """
        Compute Ledoit-Wolf shrunk covariance and correlation matrices for returns_df.
        Returns (cov_df, corr_df)
        Rows with any NaN are dropped before fitting.
        """
        clean = returns_df.dropna(how='any')
        if clean.shape[0] < 2:
            raise ValueError("Not enough clean observations for Ledoit-Wolf.")
        arr = clean.values
        cols = clean.columns
        lw = LedoitWolf().fit(arr)
        cov = pd.DataFrame(lw.covariance_, index=cols, columns=cols)
        # convert to correlation
        d = np.sqrt(np.diag(cov))
        corr = cov.div(d, axis=0).div(d, axis=1)
        self.shrunk_cov_ = cov
        return cov, corr

    # ------------------ EWMA volatility ------------------
    def ewma_vol(self, series: pd.Series, span: int = None):
        """
        EWMA volatility series computed from returns series.
        Returns a Series (same index) of EWMA sigma.
        """
        if span is None:
            span = self.ewma_span
        s = pd.to_numeric(series).fillna(0.0)
        # pandas ewm on squared returns
        ewma_var = (s ** 2).ewm(span=span, adjust=False).mean()
        ewma_sigma = np.sqrt(ewma_var)
        return ewma_sigma

    def factor_ewma_vols(self, factor_returns: pd.DataFrame, span: int = None):
        if span is None:
            span = self.ewma_span
        vols = factor_returns.apply(lambda s: self.ewma_vol(s, span=span))
        return vols

    # ------------------ summary ------------------
    def summary(self):
        return {
            "n_clusters": int(self.cluster_labels_.nunique()) if self.cluster_labels_ is not None else 0,
            "n_betas": int(self.betas_.shape[0]) if self.betas_ is not None else 0,
            "n_factors": int(self.factor_returns_.shape[1]) if self.factor_returns_ is not None else 0
        }




# RegimeModel cell
import numpy as np
import pandas as pd

class RegimeModel:
    """
    RegimeModel:
      - detect regimes deterministically via EWMA volatility threshold
      - (optional) fit HMM if you have hmmlearn installed
    """
    def __init__(self, method='threshold', ewma_span=60, thresh_quantile=0.9, min_samples=250):
        self.method = method
        self.ewma_span = ewma_span
        self.thresh_quantile = thresh_quantile
        self.min_samples = min_samples
        self.regime_series_ = None
        self.hmm_model_ = None

    def ewma_vol(self, series, span=None):
        if span is None: span = self.ewma_span
        return series.pow(2).ewm(span=span, adjust=False).mean().pipe(np.sqrt)

    def detect_threshold(self, returns_df, market_col=None):
        """
        returns_df: DataFrame (Date x asset) or a Series (market index).
        market_col: optionally a column name to use as market proxy. If None, use mean across columns.
        """
        if isinstance(returns_df, pd.DataFrame) and market_col is not None:
            m = returns_df[market_col]
        elif isinstance(returns_df, pd.DataFrame):
            m = returns_df.mean(axis=1)
        elif isinstance(returns_df, pd.Series):
            m = returns_df
        else:
            raise ValueError("returns_df must be DataFrame or Series")
        vol = self.ewma_vol(m)
        q = vol.quantile(self.thresh_quantile)
        regimes = pd.Series(np.where(vol > q, 'high', 'low'), index=vol.index, name='regime')
        self.regime_series_ = regimes
        return regimes

    def fit_hmm(self, factor_returns, n_states=2, n_iter=100):
        """Optional: fit HMM on multivariate factors (requires hmmlearn)."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except Exception as e:
            raise ImportError("hmmlearn required for fit_hmm; pip install hmmlearn") from e

        # we'll fit on the market factor (or PCA of factors) for simplicity
        X = factor_returns.fillna(method='ffill').dropna()
        # reduce to primary dimension if multivariate
        if X.shape[1] > 1:
            # use first principal component as observation
            from sklearn.decomposition import PCA
            pc = PCA(1).fit_transform(X.values)
            obs = pc
        else:
            obs = X.values.reshape(-1,1)

        model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=n_iter)
        model.fit(obs)
        states = model.predict(obs)
        idx = X.index
        regimes = pd.Series(states, index=idx).map(lambda s: f"regime_{s}")
        self.hmm_model_ = model
        self.regime_series_ = regimes
        return regimes

    def fit(self, returns, method=None, **kwargs):
        if method is None: method = self.method
        if method == 'threshold':
            return self.detect_threshold(returns, **kwargs)
        elif method == 'hmm':
            return self.fit_hmm(returns, **kwargs)
        else:
            raise ValueError("method must be 'threshold' or 'hmm'")




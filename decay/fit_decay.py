#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
decay/fit_decay.py  •  CRI v0.3-SIM (robust)

Fits the Tier-A log-linear decay:
    y = ln A_pre(τ_f) = β0 + β1 * τ_f  with β1 = -1/τ_fut

Reports OLS, WLS (if 'se_lnA' available), and left-censored Tobit MLE
(using detection bound ln A_min). All three are bootstrapped for 95% CIs.

Reads:
  - decay/output/decay_data.csv          (columns: delta, lnA_pre[, se_lnA])
  - decay/default_params.yml             (for A_min or epsilon_detection)
Writes:
  - decay/output/fit_decay_results.csv   (OLS + WLS + Tobit τ_fut and CIs)
  - decay/output/decay_band.csv          (x-grid with bootstrap CI band for OLS line)
"""
import os, sys, yaml, math
import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    from utilities.seed_manager import load_state, save_state
except Exception:
    def load_state(): pass
    def save_state(): pass

# ----------------------- IO helpers -----------------------
def _load_params():
    here = os.path.dirname(__file__)
    path = os.path.join(here, 'default_params.yml')
    with open(path, 'r', encoding='utf-8') as f:
        y = yaml.safe_load(f)
    p = y['decay'] if isinstance(y, dict) and 'decay' in y else y

    A_min = p.get('A_min', p.get('epsilon_detection', 0.01))
    try: A_min = float(A_min)
    except Exception: A_min = 0.01

    # NEW: honor your YAML keys n_boot and n_cont
    n_bootstrap = int(p.get('n_bootstrap', p.get('n_boot', 2000)))
    n_points    = int(p.get('n_points',    p.get('n_cont', 200)))

    return {
        'seed':        int(p.get('seed', 52)),
        'n_bootstrap': n_bootstrap,
        'ci_percent':  float(p.get('ci_percent', 95.0)),
        'A_min':       A_min,
        'figure_dpi':  int(p.get('figure_dpi', 1200)),
        'x_min_ms':    float(p.get('x_min_ms', 0.0)),   # not critical
        'x_max_ms':    float(p.get('x_max_ms', 20.0)),  # not critical
        'n_points':    n_points,
    }


def _load_data():
    here = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(here, 'output', 'decay_data.csv'))
    if 'delta' not in df.columns or 'lnA_pre' not in df.columns:
        raise RuntimeError("decay_data.csv must have columns: 'delta', 'lnA_pre'[, 'se_lnA'].")
    return df

# ----------------------- core fits ------------------------
def _ols_fit(x, y):
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(b[0]), float(b[1])

def _wls_fit(x, y, se=None):
    """
    Robust WLS:
      - sanitize SEs → positive finite weights
      - solve via sqrt(W) least squares (np.linalg.lstsq)
      - if rank-deficient or ill-conditioned → fallback to OLS
      - final fallback: tiny ridge on normal equations
    """
    if se is None:
        return _ols_fit(x, y)

    se = np.asarray(se, float)
    x  = np.asarray(x,  float)
    y  = np.asarray(y,  float)

    # Build positive finite weights
    se = np.where(~np.isfinite(se) | (se <= 0), np.nan, se)
    w  = 1.0 / np.square(np.maximum(se, 1e-8))
    mask = np.isfinite(w) & (w > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return _ols_fit(x[mask], y[mask]) if mask.any() else _ols_fit(x, y)

    X = np.column_stack([np.ones_like(x[mask]), x[mask]])
    # If all x equal in this resample → no slope identifiable
    if np.allclose(X[:,1], X[0,1]):
        return _ols_fit(x[mask], y[mask])

    sqrtw = np.sqrt(w[mask])[:, None]
    Xw = X * sqrtw
    yw = y[mask] * sqrtw[:, 0]
    try:
        b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        return float(b[0]), float(b[1])
    except np.linalg.LinAlgError:
        # tiny ridge as last resort
        XT_W_X = X.T @ (w[mask][:, None] * X)
        XT_W_y = X.T @ (w[mask] * y[mask])
        lam = 1e-12
        b = np.linalg.solve(XT_W_X + lam * np.eye(2), XT_W_y)
        return float(b[0]), float(b[1])

def _tobit_fit(x, y, c):
    """
    Left-censored Tobit MLE at bound c.
    y* = b0 + b1 x + eps, eps ~ N(0, σ^2); y = max(y*, c)
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    cens = (y <= c + 1e-12)

    def nll(theta):
        b0, b1, log_sig = theta
        sig = np.exp(log_sig)
        mu  = b0 + b1 * x
        z   = (y - mu) / sig
        zc  = (c - mu) / sig
        ll_unc = -0.5*np.log(2*np.pi) - log_sig - 0.5*z**2
        ll_cen = stats.norm.logcdf(zc)
        ll = np.where(cens, ll_cen, ll_unc)
        return -np.sum(ll)

    # init from OLS
    b0, b1 = _ols_fit(x, y)
    sig0   = np.std(y - (b0 + b1*x), ddof=2)
    theta0 = np.array([b0, b1, np.log(max(sig0, 1e-3))], float)
    bounds = [(None, None), (None, None), (np.log(1e-6), np.log(1e3))]

    best = None
    rng = np.random.default_rng(52)
    for k in range(6):
        start = theta0 if k == 0 else theta0 + rng.normal(scale=[0.05, 0.02, 0.05])
        res = optimize.minimize(nll, start, method='L-BFGS-B', bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
        if res.success:
            break
    b0_hat, b1_hat, _ = best.x
    return float(b0_hat), float(b1_hat)

# ----------------------- bootstrap wrappers -----------------------
def _bootstrap_ci(vals, ci_percent=95.0):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    a = (100.0 - ci_percent)/2.0
    return float(np.percentile(vals, a)), float(np.percentile(vals, 100.0 - a))

def _fit_all(df, A_min, n_boot=2000, ci_percent=95.0, seed=52, n_points=200):
    x  = df['delta'].values.astype(float)      # seconds
    y  = df['lnA_pre'].values.astype(float)
    se = df['se_lnA'].values.astype(float) if 'se_lnA' in df.columns else None
    c  = float(np.log(A_min))

    # point fits
    b0_ols,  b1_ols  = _ols_fit(x, y)
    b0_wls,  b1_wls  = _wls_fit(x, y, se)
    b0_tob,  b1_tob  = _tobit_fit(x, y, c)

    def to_tau(b1):
        # Robust: ignore non-identifiable / non-decaying bootstrap resamples
        if (not np.isfinite(b1)) or (b1 >= 0):
            return np.nan
        return -1.0 / b1


    tau_ols  = to_tau(b1_ols)
    tau_wls  = to_tau(b1_wls)
    tau_tob  = to_tau(b1_tob)

    # bootstrap
    rng = np.random.default_rng(seed)
    B = int(n_boot)
    xs = np.linspace(x.min(), x.max(), int(n_points))
    lines = []
    tau_ols_B, tau_wls_B, tau_tob_B = [], [], []

    for _ in range(B):
        idx = rng.integers(len(x), size=len(x))
        xb, yb = x[idx], y[idx]
        seb = se[idx] if se is not None else None

        # OLS
        _, b1b = _ols_fit(xb, yb)
        tau_ols_B.append(to_tau(b1b))

        # WLS (robust; may fallback internally)
        _, b1wb = _wls_fit(xb, yb, seb)
        tau_wls_B.append(to_tau(b1wb))

        # Tobit
        _, b1tb = _tobit_fit(xb, yb, c)
        tau_tob_B.append(to_tau(b1tb))

        # central OLS line for band
        b0b, b1b = _ols_fit(xb, yb)
        lines.append(b0b + b1b * xs)

    lines = np.asarray(lines)
    alpha = (100.0 - ci_percent)/2.0
    lo, hi = np.percentile(lines, [alpha, 100.0 - alpha], axis=0)
    cen    = b0_ols + b1_ols * xs

    ci_ols   = _bootstrap_ci(tau_ols_B,  ci_percent)
    ci_wls   = _bootstrap_ci(tau_wls_B,  ci_percent)
    ci_tobit = _bootstrap_ci(tau_tob_B,  ci_percent)

    band = pd.DataFrame({
        'delta_cont': xs,
        'lnA_central': cen,
        'lnA_low': lo,
        'lnA_high': hi
    })

    results = pd.DataFrame([{
        'tau_hat_ms':        tau_ols  * 1e3,
        'ci_lo_ms':          ci_ols[0]* 1e3,
        'ci_hi_ms':          ci_ols[1]* 1e3,
        'tau_hat_ms_wls':    tau_wls  * 1e3,
        'wls_ci_lo_ms':      ci_wls[0]* 1e3,
        'wls_ci_hi_ms':      ci_wls[1]* 1e3,
        'tau_hat_ms_tobit':  tau_tob  * 1e3,
        'tobit_ci_lo_ms':    ci_tobit[0]*1e3,
        'tobit_ci_hi_ms':    ci_tobit[1]*1e3
    }])

    # simple flags for SI text
    lo_ms, hi_ms = results['ci_lo_ms'].iloc[0], results['ci_hi_ms'].iloc[0]
    results['agree_wls_in_ols_CI']   = bool(lo_ms <= results['tau_hat_ms_wls'].iloc[0]   <= hi_ms)
    results['agree_tobit_in_ols_CI'] = bool(lo_ms <= results['tau_hat_ms_tobit'].iloc[0] <= hi_ms)

    return results, band

# ----------------------- main -----------------------
def main():
    load_state()
    params = _load_params()
    save_state()

    here = os.path.dirname(__file__)
    outd = os.path.join(here, 'output')
    os.makedirs(outd, exist_ok=True)

    df = _load_data()
    res, band = _fit_all(
        df, A_min=params['A_min'],
        n_boot=params['n_bootstrap'],
        ci_percent=params['ci_percent'],
        seed=params['seed'],
        n_points=params['n_points']
    )

    res.to_csv(os.path.join(outd, 'fit_decay_results.csv'), index=False)
    band.to_csv(os.path.join(outd, 'decay_band.csv'), index=False)

    print(f"τ_fut (OLS) = {float(res['tau_hat_ms'].iloc[0]):.1f} ms | "
          f"WLS={float(res['tau_hat_ms_wls'].iloc[0]):.1f} ms | "
          f"Tobit={float(res['tau_hat_ms_tobit'].iloc[0]):.1f} ms")

if __name__ == '__main__':
    main()

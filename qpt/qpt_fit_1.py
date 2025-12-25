# -*- coding: utf-8 -*-
"""
qpt/qpt_fit.py

Fits:
  1) For each γ_b, linear fit to ln P(t) to recover (γ_f+γ_b)
  2) R_mean vs λ_env → linear regression + bootstrap CI for slope

Writes:
  - qpt/output/qpt_pop_fit.csv
  - qpt/output/qpt_R_fit.csv
"""
import os, yaml
import numpy as np
import pandas as pd
from scipy.stats import linregress

def load_params():
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, 'default_params.yml')
    with open(cfg_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        cfg = yaml.safe_load(f)
    return cfg['qpt']

def fit_population(t, pop_row):
    x = np.asarray(t, dtype=float)
    y = np.log(np.asarray(pop_row, dtype=float))  # force numeric array
    slope, _, _, _, stderr = linregress(x, y)
    gamma_sum = -2.0 * slope
    err = 2.0 * stderr
    return gamma_sum, err

def fit_R(lambda_env, R_mean):
    x = np.asarray(lambda_env, dtype=float)
    y = np.asarray(R_mean, dtype=float)
    slope, intercept, _, _, stderr = linregress(x, y)
    return slope, intercept, stderr

def bootstrap_R(lambda_env, R_mean, n_boot, ci):
    rng = np.random.default_rng(0)
    x = np.asarray(lambda_env, dtype=float)
    y = np.asarray(R_mean, dtype=float)
    n = len(x)
    slopes = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        s, _, _ = fit_R(x[idx], y[idx])
        slopes.append(s)
    low, high = np.percentile(slopes, [(100-ci)/2, 100-(100-ci)/2])
    return low, high

def main():
    p = load_params()
    here = os.path.dirname(__file__)
    dat = np.load(os.path.join(here, 'output', 'qpt_sim_data.npz'))
    t            = dat['t']
    gamma_b_vals = dat['gamma_b_vals']
    pops         = dat['pops']          # 2D float array (n_gb × n_t)
    lambda_env   = dat['lambda_env']
    R_mean       = dat['R_mean']

    # Population fits
    rows = []
    for gb, pop_row in zip(gamma_b_vals, pops):
        rate, err = fit_population(t, pop_row)
        rows.append({'gamma_b': float(gb), 'gamma_sum': float(rate), 'err': float(err)})
    pd.DataFrame(rows).to_csv(os.path.join(here, 'output', 'qpt_pop_fit.csv'), index=False)

    # R fit + bootstrap CI for slope
    slope, intercept, s_err = fit_R(lambda_env, R_mean)
    ci_low, ci_high = bootstrap_R(lambda_env, R_mean, p.get('n_bootstrap', 2000), p.get('ci_percent', 95))
    df_R = pd.DataFrame([{
        'slope': float(slope),
        'intercept': float(intercept),
        'slope_err': float(s_err),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high)
    }])
    df_R.to_csv(os.path.join(here, 'output', 'qpt_R_fit.csv'), index=False)
    print("Saved qpt_pop_fit.csv and qpt_R_fit.csv")

if __name__=='__main__':
    main()

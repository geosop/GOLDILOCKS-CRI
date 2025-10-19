# -*- coding: utf-8 -*-
"""
decay/wls_tobit_robustness.py  •  CRI v0.3-SIM

Purpose (reviewer-robust):
  1) On the current synthetic dataset (matching Box-2a), fit OLS/WLS/Tobit
     and test whether WLS/Tobit τ̂ fall inside the OLS-bootstrap CI.
  2) Optionally, Monte-Carlo repeat the whole simulation across R new datasets
     to report 'coverage' (fraction of repeats for which WLS/Tobit ∈ OLS CI).

Outputs:
  - decay/output/wls_tobit_check.csv        (single-dataset check, slopes+taus)
  - decay/output/wls_tobit_coverage.csv     (Monte-Carlo summary if R>0)
"""
from __future__ import annotations

# stdlib
import os, sys, yaml, math, argparse

# third-party
import numpy as np
import pandas as pd

# --- make the repo root importable when running this file by path -------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# now these work no matter how the script is launched
from decay.fit_decay import _ols_fit, _wls_fit, _tobit_fit, _bootstrap_ci, _fit_all


HERE = os.path.dirname(__file__)
OUTD = os.path.join(HERE, "output")
os.makedirs(OUTD, exist_ok=True)

def _load_params():
    with open(os.path.join(HERE, "default_params.yml"), "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    p = y["decay"] if isinstance(y, dict) and "decay" in y else y
    return {
        "seed":        int(p.get("seed", 52)),
        "A0":          float(p.get("A0", 1.0)),
        "tau_f":       float(p.get("tau_f", 0.02)),
        "noise_log":   float(p.get("noise_log", 0.10)),
        "delta_start": float(p.get("delta_start", 0.0)),
        "delta_end":   float(p.get("delta_end", 0.02)),
        "delta_step":  float(p.get("delta_step", 0.005)),
        "n_rep":       int(p.get("n_rep", 40)),
        "A_min":       float(p.get("A_min", 0.367879441)),
        "n_boot":      int(p.get("n_boot", p.get("n_bootstrap", 2000))),
        "ci_percent":  float(p.get("ci_percent", 95.0)),
    }

def _simulate_once(seed, A0, tau_f, noise_log, delta_start, delta_end, delta_step, n_rep):
    rng = np.random.default_rng(seed)
    deltas = np.arange(delta_start, delta_end + 1e-12, delta_step)
    mu = np.log(A0) - deltas / tau_f
    rows = []
    for d, m in zip(deltas, mu):
        y = m + rng.normal(0.0, noise_log, size=n_rep)
        rows.extend({"delta": float(d), "lnA_pre_raw": float(v)} for v in y)
    df_raw = pd.DataFrame(rows)
    agg = df_raw.groupby("delta", as_index=False).agg(
        lnA_pre=("lnA_pre_raw", "mean"),
        sd=("lnA_pre_raw", "std"),
        n=("lnA_pre_raw", "size"),
    )
    agg["se_lnA"] = agg["sd"] / np.sqrt(agg["n"])
    return agg[["delta", "lnA_pre", "se_lnA"]], df_raw

def _bootstrap_slope_ci_ols(x, y, B=2000, ci=95.0, seed=52):
    rng = np.random.default_rng(seed)
    b1s = []
    n = len(x)
    for _ in range(int(B)):
        idx = rng.integers(n, size=n)
        _, b1 = _ols_fit(x[idx], y[idx])
        b1s.append(b1)
    a = (100.0 - ci)/2.0
    return float(np.percentile(b1s, a)), float(np.percentile(b1s, 100.0 - a))

def _to_tau(b1):
    return np.inf if b1 >= 0 else -1.0 / b1

def check_once(df, A_min, n_boot, ci_percent, seed):
    x = df["delta"].values.astype(float)
    y = df["lnA_pre"].values.astype(float)
    se = df["se_lnA"].values.astype(float)

    # Point fits
    b0_ols,  b1_ols  = _ols_fit(x, y)
    b0_wls,  b1_wls  = _wls_fit(x, y, se)
    b0_tob,  b1_tob  = _tobit_fit(x, y, math.log(A_min))

    tau_ols  = _to_tau(b1_ols)
    tau_wls  = _to_tau(b1_wls)
    tau_tob  = _to_tau(b1_tob)

    # Use the same robust routine to get OLS τ CI (bootstrap over resamples)
    results, _ = _fit_all(df, A_min=A_min, n_boot=n_boot, ci_percent=ci_percent, seed=seed)
    lo_tau = float(results["ci_lo_ms"].iloc[0]) / 1e3
    hi_tau = float(results["ci_hi_ms"].iloc[0]) / 1e3

    # Also get a slope CI directly for OLS, for completeness
    slo_lo, slo_hi = _bootstrap_slope_ci_ols(x, y, B=n_boot, ci=ci_percent, seed=seed)

    row = {
        "b1_ols":  b1_ols,
        "b1_wls":  b1_wls,
        "b1_tob":  b1_tob,
        "slope_CI_lo_OLS": slo_lo,
        "slope_CI_hi_OLS": slo_hi,
        "tau_ols": tau_ols,
        "tau_wls": tau_wls,
        "tau_tob": tau_tob,
        "tau_CI_lo_OLS": lo_tau,
        "tau_CI_hi_OLS": hi_tau,
        "wls_in_OLS_tau_CI": (lo_tau <= tau_wls <= hi_tau),
        "tobit_in_OLS_tau_CI": (lo_tau <= tau_tob <= hi_tau),
        "wls_in_OLS_slope_CI": (slo_lo <= b1_wls <= slo_hi),
        "tobit_in_OLS_slope_CI": (slo_lo <= b1_tob <= slo_hi),
    }
    return pd.DataFrame([row])

def main():
    ap = argparse.ArgumentParser(description="WLS/Tobit robustness vs OLS CI")
    ap.add_argument("--repeats", type=int, default=0, help="Monte-Carlo repeats (0 = single check only)")
    ap.add_argument("--seed", type=int, default=None, help="base RNG seed (default: YAML)")
    ap.add_argument("--n-boot", type=int, default=None, help="bootstrap B for CIs (default: YAML)")
    ap.add_argument("--ci", type=float, default=None, help="CI percent (default: YAML)")
    ap.add_argument("--out-prefix", default=os.path.join(OUTD, "wls_tobit"), help="output prefix")
    args = ap.parse_args()

    p = _load_params()
    base_seed   = p["seed"] if args.seed is None else int(args.seed)
    n_boot      = p["n_boot"] if args.n_boot is None else int(args.n_boot)
    ci_percent  = p["ci_percent"] if args.ci is None else float(args.ci)

    # ---------- Single-dataset check (use the exact Box-2a settings) ----------
    df, df_raw = _simulate_once(
        seed=base_seed,
        A0=p["A0"], tau_f=p["tau_f"], noise_log=p["noise_log"],
        delta_start=p["delta_start"], delta_end=p["delta_end"],
        delta_step=p["delta_step"], n_rep=p["n_rep"]
    )
    single = check_once(df, A_min=p["A_min"], n_boot=n_boot, ci_percent=ci_percent, seed=base_seed)
    single.to_csv(args.out_prefix + "_check.csv", index=False)
    print("\nSingle-dataset check:")
    print(single.to_string(index=False))

    # ---------- Optional Monte-Carlo coverage over new datasets ----------
    R = int(args.repeats)
    if R > 0:
        rows = []
        for r in range(R):
            df_r, _ = _simulate_once(
                seed=base_seed + 101 + r,
                A0=p["A0"], tau_f=p["tau_f"], noise_log=p["noise_log"],
                delta_start=p["delta_start"], delta_end=p["delta_end"],
                delta_step=p["delta_step"], n_rep=p["n_rep"]
            )
            res_r = check_once(df_r, A_min=p["A_min"], n_boot=n_boot, ci_percent=ci_percent, seed=base_seed + 202 + r)
            rows.append(res_r.iloc[0].to_dict())
        MC = pd.DataFrame(rows)
        summary = pd.DataFrame([{
            "repeats": R,
            "WLS_in_OLS_tau_CI_rate":  float(np.mean(MC["wls_in_OLS_tau_CI"])),
            "Tobit_in_OLS_tau_CI_rate":float(np.mean(MC["tobit_in_OLS_tau_CI"])),
            "WLS_in_OLS_slope_CI_rate":float(np.mean(MC["wls_in_OLS_slope_CI"])),
            "Tobit_in_OLS_slope_CI_rate":float(np.mean(MC["tobit_in_OLS_slope_CI"])),
        }])
        MC.to_csv(args.out_prefix + "_replicates.csv", index=False)
        summary.to_csv(args.out_prefix + "_coverage.csv", index=False)
        print("\nMonte-Carlo coverage summary:")
        print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

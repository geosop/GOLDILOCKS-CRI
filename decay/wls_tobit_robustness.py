#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decay/wls_tobit_robustness.py  •  CRI v0.3-SIM

Reviewer-robust artifacts showing WLS/Tobit consistency with OLS CIs
on the SAME synthetic dataset produced by simulate_decay.py (Box-2a).

Inputs:
  - decay/output/decay_data.csv
  - decay/default_params.yml   (for A_min, n_boot, CI%)

Outputs (always):
  - decay/output/wls_tobit_robustness.csv
  - figures/output/decay_wls_tobit_robustness.png
  - figures/output/decay_wls_tobit_robustness.pdf

Optional Monte-Carlo (if --repeats > 0):
  - decay/output/wls_tobit_replicates.csv
  - decay/output/wls_tobit_coverage.csv
"""
from __future__ import annotations

# stdlib
import os, sys, math, argparse

# third-party
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml

# --- make the repo root importable when running this file by path -------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Use the same fitting helpers as the pipeline for consistency
from decay.fit_decay import _ols_fit, _wls_fit, _tobit_fit, _fit_all

HERE = os.path.dirname(__file__)
OUTD = os.path.join(HERE, "output")
FIGD = os.path.join(os.path.dirname(HERE), "figures", "output")
os.makedirs(OUTD, exist_ok=True)
os.makedirs(FIGD, exist_ok=True)

# Embed TrueType in PDF (no Type 3 fonts)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42


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
    """Local simulator matching Box-2a settings; used only for Monte-Carlo repeats."""
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
    """Nonparametric bootstrap for OLS slope CI (percentile)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    b1s = np.empty(int(B), dtype=float)
    for i in range(int(B)):
        idx = rng.integers(n, size=n)
        _, b1s[i] = _ols_fit(x[idx], y[idx])
    a = (100.0 - ci) / 2.0
    lo = float(np.percentile(b1s, a))
    hi = float(np.percentile(b1s, 100.0 - a))
    return lo, hi


def _b1_to_tau_ms(b1: float) -> float:
    """Convert slope of ln(A) vs Δ (b1) to τ in ms. (Δ in seconds)"""
    if b1 >= 0:
        return float("inf")
    return (-1.0 / b1) * 1000.0


def _fit_all_from_dataset(df: pd.DataFrame, A_min: float, n_boot: int, ci_percent: float, seed: int):
    """Fit OLS/WLS/Tobit on provided aggregated dataset (delta, lnA_pre, se_lnA)."""
    x = df["delta"].astype(float).values
    y = df["lnA_pre"].astype(float).values
    se = df["se_lnA"].astype(float).values

    # Point estimates
    _, b1_ols  = _ols_fit(x, y)
    _, b1_wls  = _wls_fit(x, y, se)
    _, b1_tob  = _tobit_fit(x, y, math.log(A_min))

    tau_ols_ms  = _b1_to_tau_ms(b1_ols)
    tau_wls_ms  = _b1_to_tau_ms(b1_wls)
    tau_tob_ms  = _b1_to_tau_ms(b1_tob)

    # OLS τ CI via the same pipeline helper (bootstrap inside)
    res_df, _ = _fit_all(df, A_min=A_min, n_boot=n_boot, ci_percent=ci_percent, seed=seed)
    ols_ci_lo_ms = float(res_df["ci_lo_ms"].iloc[0])
    ols_ci_hi_ms = float(res_df["ci_hi_ms"].iloc[0])

    # OLS slope CI (for the “slope within CI” claim)
    slo_lo, slo_hi = _bootstrap_slope_ci_ols(x, y, B=n_boot, ci=ci_percent, seed=seed)

    out = {
        "b1_ols": b1_ols, "b1_wls": b1_wls, "b1_tobit": b1_tob,
        "slope_CI_lo_OLS": slo_lo, "slope_CI_hi_OLS": slo_hi,
        "tau_ols_ms": tau_ols_ms, "tau_wls_ms": tau_wls_ms, "tau_tobit_ms": tau_tob_ms,
        "ols_ci_lo_ms": ols_ci_lo_ms, "ols_ci_hi_ms": ols_ci_hi_ms,
        "wls_in_OLS_tau_CI": (ols_ci_lo_ms <= tau_wls_ms <= ols_ci_hi_ms),
        "tobit_in_OLS_tau_CI": (ols_ci_lo_ms <= tau_tob_ms <= ols_ci_hi_ms),
        "wls_in_OLS_slope_CI": (slo_lo <= b1_wls <= slo_hi),
        "tobit_in_OLS_slope_CI": (slo_lo <= b1_tob <= slo_hi),
    }
    return out

def _plot_tau_panel(out: dict, out_png: str, out_pdf: str):
    """Small visual: OLS 95% CI bar and markers for OLS/WLS/Tobit τ."""
    fig, ax = plt.subplots(figsize=(6.0, 2.5))

    y = 0.0
    # OLS CI bar (thick horizontal line)
    ax.hlines(
        y, out["ols_ci_lo_ms"], out["ols_ci_hi_ms"],
        linewidth=4, color="C0", label="OLS 95% sim. bootstrap CI", zorder=1
    )
    
    # marker-only entries (no connecting line in legend)
    ax.plot([out["tau_ols_ms"]],   [y],          marker="o", ms=7, color="C0", linestyle="None",
            label=f"OLS {out['tau_ols_ms']:.1f} ms", zorder=3)
    ax.plot([out["tau_wls_ms"]],   [y + 0.08],   marker="^", ms=7, color="C1", linestyle="None",
            label=f"WLS {out['tau_wls_ms']:.1f} ms", zorder=3)
    ax.plot([out["tau_tobit_ms"]], [y - 0.08],   marker="s", ms=7, color="C2", linestyle="None",
            label=f"Tobit {out['tau_tobit_ms']:.1f} ms", zorder=3)

    # --- single-line legend above the axes ---
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles, labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.0),  # span full axes width above
        mode="expand",                         # distribute columns across width
        ncol=4,                                # one column per entry -> one row
        frameon=False,
        fontsize=8,
        handlelength=1.2, handletextpad=0.5,
        columnspacing=0.9, labelspacing=0.2,
        borderaxespad=0.0
    )



    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf,        bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="WLS/Tobit robustness vs OLS CIs on the current synthetic dataset.")
    ap.add_argument("--data", default=os.path.join(HERE, "output", "decay_data.csv"),
                    help="Aggregated dataset with columns: delta, lnA_pre, se_lnA")
    ap.add_argument("--out-csv", default=os.path.join(HERE, "output", "wls_tobit_robustness.csv"),
                    help="Output CSV path")
    ap.add_argument("--out-png", default=os.path.join(FIGD, "decay_wls_tobit_robustness.png"),
                    help="Output figure (PNG) path")
    ap.add_argument("--out-pdf", default=os.path.join(FIGD, "decay_wls_tobit_robustness.pdf"),
                    help="Output figure (PDF) path")
    ap.add_argument("--repeats", type=int, default=0, help="Monte-Carlo repeats over NEW datasets (0=off)")
    ap.add_argument("--seed", type=int, default=None, help="Base RNG seed (default from YAML)")
    ap.add_argument("--n-boot", type=int, default=None, help="Bootstrap B for CIs (default from YAML)")
    ap.add_argument("--ci", type=float, default=None, help="CI percent (default from YAML)")
    args = ap.parse_args()

    p = _load_params()
    base_seed   = p["seed"] if args.seed is None else int(args.seed)
    n_boot      = p["n_boot"] if args.n_boot is None else int(args.n_boot)
    ci_percent  = p["ci_percent"] if args.ci is None else float(args.ci)
    A_min       = p["A_min"]

    # ------- Single-dataset check on the SAME synthetic data already produced
    if not os.path.isfile(args.data):
        raise FileNotFoundError(
            f"Expected dataset not found: {args.data}\nRun decay/simulate_decay.py first."
        )

    df = pd.read_csv(args.data)
    needed = {"delta", "lnA_pre", "se_lnA"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{args.data} must contain columns {needed}, got {df.columns.tolist()}")

    out = _fit_all_from_dataset(df, A_min=A_min, n_boot=n_boot, ci_percent=ci_percent, seed=base_seed)
    pd.DataFrame([out]).to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")
    print(pd.DataFrame([out]).to_string(index=False))

    _plot_tau_panel(out, args.out_png, args.out_pdf)
    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_pdf}")

    # ------- Optional Monte-Carlo coverage over NEW datasets (fresh sims)
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
            out_r = _fit_all_from_dataset(
                df_r, A_min=A_min, n_boot=n_boot, ci_percent=ci_percent, seed=base_seed + 202 + r
            )
            rows.append(out_r)
        MC = pd.DataFrame(rows)
        MC_path = os.path.join(OUTD, "wls_tobit_replicates.csv")
        MC.to_csv(MC_path, index=False)

        summary = pd.DataFrame([{
            "repeats": R,
            "WLS_in_OLS_tau_CI_rate":     float(np.mean(MC["wls_in_OLS_tau_CI"])),
            "Tobit_in_OLS_tau_CI_rate":   float(np.mean(MC["tobit_in_OLS_tau_CI"])),
            "WLS_in_OLS_slope_CI_rate":   float(np.mean(MC["wls_in_OLS_slope_CI"])),
            "Tobit_in_OLS_slope_CI_rate": float(np.mean(MC["tobit_in_OLS_slope_CI"])),
        }])
        SUM_path = os.path.join(OUTD, "wls_tobit_coverage.csv")
        summary.to_csv(SUM_path, index=False)
        print("\nMonte-Carlo coverage summary:")
        print(summary.to_string(index=False))
        print(f"Wrote {MC_path}")
        print(f"Wrote {SUM_path}")


if __name__ == "__main__":
    main()

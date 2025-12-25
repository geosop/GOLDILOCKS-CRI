# -*- coding: utf-8 -*-
"""
figures/make_tomography_figure.py  •  CRI Box-2(c) — rigor/provenance-hardened

Robustness upgrades (aligned with the hardened logistic pipeline):
  1) Config override via env var CRI_QPT_CONFIG (relative paths resolved repo-root first, then qpt/).
  2) Strict provenance: require qpt/output/run_manifest.json and enforce run_hash consistency against:
       - YAML(qpt) parameters (canonical hash)
       - qpt/output/qpt_sim_data.npz fields (run_hash, params_path) if present
     (can be relaxed with --allow-missing-run-hash).
  3) κ0 estimation: OLS through origin + bootstrap CI from LEFT panel only (no peeking at right).
  4) Right panel uses normalized x by default:  𝜆̃ = λ_env / κ̂0  (theory slope = 1).
     Optionally --abs-x uses absolute λ_env with theory R = λ_env/κ̂0.
  5) Empirical CI envelope: uses R_ci_low/high from NPZ; if missing, fails by default
     (can be relaxed with --allow-missing-ci).
  6) Provenance footer on the figure and a stats CSV stamped with run_hash + params_path.

Outputs (default stem "Box2c_rate_refined"):
  figures/output/Box2c_rate_refined.pdf
  figures/output/Box2c_rate_refined.png
  figures/output/Box2c_rate_refined_stats.csv
"""
from __future__ import annotations

import os
import json
import yaml
import argparse
import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:
    pd = None


# ---------------- Matplotlib defaults (portable) -----------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "legend.fontsize": 6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
LIGHT_GRAY = "#f2f2f2"
TEAL_DOT   = "#136F63"
TEAL_BAND  = "#B9E3D6"


# ---------------- Utilities ---------------------------------------------------
def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return float(default) if default is not None else None


def _canonical_json_bytes(obj: Any) -> bytes:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def _compute_run_hash(qpt_params: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(qpt_params)).hexdigest()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _npz_get_str(dat: np.lib.npyio.NpzFile, key: str) -> Optional[str]:
    if key not in dat.files:
        return None
    v = dat[key]
    try:
        if isinstance(v, np.ndarray) and v.shape == ():
            return str(v.item())
        if isinstance(v, np.ndarray) and v.size == 1:
            return str(v.reshape(-1)[0].item())
        return str(v)
    except Exception:
        return None


def _resolve_qpt_config_path(config_path: Optional[str]) -> str:
    """
    Resolve QPT config path deterministically.
    Relative paths are resolved against repo root first, then qpt/.

    Priority:
      1) explicit argument
      2) env var CRI_QPT_CONFIG
      3) qpt/default_params.yml (fallback)
    """
    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, os.pardir))
    qpt_dir = os.path.join(repo, "qpt")

    if config_path is None:
        config_path = os.getenv("CRI_QPT_CONFIG")

    if config_path is None:
        return os.path.join(qpt_dir, "default_params.yml")

    candidates = [
        config_path,
        os.path.join(repo, config_path),
        os.path.join(qpt_dir, config_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        f"QPT config not found: {config_path}\n"
        f"Tried:\n  - {candidates[0]}\n  - {candidates[1]}\n  - {candidates[2]}"
    )


def load_qpt_params(config_path: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    cfg_path = _resolve_qpt_config_path(config_path)
    with open(cfg_path, "r", encoding="utf-8-sig", errors="replace") as f:
        cfg = yaml.safe_load(f) or {}
    p = cfg.get("qpt", cfg)
    if not isinstance(p, dict):
        raise ValueError("QPT YAML did not parse to a dict under key 'qpt'.")
    return p, cfg_path


def _enforce_manifest_and_hashes(
    qpt_out: str,
    params: Dict[str, Any],
    params_path: str,
    sim_npz: np.lib.npyio.NpzFile,
    allow_missing_run_hash: bool = False,
) -> str:
    """
    Enforce:
      - qpt/output/run_manifest.json exists and contains run_hash
      - YAML(qpt) hash == manifest hash
      - NPZ run_hash (if present) == manifest hash
      - NPZ params_path (if present) matches resolved params_path

    Returns: run_hash (or "<unset>" if allow_missing_run_hash and manifest missing).
    """
    man_path = os.path.join(qpt_out, "run_manifest.json")
    if not os.path.exists(man_path):
        if allow_missing_run_hash:
            return "<unset>"
        raise RuntimeError(
            "Missing qpt/output/run_manifest.json.\n"
            "Fix: run qpt/qpt_simulation.py (hardened version) before make_tomography_figure.py,\n"
            "or pass --allow-missing-run-hash for a non-strict local run."
        )

    manifest = _read_json(man_path)
    if "run_hash" not in manifest:
        if allow_missing_run_hash:
            return "<unset>"
        raise RuntimeError("run_manifest.json missing key 'run_hash'.")

    run_hash = str(manifest["run_hash"])
    yaml_hash = _compute_run_hash(params)
    if yaml_hash != run_hash:
        raise RuntimeError(
            "YAML(qpt) dict hash does not match run_manifest.json.\n"
            f"  YAML hash:      {yaml_hash}\n"
            f"  manifest hash:  {run_hash}\n"
            "Fix: delete qpt/output/* and regenerate from the intended YAML."
        )

    npz_hash = _npz_get_str(sim_npz, "run_hash")
    if npz_hash is not None and npz_hash != run_hash:
        raise RuntimeError(
            "run_hash mismatch between qpt_sim_data.npz and run_manifest.json.\n"
            f"  npz.run_hash={npz_hash}\n"
            f"  manifest.run_hash={run_hash}\n"
            "Fix: delete qpt/output/* and regenerate."
        )

    npz_path = _npz_get_str(sim_npz, "params_path")
    if npz_path is not None and os.path.abspath(npz_path) != os.path.abspath(params_path):
        raise RuntimeError(
            "params_path mismatch between qpt_sim_data.npz and resolved config path.\n"
            f"  npz.params_path={npz_path}\n"
            f"  resolved params_path={params_path}\n"
            "Fix: delete qpt/output/* and regenerate."
        )

    return run_hash


def fit_kappa0_from_pops(t_s, gamma_fwd, gamma_b_vals, pops):
    """
    LEFT PANEL ONLY:
      ln P = -κ0 * x,  x = 0.5*(γ_fwd+γ_b)*t  → OLS through origin.
    Returns: (k_hat, se_k, n_obs, X, Y) where X,Y are the stacked design/response.
    """
    t_s = np.asarray(t_s, float)
    pops = np.asarray(pops, float)
    gb = np.asarray(gamma_b_vals, float)

    X_list, Y_list = [], []
    for j, g in enumerate(gb):
        x = 0.5 * (gamma_fwd + g) * t_s
        y = np.log(np.clip(pops[j], 1e-12, None))
        X_list.append(x)
        Y_list.append(y)

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)

    Sxx = float(np.dot(X, X))
    Sxy = float(np.dot(X, Y))
    k_hat = -Sxy / Sxx

    resid = Y + k_hat * X
    n = int(X.size)
    dof = max(n - 1, 1)
    sigma2 = float(np.dot(resid, resid)) / dof
    se_k = float(np.sqrt(sigma2 / Sxx))
    return float(k_hat), float(se_k), n, X, Y


def bootstrap_kappa0_ci(X, Y, seed: int, n_boot: int, ci_percent: float) -> Tuple[float, float]:
    """
    Bootstrap κ0 from LEFT PANEL ONLY by resampling (X_i, Y_i) pairs with replacement.
    Percentile CI.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = X.size
    if n < 5 or n_boot <= 1:
        return (np.nan, np.nan)

    rng = np.random.default_rng(int(seed) + 12345)
    kh = np.empty(int(n_boot), dtype=float)

    for b in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        Xb = X[idx]
        Yb = Y[idx]
        Sxx = float(np.dot(Xb, Xb))
        Sxy = float(np.dot(Xb, Yb))
        kh[b] = -Sxy / Sxx if Sxx > 0 else np.nan

    kh = kh[np.isfinite(kh)]
    if kh.size < max(20, int(0.2 * n_boot)):
        return (np.nan, np.nan)

    alpha = (100.0 - float(ci_percent)) / 100.0
    lo = np.percentile(kh, 100.0 * alpha / 2.0)
    hi = np.percentile(kh, 100.0 * (1.0 - alpha / 2.0))
    return float(lo), float(hi)


def ols_slope_intercept(x, y):
    """OLS y=a+bx; returns (a,b,se_a,se_b,t_b,t_b_vs1,df,R2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    xm = x.mean()
    ym = y.mean()
    Sxx = np.sum((x - xm) ** 2)
    Sxy = np.sum((x - xm) * (y - ym))
    b = Sxy / Sxx
    a = ym - b * xm
    yhat = a + b * x
    resid = y - yhat
    df = max(n - 2, 1)
    s2 = np.sum(resid ** 2) / df
    se_b = np.sqrt(s2 / Sxx)
    se_a = np.sqrt(s2 * (1.0 / n + xm ** 2 / Sxx))
    t_b = b / se_b
    t_b_vs1 = (b - 1.0) / se_b
    R2 = 1.0 - (np.sum(resid ** 2) / np.sum((y - ym) ** 2))
    return float(a), float(b), float(se_a), float(se_b), float(t_b), float(t_b_vs1), int(df), float(R2)


def design_se_slope(x_max: float, n_levels: int, sigma_R: float) -> float:
    """
    Approximate SE of slope for evenly spaced x in [0, x_max] with homoscedastic noise sigma_R.
    Uses Sxx ≈ n*x_max^2/12.
    """
    n = max(int(n_levels), 2)
    L = max(float(x_max), 1e-12)
    return float(sigma_R / (L * np.sqrt(n / 12.0)))


# ---------------- Main --------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Generate CRI Box-2(c) with κ0 calibrated from left panel + strict provenance.")
    ap.add_argument("--abs-x", action="store_true", help="Use absolute λ_env on right (default: normalized by κ̂0).")
    ap.add_argument("--no-annot", action="store_true", help="Suppress stats annotations.")
    ap.add_argument("--dpi", type=int, default=int(os.getenv("CRI_DPI", 1200)), help="PNG export DPI (PDF is vector).")

    # Provenance strictness
    ap.add_argument("--allow-missing-run-hash", action="store_true",
                    help="Do not fail if run_manifest/run_hash checks are missing (NOT recommended for CI).")
    ap.add_argument("--allow-missing-ci", action="store_true",
                    help="Fallback to a fixed CI band if R_ci_low/high are missing (NOT recommended for RSOS).")

    # Output naming
    ap.add_argument("--out-stem", default=os.getenv("CRI_BOX2C_STEM", "Box2c_rate_refined"),
                    help="Output stem placed under figures/output/ as PDF/PNG/CSV.")

    # Panel label controls (figure coordinates)
    ap.add_argument("--panel-label", default=os.getenv("CRI_PANEL_LABEL_C", "(c)"))
    ap.add_argument("--panel-x", type=float, default=float(os.getenv("CRI_PANEL_X_C", 0.008)))
    ap.add_argument("--panel-y", type=float, default=float(os.getenv("CRI_PANEL_Y_C", 0.975)))

    return ap.parse_args()


def main():
    args = parse_args()
    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, os.pardir))
    qpt_dir = os.path.join(repo, "qpt")
    qpt_out = os.path.join(qpt_dir, "output")
    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load params (with override) ----------------------------------------
    p, params_path = load_qpt_params(None)
    gamma_fwd = _safe_float(p.get("gamma_fwd", p.get("gamma_f", 1.0)), 1.0)
    left_xlim_ms = _safe_float(p.get("left_panel_tmax_ms", 200.0), 200.0)
    seed = int(p.get("seed", int(os.getenv("CRI_SEED", 52))))

    # κ0 bootstrap settings
    n_boot_k = int(p.get("n_bootstrap", 1000))
    ci_percent = float(p.get("ci_percent", 95))

    # ---- Load simulated data -------------------------------------------------
    sim_path = os.path.join(qpt_out, "qpt_sim_data.npz")
    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"Missing {sim_path}. Run qpt/qpt_simulation.py first.")

    sim = np.load(sim_path, allow_pickle=True)
    t_s          = sim["t"]
    gamma_b_vals = sim["gamma_b_vals"]
    pops         = sim["pops"]
    lambda_env   = sim["lambda_env"]
    R_mean       = sim["R_mean"]

    R_low  = sim["R_ci_low"]  if "R_ci_low"  in sim.files else None
    R_high = sim["R_ci_high"] if "R_ci_high" in sim.files else None

    # ---- Enforce manifest + run_hash ----------------------------------------
    run_hash = _enforce_manifest_and_hashes(
        qpt_out=qpt_out,
        params=p,
        params_path=params_path,
        sim_npz=sim,
        allow_missing_run_hash=bool(args.allow_missing_run_hash),
    )

    # ---- Step 1: κ0 from left panel (no peeking at right) -------------------
    k_hat, se_k, n_obs, X, Y = fit_kappa0_from_pops(t_s, gamma_fwd, gamma_b_vals, pops)
    k_lo, k_hi = bootstrap_kappa0_ci(X, Y, seed=seed, n_boot=n_boot_k, ci_percent=ci_percent)

    # ---- Step 2: normalized abscissa unless --abs-x -------------------------
    if args.abs_x:
        x_right = np.asarray(lambda_env, float)
        x_label = r"$\lambda_{\mathrm{env}}$ (s$^{-1}$)"
        theory_label = r"Theory $R=\lambda_{\mathrm{env}}/\hat{\kappa}_0$"
        theory_y = np.asarray(lambda_env, float) / float(k_hat)
        slope_null_label = "0"
    else:
        x_right = np.asarray(lambda_env, float) / float(k_hat)
        x_label = r"$\tilde{\lambda}\equiv\lambda_{\mathrm{env}}/\hat{\kappa}_0$ (–)"
        theory_label = r"Theory $R=\tilde{\lambda}$ (slope $=1$)"
        theory_y = np.asarray(x_right, float)
        slope_null_label = "1"

    # ---- Step 3: empirical CI envelope (strict by default) ------------------
    has_ci = (R_low is not None) and (R_high is not None)
    if not has_ci:
        if not args.allow_missing_ci:
            raise RuntimeError(
                "Missing R_ci_low/R_ci_high in qpt_sim_data.npz.\n"
                "Fix: update qpt/qpt_simulation.py to save these arrays, then regenerate.\n"
                "Or pass --allow-missing-ci (not recommended for RSOS)."
            )
        # Fallback only for local debugging
        R_low = np.asarray(R_mean, float) - 0.05
        R_high = np.asarray(R_mean, float) + 0.05

    # ---- Step 4: inference on right -----------------------------------------
    a, b, se_a, se_b, t_b, t_b_vs1, df, R2 = ols_slope_intercept(x_right, R_mean)

    # Design SE for slope (use x-range actually plotted)
    x_max = float(np.nanmax(x_right))
    sigma_R = _safe_float(p.get("noise_R", 0.05), 0.05)
    n_levels = int(p.get("n_lambda_env", len(np.asarray(lambda_env).ravel())))
    se_b_design = design_se_slope(x_max=x_max, n_levels=n_levels, sigma_R=sigma_R)

    # ---------------- Figure --------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(180 / 25.4, 60 / 25.4), constrained_layout=True)

    # Panel label outside axes (black, configurable)
    lbl = str(args.panel_label)
    lbl_math = rf"$({lbl.strip('()')})$" if (lbl.startswith("(") and lbl.endswith(")")) else lbl
    fig.text(
        float(args.panel_x), float(args.panel_y),
        lbl_math,
        transform=fig.transFigure,
        ha="left", va="top", fontsize=9, color="black",
        clip_on=False, zorder=10
    )

    # LEFT: P(t) with units
    t_ms = 1000.0 * np.asarray(t_s, float)
    for gb, curve in zip(gamma_b_vals, pops):
        ax1.plot(t_ms, curve, label=rf"$\gamma_b={float(gb):.1f}$")
    ax1.set_xlim(0.0, left_xlim_ms)
    ax1.set_xlabel(r"Time $t$ (ms)")
    ax1.set_ylabel(r"Population $P(t)$ (–)")
    ax1.set_title(r"$P(t)=\exp\!\left[-(\kappa_0/2)\,(\gamma_{\mathrm{fwd}}+\gamma_b)\,t\right]$")

    leg1 = ax1.legend(
        loc="upper right", bbox_to_anchor=(0.98, 0.70),
        frameon=True, fancybox=True, borderaxespad=0.0,
        handlelength=1.2, handletextpad=0.35
    )
    fr1 = leg1.get_frame()
    fr1.set_facecolor(LIGHT_GRAY)
    fr1.set_edgecolor("0.80")
    fr1.set_linewidth(0.6)
    ax1.grid(alpha=0.25, linestyle="--")

    if not args.no_annot:
        ci_line = ""
        if np.isfinite(k_lo) and np.isfinite(k_hi):
            ci_line = rf"(95% bootstrap: {k_lo:.2f}–{k_hi:.2f})"
        else:
            # fall back to OLS normal approx if bootstrap failed
            ci_line = rf"(OLS 95%: {k_hat-1.96*se_k:.2f}–{k_hat+1.96*se_k:.2f})"

        ax1.text(
            0.28, 0.95,
            rf"$\hat{{\kappa}}_0={k_hat:.2f}\ \mathrm{{s}}^{{-1}}$ {ci_line}" "\n"
            rf"$\gamma_{{\mathrm{{fwd}}}}={gamma_fwd:.1f}$, $n={n_obs}$ pts",
            transform=ax1.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8", linewidth=0.6),
            fontsize=6.5, linespacing=0.95
        )

    # RIGHT: R vs x (normalized or absolute)
    order = np.argsort(x_right)
    xr = np.asarray(x_right, float)[order]
    Rl = np.asarray(R_low, float)[order]
    Rh = np.asarray(R_high, float)[order]
    Rm = np.asarray(R_mean, float)[order]
    thy = np.asarray(theory_y, float)[order]

    ax2.fill_between(xr, Rl, Rh, color=TEAL_BAND, alpha=0.9, edgecolor="none",
                     label=f"{int(ci_percent)}% sim. bootstrap band")
    ax2.plot(xr, thy, ls="--", color="grey", label=theory_label)
    ax2.scatter(x_right, R_mean, s=28, color=TEAL_DOT, edgecolors="black", linewidths=0.4,
                label=r"Sim. bootstrap $R$")
    ax2.vlines(x_right, R_low, R_high, colors=TEAL_DOT, alpha=0.35, linewidth=0.8)

    ax2.set_xlabel(x_label)
    ax2.set_ylabel(r"Jump-weight ratio $R$ (–)")

    title_rhs = r"$R$ vs. $\lambda_{\mathrm{env}}$" if args.abs_x else r"$R$ vs. $\tilde{\lambda}$"
    ax2.set_title(title_rhs + rf"  $(\hat{{\kappa}}_0={k_hat:.2f}\ \mathrm{{s}}^{{-1}})$")

    xpad = 0.02 * (xr.max() - xr.min() if xr.max() > xr.min() else 1.0)
    ax2.set_xlim(xr.min() - xpad, xr.max() + xpad)
    y_top = max(float(Rh.max()), float(thy.max())) + 0.03
    ax2.set_ylim(0.0, max(0.22, y_top))

    leg2 = ax2.legend(
        loc="lower right", frameon=True, fancybox=True,
        labelspacing=0.2, borderpad=0.25, handlelength=1.2, handletextpad=0.35, columnspacing=0.6
    )
    fr2 = leg2.get_frame()
    fr2.set_facecolor(LIGHT_GRAY)
    fr2.set_edgecolor("0.80")
    fr2.set_linewidth(0.6)
    ax2.grid(alpha=0.25, linestyle="--")

    if not args.no_annot:
        if args.abs_x:
            t_line = rf"$t(\hat\beta\!=\!0)={t_b:.2f},\ \mathrm{{df}}={df}$"
        else:
            t_line = rf"$t(\hat\beta\!=\!0)={t_b:.2f},\ t(\hat\beta\!=\!1)={t_b_vs1:.2f},\ \mathrm{{df}}={df}$"

        stats_txt = (
            rf"OLS: $\hat\beta={b:.3f}\pm{1.96*se_b:.3f}$, "
            rf"$\hat\alpha={a:.3f}\pm{1.96*se_a:.3f}$;  $R^2={R2:.2f}$" "\n"
            + t_line
            + f"\nDesign SE(β) ≈ {se_b_design:.3f}  (H0: slope={slope_null_label})"
        )
        ax2.text(
            0.02, 0.98, stats_txt,
            transform=ax2.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="none", edgecolor="black", linewidth=0.5),
            fontsize=6.2, zorder=6, linespacing=0.95
        )

    # ---- Provenance footer ---------------------------------------------------
    run_hash_short = run_hash if run_hash in ("<unset>", "") else run_hash[:12]
    footer = (
        f"run_hash={run_hash_short}  seed={seed}  gamma_fwd={gamma_fwd:.2f}  "
        f"kappa0_hat={k_hat:.3f} s^-1  qpt_cfg={os.path.basename(params_path)}"
    )
    fig.text(0.01, 0.01, footer, transform=fig.transFigure,
             ha="left", va="bottom", fontsize=6.2, color="0.35", clip_on=False)

    # ---- Save + stats CSV ----------------------------------------------------
    out_pdf = os.path.join(out_dir, f"{args.out_stem}.pdf")
    out_png = os.path.join(out_dir, f"{args.out_stem}.png")
    out_csv = os.path.join(out_dir, f"{args.out_stem}_stats.csv")

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    # Stats CSV (optional dependency on pandas)
    stats_row = {
        "run_hash": run_hash,
        "params_path": os.path.abspath(params_path),
        "abs_x": bool(args.abs_x),
        "seed": int(seed),
        "gamma_fwd": float(gamma_fwd),
        "kappa0_hat_s^-1": float(k_hat),
        "kappa0_se_ols_s^-1": float(se_k),
        "kappa0_ci_lo_boot_s^-1": float(k_lo) if np.isfinite(k_lo) else np.nan,
        "kappa0_ci_hi_boot_s^-1": float(k_hi) if np.isfinite(k_hi) else np.nan,
        "kappa0_n_obs": int(n_obs),
        "ols_intercept_a": float(a),
        "ols_slope_b": float(b),
        "ols_se_a": float(se_a),
        "ols_se_b": float(se_b),
        "ols_t_b": float(t_b),
        "ols_t_b_vs1": float(t_b_vs1),
        "ols_df": int(df),
        "ols_R2": float(R2),
        "design_se_b": float(se_b_design),
        "ci_percent": float(ci_percent),
    }
    if pd is not None:
        pd.DataFrame([stats_row]).to_csv(out_csv, index=False)
    else:
        # minimal CSV writer if pandas absent
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(stats_row.keys()) + "\n")
            f.write(",".join("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in stats_row.values()) + "\n")

    print(f"Saved: {out_pdf} and {out_png}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

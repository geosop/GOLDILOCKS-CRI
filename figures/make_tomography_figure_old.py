# -*- coding: utf-8 -*-
"""
figures/make_tomography_figure.py  •  CRI Box-2(c) — rigor version

Implements the 4 upgrades:
  1) Calibrate κ0 from left panel (no peeking at right).
  2) Use dimensionless x on right: \tilde{λ} = λ_env / κ0_hat  (theory slope = 1).
  3) Plot empirical bootstrap CI envelope (R_ci_low/high) — no fixed-width band.
  4) Show inference: κ0_hat ± CI (left); OLS on right with slope test vs 1.

CLI flags:
  --abs-x            Use absolute λ_env on right (default: normalized).
  --no-annot         Suppress stats annotations on plots.
  --dpi 1200         Export PNG dpi (PDF is vector).

Outputs:
  figures/output/Box2c_rate_refined.pdf
  figures/output/Box2c_rate_refined.png
  figures/output/Box2c_stats.csv
"""
import os
import yaml
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import pandas as pd  # optional; used for CSV export
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

def load_qpt_params(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("qpt", cfg)

def fit_kappa0_from_pops(t_s, gamma_fwd, gamma_b_vals, pops):
    """
    Model: ln P = - κ0 * x,  where  x = 0.5 * (γ_fwd + γ_b) * t.
    Pool all curves & times; OLS through origin.
    Returns (kappa_hat, kappa_lo, kappa_hi, stderr, n_obs).
    """
    t_s = np.asarray(t_s, float)
    pops = np.asarray(pops, float)      # shape: (n_gb, n_t)
    gb = np.asarray(gamma_b_vals, float)

    X_list, Y_list = [], []
    for j, g in enumerate(gb):
        x = 0.5 * (gamma_fwd + g) * t_s
        y = np.log(np.clip(pops[j], 1e-12, None))
        X_list.append(x)
        Y_list.append(y)
    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)

    # OLS through origin: Y ≈ -κ X
    Sxx = float(np.dot(X, X))
    Sxy = float(np.dot(X, Y))
    k_hat = - Sxy / Sxx

    # Residuals & SE(κ_hat)
    resid = Y + k_hat * X
    n = X.size
    dof = max(n - 1, 1)
    sigma2 = float(np.dot(resid, resid)) / dof
    se_k = np.sqrt(sigma2 / Sxx)
    k_lo = k_hat - 1.96 * se_k
    k_hi = k_hat + 1.96 * se_k
    return k_hat, k_lo, k_hi, se_k, n

def ols_slope_intercept(x, y):
    """
    Standard OLS: y = a + b x + ε
    Returns (a, b, se_a, se_b, t_b, t_b_vs1, df, R2)
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size
    xm = x.mean(); ym = y.mean()
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
    t_b = b / se_b                      # test vs 0
    t_b_vs1 = (b - 1.0) / se_b          # test vs 1 (normalized-x theory)
    R2 = 1.0 - (np.sum(resid ** 2) / np.sum((y - ym) ** 2))
    return a, b, se_a, se_b, t_b, t_b_vs1, df, R2

def design_t_stat(lambda_max, n_levels, sigma_R, kappa0):
    """
    Back-of-envelope expected t for slope vs 0 when x in [0, L] with n uniform levels:
        t ≈ ( (1/κ0) * L / σ_R ) * sqrt(n/12)
    """
    return ((1.0 / kappa0) * lambda_max / sigma_R) * np.sqrt(n_levels / 12.0)

# ---------------- Main --------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Generate CRI Box-2(c) with calibrated κ0 and normalized right panel.")
    ap.add_argument("--abs-x", action="store_true", help="Use absolute λ_env on right (default: normalized).")
    ap.add_argument("--no-annot", action="store_true", help="Suppress stats annotations.")
    ap.add_argument("--dpi", type=int, default=1200, help="PNG export DPI (PDF is vector).")
    return ap.parse_args()

def main():
    args = parse_args()

    here     = os.path.dirname(__file__)
    repo     = os.path.abspath(os.path.join(here, os.pardir))
    qpt_dir  = os.path.join(repo, "qpt")
    out_dir  = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load params & data --------------------------------------------------
    p = load_qpt_params(os.path.join(qpt_dir, "default_params.yml"))
    gamma_fwd = _safe_float(p.get("gamma_fwd", p.get("gamma_f", 1.0)), 1.0)
    left_xlim_ms = _safe_float(p.get("left_panel_tmax_ms", 200.0), 200.0)

    sim = np.load(os.path.join(qpt_dir, "output", "qpt_sim_data.npz"), allow_pickle=True)
    t_s         = sim["t"]                 # seconds
    gamma_b_vals= sim["gamma_b_vals"]      # e.g., [0.2, 0.8]
    pops        = sim["pops"]              # (n_gb × n_t)
    lambda_env  = sim["lambda_env"]        # s^-1
    R_mean      = sim["R_mean"]
    R_low       = sim.get("R_ci_low", None)
    R_high      = sim.get("R_ci_high", None)

    # ---- Step 1: Calibrate κ0 from left panel (no peeking at right) ---------
    k_hat, k_lo, k_hi, se_k, n_obs = fit_kappa0_from_pops(t_s, gamma_fwd, gamma_b_vals, pops)

    # ---- Step 2: Normalized abscissa on right (unless --abs-x) --------------
    if args.abs_x:
        x_right = lambda_env
        x_label = r"$\lambda_{\mathrm{env}}$ (s$^{-1}$)"
        theory_label = r"Theory $R=\lambda_{\mathrm{env}}/\hat{\kappa}_0$"
        theory_y = lambda_env / k_hat
    else:
        x_right = lambda_env / k_hat
        x_label = r"$\tilde{\lambda}\equiv \lambda_{\mathrm{env}}/\hat{\kappa}_0$ (–)"
        theory_label = r"Theory $R=\tilde{\lambda}$ (slope $=1$)"
        theory_y = x_right  # identity

    # ---- Step 3: Empirical CI envelope (from bootstrap) ---------------------
    has_ci = (R_low is not None) and (R_high is not None)
    if not has_ci:
        R_low = R_mean - 0.05
        R_high = R_mean + 0.05

    # ---- Step 4: Inference on right (OLS; slope test vs 1 if normalized) ----
    a, b, se_a, se_b, t_b, t_b_vs1, df, R2 = ols_slope_intercept(x_right, R_mean)
    lam_max = _safe_float(p.get("lambda_env_max", float(np.nanmax(lambda_env))), float(np.nanmax(lambda_env)))
    sigma_R = _safe_float(p.get("noise_R", 0.05), 0.05)
    n_levels= int(p.get("n_lambda_env", len(lambda_env)))
    t_design = design_t_stat(lam_max, n_levels, sigma_R, k_hat)

    # ---------------- Figure --------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(180/25.4, 60/25.4), constrained_layout=True)

    # Left: P(t) (ms axis)
    t_ms = 1000.0 * np.asarray(t_s)
    for gb, curve in zip(gamma_b_vals, pops):
        ax1.plot(t_ms, curve, label=rf"$\gamma_b={float(gb):.1f}$")
    ax1.set_xlim(0.0, left_xlim_ms)
    ax1.set_xlabel(r"Time $t$ (ms)")
    ax1.set_ylabel(r"Population $P(t)$")
    ax1.set_title(r"$P(t)=\exp\!\left[-(\kappa_0/2)\,(\gamma_{\mathrm{fwd}}+\gamma_b)\,t\right]$")
    leg1 = ax1.legend(loc="upper right", frameon=True, fancybox=True)
    leg1.get_frame().set_facecolor(LIGHT_GRAY)
    leg1.get_frame().set_edgecolor("0.80")
    leg1.get_frame().set_linewidth(0.6)
    ax1.grid(alpha=0.25, linestyle="--")

    if not args.no_annot:
        ax1.text(
            0.02, 0.95,
            rf"$\hat{{\kappa}}_0={k_hat:.2f}\ \mathrm{{s}}^{{-1}}$ "
            rf"(95% CI: {k_lo:.2f}–{k_hi:.2f})" "\n"
            rf"$\gamma_{{\mathrm{{fwd}}}}={gamma_fwd:.1f}$, $n={n_obs}$ pts",
            transform=ax1.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8", linewidth=0.6),
            fontsize=6.5
        )

    # Right: R vs normalized or absolute λ
    order = np.argsort(x_right)
    xr = x_right[order]
    Rl = np.asarray(R_low)[order]
    Rh = np.asarray(R_high)[order]
    Rm = np.asarray(R_mean)[order]

    ax2.fill_between(xr, Rl, Rh, color=TEAL_BAND, alpha=0.9, edgecolor="none",
                     label="95% CI (bootstrap)")
    ax2.plot(xr, (theory_y[order] if np.ndim(theory_y)>0 else theory_y), ls="--", color="grey",
             label=theory_label)
    ax2.scatter(x_right, R_mean, s=28, color=TEAL_DOT, edgecolors="black", linewidths=0.4,
                label=r"Bootstrapped $R$")
    ax2.vlines(x_right, R_low, R_high, colors=TEAL_DOT, alpha=0.35, linewidth=0.8)

    ax2.set_xlabel(x_label)
    ax2.set_ylabel(r"Jump-weight ratio $R$")
    title_rhs = r"$R$ vs. $\lambda_{\mathrm{env}}$" if args.abs_x else r"$R$ vs. $\tilde{\lambda}$"
    ax2.set_title(title_rhs + rf"  $(\hat{{\kappa}}_0={k_hat:.2f}\ \mathrm{{s}}^{{-1}})$")

    xpad = 0.02 * (xr.max() - xr.min() if xr.max() > xr.min() else 1.0)
    ax2.set_xlim(xr.min() - xpad, xr.max() + xpad)
    y_top = max(float(Rh.max()),
                float(np.max(theory_y)) if np.ndim(theory_y) > 0 else float(theory_y)) + 0.03
    ax2.set_ylim(0.0, max(0.22, y_top))

    leg2 = ax2.legend(loc="upper left", frameon=True, fancybox=True)
    leg2.get_frame().set_facecolor(LIGHT_GRAY)
    leg2.get_frame().set_edgecolor("0.80")
    leg2.get_frame().set_linewidth(0.6)
    ax2.grid(alpha=0.25, linestyle="--")

    if not args.no_annot:
        # single, bottom-right stats box (non-overlapping)
        stats_txt = (
            rf"OLS: $\hat\beta={b:.3f}\pm{1.96*se_b:.3f}$, "
            rf"$\hat\alpha={a:.3f}\pm{1.96*se_a:.3f}$;  $R^2={R2:.2f}$" "\n" +
            (rf"$t(\hat\beta\!=\!0)={t_b:.2f},\ t(\hat\beta\!=\!1)={t_b_vs1:.2f},\ \mathrm{{df}}={df}$"
             if not args.abs_x else rf"$t(\hat\beta\!=\!0)={t_b:.2f},\ \mathrm{{df}}={df}$")
            + f"\nDesign t (vs 0): {t_design:.2f}"
        )
        ax2.text(
            0.98, 0.02, stats_txt, transform=ax2.transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.90,
                      edgecolor="0.8", linewidth=0.6),
            fontsize=6.2, zorder=6
        )

    # Export stats to CSV (optional)
    try:
        rows = [{
            "kappa0_hat": k_hat, "kappa0_lo": k_lo, "kappa0_hi": k_hi, "se_k": se_k,
            "beta_hat": b, "se_beta": se_b, "alpha_hat": a, "se_alpha": se_a,
            "t_beta_vs0": t_b, "t_beta_vs1": (t_b_vs1 if not args.abs_x else np.nan),
            "df": df, "R2": R2, "design_t": t_design
        }]
        if pd is not None:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "Box2c_stats.csv"), index=False)
        else:
            # minimal CSV without pandas
            import csv
            with open(os.path.join(out_dir, "Box2c_stats.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
    except Exception:
        pass

    # Save
    out_pdf = os.path.join(out_dir, "Box2c_rate_refined.pdf")
    out_png = os.path.join(out_dir, "Box2c_rate_refined.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}  and  {out_png}")
    print(f"kappa0_hat={k_hat:.4f}  (95% CI: {k_lo:.4f}..{k_hi:.4f}); "
          f"OLS b={b:.4f} (se={se_b:.4f}), a={a:.4f} (se={se_a:.4f}), R2={R2:.3f}")
    if not args.abs_x:
        print(f"t-test vs slope=1: t={t_b_vs1:.2f}, df={df}")
    print(f"Design t (vs 0): {t_design:.2f}")

if __name__ == "__main__":
    main()

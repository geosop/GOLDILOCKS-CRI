# -*- coding: utf-8 -*-
"""
ADMIN
figures/make_tierB_tempered_figure.py

Panels:
 (A) Model comparison: AIC or AICc bar plot + LRT (Wilks χ²₂, boundary caveat)
 (B) Early-time log-survival curvature (ms) with KM ± Greenwood band (if sample_B.csv exists)
 (C) 95% CIs for (η_fast, τ_fast, τ_slow)

Notes:
 - Reads tierB_tempered/output/* produced by tierB_tempered/simulate_and_fit.py
 - If outputs are missing, auto-runs simulate_and_fit (with light bootstrap via env override)
 - Robust to both legacy columns (eta, AIC_*) and updated ones (eta_fast, AICc_*, lrt_valid)
 - η is interpreted as FAST-component weight in displays and annotations
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# CI-safe font (prevents Arial warnings on CI)
mpl.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path(__file__).resolve().parents[1]
TIERB = ROOT / "tierB_tempered"
OUT_TIERB = TIERB / "output"
OUT_FIG = Path(__file__).resolve().parent / "output"
OUT_FIG.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------- helpers ----------------------------
def ensure_data() -> None:
    """Ensure all expected outputs exist; if not, build them (quick bootstrap)."""
    need = [
        OUT_TIERB / "aic_lrt.csv",
        OUT_TIERB / "curvature_demo.csv",
        OUT_TIERB / "ci_2exp.csv",
        OUT_TIERB / "fit_1exp.csv",
        OUT_TIERB / "fit_2exp.csv",
    ]
    if all(p.exists() for p in need):
        return
    os.environ.setdefault("TIERB_BOOT", "60")  # light bootstrap for figure generation
    from tierB_tempered.simulate_and_fit import main as build
    print("Tier-B data missing → running simulate_and_fit.py (quick bootstrap=60)")
    build()


def _read_fit_rows():
    """Read fit_1exp.csv / fit_2exp.csv; be robust to legacy/new schemas."""
    f1 = pd.read_csv(OUT_TIERB / "fit_1exp.csv")
    f2 = pd.read_csv(OUT_TIERB / "fit_2exp.csv")

    # --- 1-exp ---
    # Legacy schema: columns: param, value, loglik, aic
    # New schema   : same; keep compatibility
    if "value" in f1.columns:
        tau1 = float(f1.loc[0, "value"])
    elif "tau" in f1.columns:
        tau1 = float(f1.loc[0, "tau"])
    else:
        # very old: try first numeric column as value
        tau1 = float(pd.to_numeric(f1.select_dtypes("number"), errors="coerce").iloc[0, 0])

    # --- 2-exp ---
    # New schema columns: eta_fast, tau_fast, tau_slow
    # Legacy schema columns: eta, tau_fast, tau_slow
    if "eta_fast" in f2.columns:
        eta_fast = float(f2.loc[0, "eta_fast"])
    elif "eta" in f2.columns:
        eta_fast = float(f2.loc[0, "eta"])  # interpret as fast-weight for display
    else:
        # fallback: try key/value style
        try:
            eta_fast = float(f2.query("param=='eta_fast'")["value"].iloc[0])
        except Exception:
            eta_fast = float(f2.query("param=='eta'")["value"].iloc[0])

    if "tau_fast" in f2.columns:
        tau_fast = float(f2.loc[0, "tau_fast"])
    else:
        tau_fast = float(f2.query("param=='tau_fast'")["value"].iloc[0])

    if "tau_slow" in f2.columns:
        tau_slow = float(f2.loc[0, "tau_slow"])
    else:
        tau_slow = float(f2.query("param=='tau_slow'")["value"].iloc[0])

    return tau1, eta_fast, tau_fast, tau_slow


def _read_aic_lrt():
    """Read aic_lrt.csv; support AIC and AICc columns + LRT caveat flag."""
    aic = pd.read_csv(OUT_TIERB / "aic_lrt.csv")

    # Prefer AICc if present and n/k<40 or user forces AICc via env
    use_aicc_env = os.environ.get("TIERB_USE_AICC", "").strip() == "1"
    have_aicc = all(c in aic.columns for c in ("AICc_1exp", "AICc_2exp"))
    have_aic = all(c in aic.columns for c in ("AIC_1exp", "AIC_2exp"))

    metric = "AIC"
    if use_aicc_env and have_aicc:
        metric = "AICc"
    elif have_aicc and "use_aicc" in aic.columns and bool(aic["use_aicc"].iloc[0]):
        metric = "AICc"

    if metric == "AICc" and have_aicc:
        m1 = float(aic["AICc_1exp"].iloc[0])
        m2 = float(aic["AICc_2exp"].iloc[0])
        delta = float(aic.get("Delta_AICc", m1 - m2))
    else:
        m1 = float(aic["AIC_1exp"].iloc[0]) if have_aic else float(aic["AICc_1exp"].iloc[0])
        m2 = float(aic["AIC_2exp"].iloc[0]) if have_aic else float(aic["AICc_2exp"].iloc[0])
        delta = float(aic.get("Delta_AIC", m1 - m2))

    lrt = float(aic.get("LRT", np.nan))
    pval = float(aic.get("p_value", np.nan))
    df = int(aic.get("df", 2))
    lrt_valid = bool(aic.get("lrt_valid", False)) if "lrt_valid" in aic.columns else False
    n_eff = int(aic.get("n", np.nan)) if "n" in aic.columns else np.nan

    return {
        "metric": metric,
        "m1": m1,
        "m2": m2,
        "delta": delta,
        "lrt": lrt,
        "pval": pval,
        "df": df,
        "lrt_valid": lrt_valid,
        "n": n_eff,
    }


def km_with_var(x: np.ndarray):
    """
    Kaplan–Meier for uncensored lifetimes + Greenwood SE for S(t),
    then delta-method SE for log S(t).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return x, np.full_like(x, np.nan), np.full_like(x, np.nan)

    xs, counts = np.unique(np.sort(x), return_counts=True)
    n = x.size
    S = 1.0
    S_list, t_list, varS_list = [], [], []
    cum = 0.0
    for xi, di in zip(xs, counts):
        if n <= 0:
            break
        S *= (1.0 - di / n)
        if n - di > 0:
            cum += di / (n * (n - di))  # Greenwood increment
        varS = (S ** 2) * cum
        n -= di
        if S <= 0:
            break
        S_list.append(S)
        t_list.append(xi)
        varS_list.append(varS)

    S = np.asarray(S_list)
    t = np.asarray(t_list)
    varS = np.asarray(varS_list)
    logS = np.log(S)
    se_logS = np.sqrt(varS) / S  # delta method
    return t, logS, se_logS


# ---------------------------- main ----------------------------
def main():
    ensure_data()

    # --- read model-comparison stats ---
    comp = _read_aic_lrt()
    metric, m1, m2, dM = comp["metric"], comp["m1"], comp["m2"], comp["delta"]
    lrt, p, df = comp["lrt"], comp["pval"], comp["df"]
    lrt_valid = comp["lrt_valid"]
    n_eff = comp["n"]

    # --- read fits and CI table ---
    tau1, eta_fast, tf, ts = _read_fit_rows()
    curv = pd.read_csv(OUT_TIERB / "curvature_demo.csv")

    # CI table: robust to old/new column names
    ci = pd.read_csv(OUT_TIERB / "ci_2exp.csv", index_col=0)
    col_eta = "eta_fast" if "eta_fast" in ci.columns else ("eta" if "eta" in ci.columns else None)
    if col_eta is None:
        raise ValueError("ci_2exp.csv must contain 'eta_fast' or 'eta' column.")
    col_tf = "tau_fast" if "tau_fast" in ci.columns else "tau_f"
    col_ts = "tau_slow" if "tau_slow" in ci.columns else "tau_s"

    # --- figure canvas ---
    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0.28)

    # ==================== (A) AIC/AICc + LRT ====================
    axA = fig.add_subplot(gs[0, 0])
    axA.set_title("Model comparison (Tier-B)", fontsize=11)
    bars = axA.bar([0, 1], [m1, m2], tick_label=["1-exp", "2-exp"])
    for b in bars:
        axA.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Summary textbox (metric + LRT caveat)
    lrt_line = (
        f"LRT={lrt:.2f} (χ²₍{df}₎), p={p:.3g}"
        if np.isfinite(lrt) and np.isfinite(p)
        else "LRT: n/a"
    )
    caveat = "Wilks approx; mixture boundary → p approx." if not lrt_valid else "Wilks valid."
    extra = f"  n={n_eff}" if np.isfinite(n_eff) else ""
    axA.text(
        0.02,
        0.98,
        f"Δ{metric}={dM:.1f}   {lrt_line}{extra}\n{caveat}",
        transform=axA.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.6,
            boxstyle="round,pad=0.2",
        ),
    )
    axA.set_ylabel(metric)

    # ==================== (B) Early log-survival curvature ====================
    axB = fig.add_subplot(gs[0, 1])
    axB.set_title("Log-survival curvature (early, ms)", fontsize=11)

    # Prefer KM from raw synthetic sample → CI band; else fall back to binned curve
    sample_path = OUT_TIERB / "sample_B.csv"
    if sample_path.exists():
        df_s = pd.read_csv(sample_path)
        s = (df_s["t"] if "t" in df_s.columns else df_s.iloc[:, 0]).to_numpy()
        t_emp_s, log_emp, se_log_emp = km_with_var(s)
    else:
        t_emp_s = curv["t"].to_numpy()
        log_emp = curv["log_surv_emp"].to_numpy()
        se_log_emp = None

    # Convert to ms and restrict to an "early" window tied to τ_slow
    t_emp_ms = 1e3 * t_emp_s
    t_mod_ms = 1e3 * curv["t"].to_numpy()
    # early window upper bound: min(200 ms, 5 * τ_slow) but at least 60 ms
    tau_slow_ms = max(60.0, min(200.0, ts * 1e3 * 5.0))
    m_emp = t_emp_ms <= tau_slow_ms
    m_mod = t_mod_ms <= tau_slow_ms

    # KM curve + 95% CI band (if available)
    if se_log_emp is not None and np.all(np.isfinite(se_log_emp)):
        lo = log_emp - 1.96 * se_log_emp
        hi = log_emp + 1.96 * se_log_emp
        axB.fill_between(t_emp_ms[m_emp], lo[m_emp], hi[m_emp], alpha=0.15, label="KM 95% CI")
    axB.plot(t_emp_ms[m_emp], log_emp[m_emp], marker="o", lw=0, ms=2.5, label="KM/ECDF")

    # Overlay model lines from curvature_demo
    axB.plot(t_mod_ms[m_mod], curv["log_surv_1exp"].to_numpy()[m_mod], lw=2, label="1-exp fit")
    axB.plot(t_mod_ms[m_mod], curv["log_surv_2exp"].to_numpy()[m_mod], lw=2, label="2-exp fit")

    axB.set_xlabel("time (ms)")
    axB.set_ylabel("log survival")
    axB.set_xlim(0, tau_slow_ms)
    axB.legend(fontsize=9, frameon=False, loc="lower left")

    # Simple curvature index κ = slope(10–50 ms) − slope(120–200 ms)
    def _slope(x, y):
        if x.size < 2 or y.size < 2:
            return np.nan
        return np.polyfit(x, y, 1)[0]

    W1 = (t_emp_ms >= 10) & (t_emp_ms <= 50)
    W2 = (t_emp_ms >= 120) & (t_emp_ms <= 200)
    k_hat = _slope(t_emp_ms[W1], log_emp[W1]) - _slope(t_emp_ms[W2], log_emp[W2])

    # Light bootstrap for κ CI if raw sample is available
    ci_text = [r"$\kappa$ = {:.3f}".format(k_hat)]
    if sample_path.exists():
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(300):  # light CI
            xb = rng.choice(s, size=s.size, replace=True)
            tb, lb, _seb = km_with_var(xb)
            tb_ms = 1e3 * tb
            W1b = (tb_ms >= 10) & (tb_ms <= 50)
            W2b = (tb_ms >= 120) & (tb_ms <= 200)
            kb = _slope(tb_ms[W1b], lb[W1b]) - _slope(tb_ms[W2b], lb[W2b])
            if np.isfinite(kb):
                boots.append(kb)
        if boots:
            lo_k, hi_k = np.percentile(boots, [2.5, 97.5])
            ci_text.append("95% CI [{:.3f}, {:.3f}]".format(lo_k, hi_k))

    axB.text(
        0.02,
        0.98,
        "\n".join(ci_text),
        transform=axB.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="round,pad=0.2"),
    )

    # ==================== (C) Parameter CIs ====================
    axC = fig.add_subplot(gs[0, 2])
    axC.set_title("95% CIs for (η_fast, τ_fast, τ_slow)", fontsize=11)

    labels = [r"η$_{\rm fast}$", r"τ$_{\rm fast}$ (ms)", r"τ$_{\rm slow}$ (ms)"]
    centers = [eta_fast, tf * 1e3, ts * 1e3]
    lo = [
        ci.loc["lo", col_eta],
        ci.loc["lo", col_tf] * 1e3,
        ci.loc["lo", col_ts] * 1e3,
    ]
    hi = [
        ci.loc["hi", col_eta],
        ci.loc["hi", col_tf] * 1e3,
        ci.loc["hi", col_ts] * 1e3,
    ]
    x = np.arange(len(labels))
    axC.errorbar(
        x,
        centers,
        yerr=[np.array(centers) - np.array(lo), np.array(hi) - np.array(centers)],
        fmt="o",
        capsize=4,
    )
    axC.set_xticks(x, labels)
    axC.set_xlim(-0.5, len(labels) - 0.5)
    axC.grid(axis="y", alpha=0.3)

    # Summary box (includes which metric was used)
    axC.text(
        0.02,
        0.98,
        (
            f"1-exp $\\hat{{\\tau}}$ = {tau1*1e3:.1f} ms\n"
            f"2-exp $\\hat{{\\eta}}_\\mathrm{{fast}}$ = {eta_fast:.2f}\n"
            f"$\\hat{{\\tau}}_\\mathrm{{fast}}$ = {tf*1e3:.1f} ms\n"
            f"$\\hat{{\\tau}}_\\mathrm{{slow}}$ = {ts*1e3:.1f} ms\n"
            f"Metric: {metric}"
        ),
        transform=axC.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="round,pad=0.2"),
    )

    fig.suptitle("Tier-B tempered mixtures: 1-exp vs 2-exp", fontsize=12, y=1.02)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"tierB_tempered_modelcomp.{ext}", bbox_inches="tight", dpi=300)

    print(f"Saved figures → {OUT_FIG}/tierB_tempered_modelcomp.[png|pdf]")


if __name__ == "__main__":
    main()

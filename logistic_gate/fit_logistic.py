#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
logistic_gate/fit_logistic.py  •  CRI (Box 2b / SI-ready)

Backbone: trial-wise Bernoulli logistic fits for two arousal levels with shared alpha.
Diagnostics (SI): kernel-smoothed curves + calibration curve and metrics.

Reads:
  - logistic_gate/output/logistic_trials.csv   (q, y, a[, p0])
  - logistic_gate/default_params.yml

Writes (same schema as the current pipeline):
  - logistic_gate/output/fit_logistic_results.csv
      p0_hat_a1, [p0_hat_a2], alpha_hat, Delta_p0, Delta_p0_lo, Delta_p0_hi,
      LRT_chi2, LRT_df, LRT_pval,
      Brier_a1, Brier_a2, Brier_all, ECE_a1, ECE_a2, ECE_all
  - logistic_gate/output/logistic_band.csv
      q, G_central_a1, G_low_a1, G_high_a1, [G_central_a2, G_low_a2, G_high_a2]
  - logistic_gate/output/logistic_derivative.csv
      q, dGdq_a1, [dGdq_a2]
  - logistic_gate/output/logistic_kernel.csv            (diagnostic)
      q, Gk_central_a1, Gk_low_a1, Gk_high_a1, [Gk_central_a2, Gk_low_a2, Gk_high_a2]
  - logistic_gate/output/logistic_calibration.csv       (diagnostic)
      bin_left, bin_right, bin_center, n_bin, pred_mean, obs_rate, lo, hi, a
  - logistic_gate/output/logistic_calibration_metrics.csv (diagnostic)
      metric, a, value

Key corrections:
  1) Proper LRT p-value via Chi-square survival function (df=1).
  2) Deterministic a1/a2 mapping (uses YAML a1/a2 if available; else sorted unique).
  3) Robust bootstrap loop: skips failed optimizations instead of poisoning percentiles.
  4) Numerically stable expit and clipping to avoid log(0).
"""
from __future__ import annotations

import os
import sys
import yaml
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import expit  # stable logistic
from scipy.stats import chi2     # correct LRT p-values

from math import sqrt

# -----------------------------------------------------------------------------
# Bootstrapping reproducibility hooks
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    from utilities.seed_manager import load_state, save_state
except Exception:
    def load_state():  # noqa: D401
        """No-op if seed_manager not available."""
        pass

    def save_state():  # noqa: D401
        """No-op if seed_manager not available."""
        pass

EPS = 1e-12


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def load_params() -> dict:
    here = os.path.dirname(__file__)
    path = os.path.join(here, "default_params.yml")
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    p = obj["logistic"] if isinstance(obj, dict) and "logistic" in obj else obj
    if not isinstance(p, dict):
        raise ValueError("default_params.yml did not parse into a dict under key 'logistic'.")

    # Back-compat defaults
    p.setdefault("q_min", p.get("p_min", 0.0))
    p.setdefault("q_max", p.get("p_max", 1.0))
    p.setdefault("n_points", 400)
    p.setdefault("seed", 52)

    p.setdefault("n_bootstrap", 2000)
    p.setdefault("ci_percent", 95)

    # Fitting initial guesses
    p.setdefault("p0_guess_a1", 0.50)
    p.setdefault("p0_guess_a2", 0.52)
    p.setdefault("alpha_guess", 0.05)
    p.setdefault("share_alpha", True)  # kept for clarity; this script always uses shared alpha in two-condition mode

    # Diagnostics
    p.setdefault("kernel_enabled", True)
    p.setdefault("kernel_bandwidth", 0.08)
    p.setdefault("kernel_bootstrap", min(1000, int(p["n_bootstrap"])))
    p.setdefault("calib_bins", 10)

    # prefer YAML arousal levels for mapping
    # p.setdefault("a1", 0.30)
    # p.setdefault("a2", 0.70)

    return p


# -----------------------------------------------------------------------------
# Model / utilities
# -----------------------------------------------------------------------------
def logistic(q: np.ndarray, p0: float, alpha: float) -> np.ndarray:
    """Stable logistic: G(q) = expit((q - p0) / alpha)."""
    alpha = max(float(alpha), 1e-12)
    q = np.asarray(q, dtype=float)
    return expit((q - float(p0)) / alpha)


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, EPS, 1.0 - EPS)


# Negative log-likelihoods
def _nll_single(theta, q, y):
    p0, alpha = theta
    if not (0.0 <= p0 <= 1.0 and alpha > 1e-8):
        return 1e12
    p = clamp01(logistic(q, p0, alpha))
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def _nll_shared_alpha(theta, q, y, a, a1, a2):
    p0_1, p0_2, alpha = theta
    if not (0.0 <= p0_1 <= 1.0 and 0.0 <= p0_2 <= 1.0 and alpha > 1e-8):
        return 1e12
    p = np.where(
        np.isclose(a, a1),
        clamp01(logistic(q, p0_1, alpha)),
        clamp01(logistic(q, p0_2, alpha)),
    )
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def _nll_shared_p0(theta, q, y):
    # H0: shared p0 and shared alpha
    p0, alpha = theta
    if not (0.0 <= p0 <= 1.0 and alpha > 1e-8):
        return 1e12
    p = clamp01(logistic(q, p0, alpha))
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def _minimize_with_restarts(fun, x0, args=(), bounds=None, max_restarts=5, rng=None):
    """
    Robustify optimizer with small random restarts.
    Returns best scipy.optimize.OptimizeResult found (may be unsuccessful).
    """
    best = None
    x0 = np.asarray(x0, dtype=float)

    for k in range(max_restarts + 1):
        if k == 0:
            start = x0.copy()
        else:
            noise = rng.normal(scale=0.02, size=len(x0)) if rng is not None else 0.02 * np.ones_like(x0)
            start = x0 + noise

        # project into bounds
        if bounds is not None:
            lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds], dtype=float)
            hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds], dtype=float)
            start = np.minimum(np.maximum(start, lo + 1e-12), hi - 1e-12)

        res = minimize(fun, start, args=args, method="L-BFGS-B", bounds=bounds)

        if best is None:
            best = res
        else:
            # Prefer successful; otherwise lowest fun
            if res.success and (not best.success or res.fun < best.fun):
                best = res
            elif (not best.success) and res.fun < best.fun:
                best = res

        if res.success and np.isfinite(res.fun):
            break

    return best


# Kernel smoother (Gaussian) for diagnostics
def gaussian_kernel(u):
    return np.exp(-0.5 * u * u)


def kernel_smoother(q_grid, q_obs, y_obs, h):
    """Nadaraya–Watson estimator with Gaussian kernel (diagnostic only)."""
    h = max(float(h), 1e-6)
    q_obs = np.asarray(q_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    preds = np.empty_like(q_grid, dtype=float)
    for i, q0 in enumerate(q_grid):
        w = gaussian_kernel((q0 - q_obs) / h)
        s = np.sum(w)
        preds[i] = np.sum(w * y_obs) / s if s > 0 else np.nan
    return preds


# Wilson interval for a binomial proportion (clamped & ordered)
# Brown, Cai, DasGupta (2001) Statistical Science 16(2)
def wilson_interval(s, n, z=1.959963984540054):  # 95% ~ N(0,1)
    if n <= 0:
        return (np.nan, np.nan)
    phat = s / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = z * sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n)) / denom
    lo, hi = center - half, center + half
    lo = max(0.0, min(1.0, lo))
    hi = max(0.0, min(1.0, hi))
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


# Calibration metrics
def brier_score(y, p):
    return float(np.mean((y - p) ** 2))


def ece_score(y, p, m=10):
    edges = np.linspace(0, 1, m + 1)
    total = len(y)
    ece = 0.0
    for i in range(m):
        if i < m - 1:
            idx = (p >= edges[i]) & (p < edges[i + 1])
        else:
            idx = (p >= edges[i]) & (p <= edges[i + 1])
        n = int(np.sum(idx))
        if n == 0:
            continue
        obs = float(np.mean(y[idx]))
        pred = float(np.mean(p[idx]))
        ece += (n / total) * abs(obs - pred)
    return float(ece)


# -----------------------------------------------------------------------------
# Fitting pipelines
# -----------------------------------------------------------------------------
def _resolve_a1_a2(df_trials: pd.DataFrame, p: dict) -> tuple[float, float]:
    """
    Deterministically resolve (a1,a2):
      - If YAML provides a1/a2 and both are present in the data (within tolerance),
        map to those.
      - Else, use sorted unique arousal values from data.
    """
    a_vals = np.unique(np.round(df_trials["a"].values.astype(float), 6))
    if len(a_vals) < 2:
        raise RuntimeError("Two-condition fit requested but only one arousal level present.")

    a_sorted = np.sort(a_vals)

    if "a1" in p and "a2" in p:
        target = np.array([float(p["a1"]), float(p["a2"])], dtype=float)
        # pick closest matches in data to YAML a1/a2
        # (robust to rounding)
        def _closest(val):
            return float(a_sorted[np.argmin(np.abs(a_sorted - val))])

        a1 = _closest(target[0])
        a2 = _closest(target[1])
        if np.isclose(a1, a2):
            # fallback to first two unique values if YAML mapping collapses
            a1, a2 = float(a_sorted[0]), float(a_sorted[1])
        return float(a1), float(a2)

    return float(a_sorted[0]), float(a_sorted[1])


def fit_two_condition(df_trials, p):
    # Identify a1, a2
    a1, a2 = _resolve_a1_a2(df_trials, p)

    q = df_trials["q"].values.astype(float)
    y = df_trials["y"].values.astype(float)
    a = df_trials["a"].values.astype(float)

    theta0 = np.array(
        [p.get("p0_guess_a1", 0.5), p.get("p0_guess_a2", 0.52), p.get("alpha_guess", 0.05)],
        dtype=float,
    )
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, None)]
    rng = np.random.default_rng(int(p["seed"]))

    res = _minimize_with_restarts(_nll_shared_alpha, theta0, args=(q, y, a, a1, a2), bounds=bounds, rng=rng)
    if res is None or not np.isfinite(res.fun):
        raise RuntimeError("Optimization failed for two-condition model (H1).")
    p0_1, p0_2, alpha_hat = res.x
    ll_H1 = -float(res.fun)

    # Null: shared p0, shared alpha
    theta0_H0 = np.array([(p0_1 + p0_2) / 2.0, alpha_hat], dtype=float)
    bounds_H0 = [(0.0, 1.0), (1e-6, None)]
    res_H0 = _minimize_with_restarts(_nll_shared_p0, theta0_H0, args=(q, y), bounds=bounds_H0, rng=rng)
    if res_H0 is None or not np.isfinite(res_H0.fun):
        raise RuntimeError("Optimization failed for null model (H0).")
    p0_shared, alpha_shared = res_H0.x
    ll_H0 = -float(res_H0.fun)

    # LRT (df=1): p-value = P(Chi2_1 >= chi2)
    chi2_stat = max(0.0, 2.0 * (ll_H1 - ll_H0))
    df_lrt = 1
    pval = float(chi2.sf(chi2_stat, df=df_lrt))

    # Bootstrap bands and Δp0 CI
    n_boot = int(p["n_bootstrap"])
    qs = np.linspace(float(p["q_min"]), float(p["q_max"]), int(p["n_points"]))

    idx1 = np.where(np.isclose(a, a1))[0]
    idx2 = np.where(np.isclose(a, a2))[0]
    q1, y1 = q[idx1], y[idx1]
    q2, y2 = q[idx2], y[idx2]

    curves1, curves2, dp0 = [], [], []
    n_success = 0
    max_attempts = max(n_boot, int(1.2 * n_boot))  # allow skips without shrinking too much

    for _ in range(max_attempts):
        if n_success >= n_boot:
            break

        b1 = rng.integers(len(q1), size=len(q1))
        b2 = rng.integers(len(q2), size=len(q2))
        q_b = np.concatenate([q1[b1], q2[b2]])
        y_b = np.concatenate([y1[b1], y2[b2]])
        a_b = np.concatenate(
            [np.full_like(b1, a1, dtype=float), np.full_like(b2, a2, dtype=float)]
        )

        r = _minimize_with_restarts(_nll_shared_alpha, theta0, args=(q_b, y_b, a_b, a1, a2), bounds=bounds, rng=rng)
        if r is None or (not r.success) or (not np.isfinite(r.fun)):
            continue

        p0_b1, p0_b2, alpha_b = r.x
        curves1.append(logistic(qs, p0_b1, alpha_b))
        curves2.append(logistic(qs, p0_b2, alpha_b))
        dp0.append(p0_b2 - p0_b1)
        n_success += 1

    if n_success < max(50, int(0.6 * n_boot)):
        raise RuntimeError(f"Bootstrap produced too few successful fits ({n_success}/{n_boot}).")

    alpha_ci = (100.0 - float(p["ci_percent"])) / 100.0
    curves1 = np.asarray(curves1, dtype=float)
    curves2 = np.asarray(curves2, dtype=float)

    lo1 = np.percentile(curves1, 100 * alpha_ci / 2, axis=0)
    hi1 = np.percentile(curves1, 100 * (1 - alpha_ci / 2), axis=0)
    lo2 = np.percentile(curves2, 100 * alpha_ci / 2, axis=0)
    hi2 = np.percentile(curves2, 100 * (1 - alpha_ci / 2), axis=0)

    dp0 = np.asarray(dp0, dtype=float)
    dp0_lo = float(np.percentile(dp0, 100 * alpha_ci / 2))
    dp0_hi = float(np.percentile(dp0, 100 * (1 - alpha_ci / 2)))

    # Diagnostics: kernel smoother (per condition)
    kernel_csv = None
    if bool(p.get("kernel_enabled", True)):
        h = float(p.get("kernel_bandwidth", 0.08))
        n_b = int(p.get("kernel_bootstrap", min(1000, n_boot)))
        grid = qs

        Gk1 = kernel_smoother(grid, q1, y1, h)
        Gk2 = kernel_smoother(grid, q2, y2, h)

        c1, c2 = [], []
        # kernel bootstrap uses resampling within each condition
        for _ in range(n_b):
            bb1 = rng.integers(len(q1), size=len(q1))
            bb2 = rng.integers(len(q2), size=len(q2))
            c1.append(kernel_smoother(grid, q1[bb1], y1[bb1], h))
            c2.append(kernel_smoother(grid, q2[bb2], y2[bb2], h))

        c1 = np.asarray(c1, dtype=float)
        c2 = np.asarray(c2, dtype=float)
        lo1_k = np.percentile(c1, 100 * alpha_ci / 2, axis=0)
        hi1_k = np.percentile(c1, 100 * (1 - alpha_ci / 2), axis=0)
        lo2_k = np.percentile(c2, 100 * alpha_ci / 2, axis=0)
        hi2_k = np.percentile(c2, 100 * (1 - alpha_ci / 2), axis=0)

        kernel_csv = pd.DataFrame(
            {
                "q": grid,
                "Gk_central_a1": Gk1,
                "Gk_low_a1": lo1_k,
                "Gk_high_a1": hi1_k,
                "Gk_central_a2": Gk2,
                "Gk_low_a2": lo2_k,
                "Gk_high_a2": hi2_k,
            }
        )

    # Calibration diagnostics
    calib_bins = max(2, int(p.get("calib_bins", 10)))
    p_pred = np.where(
        np.isclose(a, a1),
        clamp01(logistic(q, p0_1, alpha_hat)),
        clamp01(logistic(q, p0_2, alpha_hat)),
    )

    edges = np.linspace(0, 1, calib_bins + 1)
    rows = []
    for val in [a1, a2]:
        sel = np.isclose(a, val)
        yv = y[sel]
        pv = p_pred[sel]
        for i in range(calib_bins):
            left, right = edges[i], edges[i + 1]
            if i < calib_bins - 1:
                idx = (pv >= left) & (pv < right)
            else:
                idx = (pv >= left) & (pv <= right)
            n = int(np.sum(idx))
            if n == 0:
                continue
            s = int(np.sum(yv[idx]))
            obs = s / n
            lo_w, hi_w = wilson_interval(s, n)
            rows.append(
                {
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "bin_center": float(0.5 * (left + right)),
                    "n_bin": int(n),
                    "pred_mean": float(np.mean(pv[idx])),
                    "obs_rate": float(obs),
                    "lo": float(lo_w),
                    "hi": float(hi_w),
                    "a": float(val),
                }
            )
    calib_csv = pd.DataFrame(rows)

    # Metrics: Brier & ECE
    metrics = []
    for label, mask in [
        ("a1", np.isclose(a, a1)),
        ("a2", np.isclose(a, a2)),
        ("all", np.full_like(a, True, dtype=bool)),
    ]:
        yv = y[mask]
        pv = p_pred[mask]
        metrics.append({"metric": "Brier", "a": label, "value": brier_score(yv, pv)})
        metrics.append({"metric": "ECE", "a": label, "value": ece_score(yv, pv, m=calib_bins)})
    metrics_csv = pd.DataFrame(metrics)

    # Assemble outputs
    G1 = logistic(qs, p0_1, alpha_hat)
    G2 = logistic(qs, p0_2, alpha_hat)
    d1 = (G1 * (1.0 - G1)) / max(alpha_hat, 1e-12)
    d2 = (G2 * (1.0 - G2)) / max(alpha_hat, 1e-12)

    band_csv = pd.DataFrame(
        {
            "q": qs,
            "G_central_a1": G1,
            "G_low_a1": lo1,
            "G_high_a1": hi1,
            "G_central_a2": G2,
            "G_low_a2": lo2,
            "G_high_a2": hi2,
        }
    )
    der_csv = pd.DataFrame({"q": qs, "dGdq_a1": d1, "dGdq_a2": d2})

    res_csv = pd.DataFrame(
        [
            {
                "p0_hat_a1": float(p0_1),
                "p0_hat_a2": float(p0_2),
                "alpha_hat": float(alpha_hat),
                "Delta_p0": float(p0_2 - p0_1),
                "Delta_p0_lo": float(dp0_lo),
                "Delta_p0_hi": float(dp0_hi),
                "LRT_chi2": float(chi2_stat),
                "LRT_df": int(df_lrt),
                "LRT_pval": float(pval),
                "Brier_a1": float(metrics_csv.query("metric=='Brier' and a=='a1'")["value"].iloc[0]),
                "Brier_a2": float(metrics_csv.query("metric=='Brier' and a=='a2'")["value"].iloc[0]),
                "Brier_all": float(metrics_csv.query("metric=='Brier' and a=='all'")["value"].iloc[0]),
                "ECE_a1": float(metrics_csv.query("metric=='ECE' and a=='a1'")["value"].iloc[0]),
                "ECE_a2": float(metrics_csv.query("metric=='ECE' and a=='a2'")["value"].iloc[0]),
                "ECE_all": float(metrics_csv.query("metric=='ECE' and a=='all'")["value"].iloc[0]),
            }
        ]
    )

    return res_csv, band_csv, der_csv, kernel_csv, calib_csv, metrics_csv


def fit_single_condition(df_trials, p):
    q = df_trials["q"].values.astype(float)
    y = df_trials["y"].values.astype(float)

    theta0 = np.array([p.get("p0_guess_a1", 0.5), p.get("alpha_guess", 0.05)], dtype=float)
    bounds = [(0.0, 1.0), (1e-6, None)]
    rng = np.random.default_rng(int(p["seed"]))

    res = _minimize_with_restarts(_nll_single, theta0, args=(q, y), bounds=bounds, rng=rng)
    if res is None or not np.isfinite(res.fun):
        raise RuntimeError("Optimization failed for single-condition model.")
    p0_hat, alpha_hat = res.x

    # Bootstrap band
    n_boot = int(p["n_bootstrap"])
    qs = np.linspace(float(p["q_min"]), float(p["q_max"]), int(p["n_points"]))

    curves = []
    n_success = 0
    max_attempts = max(n_boot, int(1.2 * n_boot))
    for _ in range(max_attempts):
        if n_success >= n_boot:
            break
        idx = rng.integers(len(q), size=len(q))
        r = _minimize_with_restarts(_nll_single, theta0, args=(q[idx], y[idx]), bounds=bounds, rng=rng)
        if r is None or (not r.success) or (not np.isfinite(r.fun)):
            continue
        p0_b, alpha_b = r.x
        curves.append(logistic(qs, p0_b, alpha_b))
        n_success += 1

    if n_success < max(50, int(0.6 * n_boot)):
        raise RuntimeError(f"Bootstrap produced too few successful fits ({n_success}/{n_boot}).")

    curves = np.asarray(curves, dtype=float)
    alpha_ci = (100.0 - float(p["ci_percent"])) / 100.0
    lo = np.percentile(curves, 100 * alpha_ci / 2, axis=0)
    hi = np.percentile(curves, 100 * (1 - alpha_ci / 2), axis=0)

    G = logistic(qs, p0_hat, alpha_hat)
    d = (G * (1.0 - G)) / max(alpha_hat, 1e-12)

    band_csv = pd.DataFrame({"q": qs, "G_central_a1": G, "G_low_a1": lo, "G_high_a1": hi})
    der_csv = pd.DataFrame({"q": qs, "dGdq_a1": d})

    # Diagnostics (kernel + calibration) for a1
    kernel_csv = None
    if bool(p.get("kernel_enabled", True)):
        h = float(p.get("kernel_bandwidth", 0.08))
        n_b = int(p.get("kernel_bootstrap", min(1000, n_boot)))
        Gk = kernel_smoother(qs, q, y, h)
        B = []
        for _ in range(n_b):
            idx = rng.integers(len(q), size=len(q))
            B.append(kernel_smoother(qs, q[idx], y[idx], h))
        B = np.asarray(B, dtype=float)
        lo_k = np.percentile(B, 100 * alpha_ci / 2, axis=0)
        hi_k = np.percentile(B, 100 * (1 - alpha_ci / 2), axis=0)
        kernel_csv = pd.DataFrame({"q": qs, "Gk_central_a1": Gk, "Gk_low_a1": lo_k, "Gk_high_a1": hi_k})

    p_pred = clamp01(logistic(q, p0_hat, alpha_hat))
    calib_bins = max(2, int(p.get("calib_bins", 10)))
    edges = np.linspace(0, 1, calib_bins + 1)
    rows = []
    for i in range(calib_bins):
        left, right = edges[i], edges[i + 1]
        if i < calib_bins - 1:
            idx = (p_pred >= left) & (p_pred < right)
        else:
            idx = (p_pred >= left) & (p_pred <= right)
        n = int(np.sum(idx))
        if n == 0:
            continue
        s = int(np.sum(y[idx]))
        obs = s / n
        lo_w, hi_w = wilson_interval(s, n)
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "bin_center": float(0.5 * (left + right)),
                "n_bin": int(n),
                "pred_mean": float(np.mean(p_pred[idx])),
                "obs_rate": float(obs),
                "lo": float(lo_w),
                "hi": float(hi_w),
                "a": "a1",
            }
        )
    calib_csv = pd.DataFrame(rows)

    metrics_csv = pd.DataFrame(
        [
            {"metric": "Brier", "a": "a1", "value": brier_score(y, p_pred)},
            {"metric": "ECE", "a": "a1", "value": ece_score(y, p_pred, m=calib_bins)},
        ]
    )

    res_csv = pd.DataFrame(
        [
            {
                "p0_hat_a1": float(p0_hat),
                "alpha_hat": float(alpha_hat),
                "Delta_p0": np.nan,
                "Delta_p0_lo": np.nan,
                "Delta_p0_hi": np.nan,
                "LRT_chi2": np.nan,
                "LRT_df": 0,
                "LRT_pval": np.nan,
                "Brier_a1": float(metrics_csv.query("metric=='Brier' and a=='a1'")["value"].iloc[0]),
                "Brier_a2": np.nan,
                "Brier_all": float(metrics_csv.query("metric=='Brier' and a=='a1'")["value"].iloc[0]),
                "ECE_a1": float(metrics_csv.query("metric=='ECE' and a=='a1'")["value"].iloc[0]),
                "ECE_a2": np.nan,
                "ECE_all": float(metrics_csv.query("metric=='ECE' and a=='a1'")["value"].iloc[0]),
            }
        ]
    )

    return res_csv, band_csv, der_csv, kernel_csv, calib_csv, metrics_csv


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    load_state()
    p = load_params()
    save_state()

    here = os.path.dirname(__file__)
    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    trials_path = os.path.join(out_dir, "logistic_trials.csv")
    if not os.path.exists(trials_path):
        raise FileNotFoundError("Missing logistic_trials.csv. Run simulate_logistic.py first.")

    df_trials = pd.read_csv(trials_path)

    # Ensure required columns exist
    for col in ("q", "y", "a"):
        if col not in df_trials.columns:
            raise ValueError(f"logistic_trials.csv missing required column '{col}'.")

    a_levels = np.unique(np.round(df_trials["a"].values.astype(float), 6))
    two = bool(p.get("use_two_conditions", True)) and (len(a_levels) >= 2)

    if two:
        df_res, df_band, df_der, df_kernel, df_calib, df_metrics = fit_two_condition(df_trials, p)
    else:
        df_res, df_band, df_der, df_kernel, df_calib, df_metrics = fit_single_condition(df_trials, p)

    df_res.to_csv(os.path.join(out_dir, "fit_logistic_results.csv"), index=False)
    df_band.to_csv(os.path.join(out_dir, "logistic_band.csv"), index=False)
    df_der.to_csv(os.path.join(out_dir, "logistic_derivative.csv"), index=False)
    if df_kernel is not None:
        df_kernel.to_csv(os.path.join(out_dir, "logistic_kernel.csv"), index=False)
    df_calib.to_csv(os.path.join(out_dir, "logistic_calibration.csv"), index=False)
    df_metrics.to_csv(os.path.join(out_dir, "logistic_calibration_metrics.csv"), index=False)

    print(f"Saved logistic fit, bands, derivatives, and diagnostics to {out_dir}")


if __name__ == "__main__":
    main()

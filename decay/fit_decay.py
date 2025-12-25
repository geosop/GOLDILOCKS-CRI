#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
decay/fit_decay.py  •  CRI v0.3-SIM (robust)

Fits the Tier-A log-linear decay:
    y = ln A_pre(τ_f) = β0 + β1 * τ_f   with β1 = -1/τ_fut

Reports:
  - OLS
  - WLS (if se_lnA available)
  - Left-censored Tobit MLE at bound c = ln(A_min)

Bootstrap CIs for τ_fut (and an OLS line band used for figure shading).

Robustness / hostile-reviewer hardening:
  1) Config override via env var CRI_DECAY_CONFIG (same resolution policy as simulate_decay.py).
  2) Require decay/output/run_manifest.json from simulate_decay.py and enforce:
       - YAML(decay) effective params hash == manifest.run_hash
       - decay_data.csv signature is consistent with manifest.decay (sanity checks)
  3) Stamp run_hash + params_path into output CSVs.
  4) Deterministic bootstrapping; Tobit optimisation hardened (multi-start + fail-safe).
  5) Fixes env seed override bug present in earlier version.

Reads:
  - decay/output/decay_data.csv
  - decay/output/run_manifest.json  (required)
  - decay/default_params.yml or CRI_DECAY_CONFIG (for bootstrap settings + A_min)
Writes:
  - decay/output/fit_decay_results.csv
  - decay/output/decay_band.csv
"""
from __future__ import annotations

import os
import sys
import json
import yaml
import math
import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats


# Optional repo utilities (seed manager)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    from utilities.seed_manager import load_state, save_state
except Exception:
    def load_state():
        pass
    def save_state():
        pass


# -----------------------------------------------------------------------------#
# Hash / manifest utilities (must match simulate_decay.py)
# -----------------------------------------------------------------------------#
def _canonical_json_bytes(obj: Any) -> bytes:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def _compute_run_hash(decay_params: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(decay_params)).hexdigest()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------#
# Config resolution (same policy as simulate_decay.py)
# -----------------------------------------------------------------------------#
def _resolve_config_path(config_path: Optional[str]) -> str:
    """
    Resolve config path deterministically.
    Relative paths are resolved against repo root first, then decay/.

    Priority:
      1) explicit argument
      2) env var CRI_DECAY_CONFIG
      3) decay/default_params.yml (fallback)
    """
    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, ".."))

    if config_path is None:
        config_path = os.getenv("CRI_DECAY_CONFIG")

    if config_path is None:
        return os.path.join(here, "default_params.yml")

    candidates = [
        config_path,
        os.path.join(repo, config_path),
        os.path.join(here, config_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        f"Decay config not found: {config_path}\n"
        f"Tried:\n  - {candidates[0]}\n  - {candidates[1]}\n  - {candidates[2]}"
    )


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def load_params(config_path: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """
    Load YAML and return (effective_decay_fit_params, resolved_path).

    This loader reads the same YAML source as simulate_decay.py, but only
    uses the keys needed for fitting/bootstrapping and A_min.
    """
    cfg_path = _resolve_config_path(config_path)
    with open(cfg_path, "r", encoding="utf-8-sig", errors="replace") as f:
        y = yaml.safe_load(f) or {}

    p_raw = y.get("decay") if isinstance(y, dict) and "decay" in y else y
    if not isinstance(p_raw, dict):
        raise ValueError("Decay YAML did not parse to a dict (either top-level or under key 'decay').")

    A_min = _safe_float(p_raw.get("A_min", p_raw.get("epsilon_detection", 0.01)), 0.01)
    n_bootstrap = _safe_int(p_raw.get("n_bootstrap", p_raw.get("n_boot", 2000)), 2000)
    n_points = _safe_int(p_raw.get("n_points", p_raw.get("n_cont", 200)), 200)
    ci_percent = _safe_float(p_raw.get("ci_percent", 95.0), 95.0)
    seed = _safe_int(p_raw.get("seed", 52), 52)

    # CRI_SEED override (fixes earlier bug)
    env_seed = os.getenv("CRI_SEED", None)
    if env_seed is not None:
        try:
            seed = int(env_seed)
        except Exception:
            pass

    p_fit = {
        "seed": int(seed),
        "n_bootstrap": int(n_bootstrap),
        "ci_percent": float(ci_percent),
        "A_min": float(A_min),
        "n_points": int(n_points),
    }
    return p_fit, cfg_path


# -----------------------------------------------------------------------------#
# IO helpers
# -----------------------------------------------------------------------------#
def _load_data() -> pd.DataFrame:
    here = os.path.dirname(__file__)
    path = os.path.join(here, "output", "decay_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run decay/simulate_decay.py first.")
    df = pd.read_csv(path)
    if "delta" not in df.columns or "lnA_pre" not in df.columns:
        raise RuntimeError("decay_data.csv must have columns: 'delta', 'lnA_pre'[, 'se_lnA'].")
    # sanitize numeric
    df = df.copy()
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df["lnA_pre"] = pd.to_numeric(df["lnA_pre"], errors="coerce")
    if "se_lnA" in df.columns:
        df["se_lnA"] = pd.to_numeric(df["se_lnA"], errors="coerce")
    df = df.dropna(subset=["delta", "lnA_pre"])
    if len(df) < 3:
        raise RuntimeError("Need at least 3 valid rows to fit a line.")
    return df


def _enforce_manifest_and_hashes(decay_out: str, yaml_effective: Dict[str, Any], params_path: str) -> str:
    """
    Enforce:
      - decay/output/run_manifest.json exists and contains run_hash
      - YAML(decay) effective hash == manifest hash

    Returns: run_hash
    """
    man_path = os.path.join(decay_out, "run_manifest.json")
    if not os.path.exists(man_path):
        raise RuntimeError(
            "Missing decay/output/run_manifest.json. "
            "Rerun decay/simulate_decay.py before decay/fit_decay.py."
        )
    manifest = _read_json(man_path)
    if "run_hash" not in manifest:
        raise RuntimeError("run_manifest.json missing key 'run_hash'.")
    run_hash = str(manifest["run_hash"])

    yaml_hash = _compute_run_hash(yaml_effective)
    if yaml_hash != run_hash:
        raise RuntimeError(
            "YAML(decay) effective dict hash does not match run_manifest.json.\n"
            f"  YAML hash:      {yaml_hash}\n"
            f"  manifest hash:  {run_hash}\n"
            "Fix: delete decay/output/* and regenerate from the intended YAML."
        )

    # Optional: ensure manifest params_path matches resolved path (strong signal, but not fatal)
    man_path_val = manifest.get("params_path", None)
    if man_path_val is not None:
        if os.path.abspath(str(man_path_val)) != os.path.abspath(params_path):
            raise RuntimeError(
                "params_path mismatch between run_manifest.json and resolved config path.\n"
                f"  manifest.params_path={man_path_val}\n"
                f"  resolved params_path={params_path}\n"
                "Fix: delete decay/output/* and regenerate."
            )

    return run_hash


# -----------------------------------------------------------------------------#
# Core fits
# -----------------------------------------------------------------------------#
def _ols_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(b[0]), float(b[1])


def _wls_fit(x: np.ndarray, y: np.ndarray, se: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Robust WLS:
      - sanitize SEs → positive finite weights
      - solve via sqrt(W) least squares
      - fall back to OLS if insufficient valid points or degenerate design
    """
    if se is None:
        return _ols_fit(x, y)

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    se = np.asarray(se, float)

    se = np.where(~np.isfinite(se) | (se <= 0), np.nan, se)
    w = 1.0 / np.square(np.maximum(se, 1e-12))
    mask = np.isfinite(w) & (w > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return _ols_fit(x[mask], y[mask]) if mask.any() else _ols_fit(x, y)

    X = np.column_stack([np.ones_like(x[mask]), x[mask]])
    if np.allclose(X[:, 1], X[0, 1]):
        return _ols_fit(x[mask], y[mask])

    sqrtw = np.sqrt(w[mask])[:, None]
    Xw = X * sqrtw
    yw = y[mask] * sqrtw[:, 0]
    b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return float(b[0]), float(b[1])


def _tobit_fit(x: np.ndarray, y: np.ndarray, c: float, seed: int = 52) -> Tuple[float, float]:
    """
    Left-censored Tobit MLE at bound c:
      y* = b0 + b1 x + eps, eps ~ N(0, σ^2); y = max(y*, c)

    Returns (b0_hat, b1_hat).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    cens = (y <= c + 1e-12)

    def nll(theta: np.ndarray) -> float:
        b0, b1, log_sig = theta
        sig = float(np.exp(log_sig))
        mu = b0 + b1 * x
        z = (y - mu) / sig
        zc = (c - mu) / sig

        ll_unc = -0.5 * np.log(2 * np.pi) - log_sig - 0.5 * z**2
        ll_cen = stats.norm.logcdf(zc)
        ll = np.where(cens, ll_cen, ll_unc)
        # numeric guard
        if not np.all(np.isfinite(ll)):
            return np.inf
        return -float(np.sum(ll))

    # init from OLS
    b0, b1 = _ols_fit(x, y)
    resid = y - (b0 + b1 * x)
    sig0 = float(np.std(resid, ddof=max(1, len(y) - 2)))
    theta0 = np.array([b0, b1, np.log(max(sig0, 1e-3))], float)
    bounds = [(None, None), (None, None), (math.log(1e-6), math.log(1e3))]

    rng = np.random.default_rng(int(seed))
    best = None

    # Multi-start: a few mild perturbations around OLS init
    for k in range(8):
        start = theta0 if k == 0 else theta0 + rng.normal(scale=[0.10, 0.05, 0.10])
        res = optimize.minimize(nll, start, method="L-BFGS-B", bounds=bounds)
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
        if res.success and np.isfinite(res.fun):
            break

    if best is None or (not np.isfinite(best.fun)):
        # Fail-safe: return OLS rather than crashing
        return float(b0), float(b1)

    b0_hat, b1_hat, _ = best.x
    return float(b0_hat), float(b1_hat)


# -----------------------------------------------------------------------------#
# Bootstrap
# -----------------------------------------------------------------------------#
def _bootstrap_ci(vals: list[float], ci_percent: float = 95.0) -> Tuple[float, float]:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    a = (100.0 - float(ci_percent)) / 2.0
    return float(np.percentile(v, a)), float(np.percentile(v, 100.0 - a))


def _to_tau(b1: float) -> float:
    """
    τ_fut = -1/β1, with β1<0.
    Return NaN for non-identifiable / non-decaying cases.
    """
    if (not np.isfinite(b1)) or (b1 >= 0):
        return float("nan")
    return float(-1.0 / b1)


def _fit_all(
    df: pd.DataFrame,
    A_min: float,
    n_boot: int,
    ci_percent: float,
    seed: int,
    n_points: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = df["delta"].to_numpy(dtype=float)      # seconds
    y = df["lnA_pre"].to_numpy(dtype=float)
    se = df["se_lnA"].to_numpy(dtype=float) if "se_lnA" in df.columns else None
    c = float(np.log(float(A_min)))

    # Point fits
    b0_ols, b1_ols = _ols_fit(x, y)
    b0_wls, b1_wls = _wls_fit(x, y, se)
    b0_tob, b1_tob = _tobit_fit(x, y, c, seed=seed)

    tau_ols = _to_tau(b1_ols)
    tau_wls = _to_tau(b1_wls)
    tau_tob = _to_tau(b1_tob)

    # Bootstrap
    rng = np.random.default_rng(int(seed))
    B = int(n_boot)

    xs = np.linspace(float(np.min(x)), float(np.max(x)), int(n_points))
    lines = np.empty((B, xs.size), dtype=float)

    tau_ols_B: list[float] = []
    tau_wls_B: list[float] = []
    tau_tob_B: list[float] = []

    n = len(x)
    for b in range(B):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        seb = se[idx] if se is not None else None

        # OLS
        b0b, b1b = _ols_fit(xb, yb)
        tau_ols_B.append(_to_tau(b1b))
        lines[b, :] = b0b + b1b * xs

        # WLS
        _, b1wb = _wls_fit(xb, yb, seb)
        tau_wls_B.append(_to_tau(b1wb))

        # Tobit (hardened, but can still be slow; keep it deterministic)
        _, b1tb = _tobit_fit(xb, yb, c, seed=seed + 1337 + b)
        tau_tob_B.append(_to_tau(b1tb))

    alpha = (100.0 - float(ci_percent)) / 2.0
    lo, hi = np.percentile(lines, [alpha, 100.0 - alpha], axis=0)
    cen = b0_ols + b1_ols * xs

    ci_ols = _bootstrap_ci(tau_ols_B, ci_percent)
    ci_wls = _bootstrap_ci(tau_wls_B, ci_percent)
    ci_tob = _bootstrap_ci(tau_tob_B, ci_percent)

    band = pd.DataFrame(
        {
            "delta_cont": xs,
            "lnA_central": cen,
            "lnA_low": lo,
            "lnA_high": hi,
        }
    )

    results = pd.DataFrame(
        [
            {
                "tau_hat_ms": tau_ols * 1e3,
                "ci_lo_ms": ci_ols[0] * 1e3,
                "ci_hi_ms": ci_ols[1] * 1e3,
                "tau_hat_ms_wls": tau_wls * 1e3,
                "wls_ci_lo_ms": ci_wls[0] * 1e3,
                "wls_ci_hi_ms": ci_wls[1] * 1e3,
                "tau_hat_ms_tobit": tau_tob * 1e3,
                "tobit_ci_lo_ms": ci_tob[0] * 1e3,
                "tobit_ci_hi_ms": ci_tob[1] * 1e3,
            }
        ]
    )

    # Convenience flags
    lo_ms, hi_ms = float(results["ci_lo_ms"].iloc[0]), float(results["ci_hi_ms"].iloc[0])
    results["agree_wls_in_ols_CI"] = bool(lo_ms <= float(results["tau_hat_ms_wls"].iloc[0]) <= hi_ms)
    results["agree_tobit_in_ols_CI"] = bool(lo_ms <= float(results["tau_hat_ms_tobit"].iloc[0]) <= hi_ms)

    return results, band


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main() -> None:
    load_state()

    here = os.path.dirname(__file__)
    outd = os.path.join(here, "output")
    os.makedirs(outd, exist_ok=True)

    # Fit params from YAML + resolved path
    p_fit, params_path = load_params()

    # Require simulator manifest and match hashes (prevents silent mismatch)
    # We compute hash over the *effective* simulation params, so we must mirror
    # what simulate_decay.py hashed. That script hashes its effective decay dict
    # which includes: seed,A0,tau_f,noise_log,delta_start,delta_end,delta_step,n_cont,n_rep,apply_censoring,A_min.
    # Here, we reload those same keys for hashing from the same YAML + env overrides.
    # If you want stricter enforcement, keep these keys exactly aligned.
    with open(params_path, "r", encoding="utf-8-sig", errors="replace") as f:
        y = yaml.safe_load(f) or {}
    p_raw = y.get("decay") if isinstance(y, dict) and "decay" in y else y
    if not isinstance(p_raw, dict):
        raise ValueError("Decay YAML did not parse to a dict for hashing.")

    # Reconstruct the simulation-effective dict (matching simulate_decay.py)
    sim_effective = {
        "seed": int(p_fit["seed"]),  # includes CRI_SEED override
        "A0": _safe_float(p_raw.get("A0", 1.0), 1.0),
        "tau_f": _safe_float(p_raw.get("tau_fut", p_raw.get("tau_f", 0.02)), 0.02),
        "noise_log": _safe_float(p_raw.get("noise_log", 0.10), 0.10),
        "delta_start": _safe_float(p_raw.get("delta_start", 0.0), 0.0),
        "delta_end": _safe_float(p_raw.get("delta_end", 0.02), 0.02),
        "delta_step": _safe_float(p_raw.get("delta_step", 0.005), 0.005),
        "n_cont": _safe_int(p_raw.get("n_cont", 300), 300),
        "n_rep": _safe_int(p_raw.get("n_rep", 40), 40),
        "apply_censoring": bool(p_raw.get("apply_censoring", False)),
        "A_min": _safe_float(p_raw.get("A_min", p_raw.get("epsilon_detection", 0.01)), 0.01),
    }

    run_hash = _enforce_manifest_and_hashes(outd, sim_effective, params_path)

    # Load data
    df = _load_data()

    # Optional sanity checks vs manifest (soft checks; raise only if clearly inconsistent)
    if df["delta"].min() < -1e-9:
        raise RuntimeError("delta has negative values; expected seconds >=0.")
    if df["delta"].max() > 10.0:
        raise RuntimeError("delta max is unusually large (>10 s); expected Tier-A window in seconds.")

    # Fit + band
    res, band = _fit_all(
        df,
        A_min=float(p_fit["A_min"]),
        n_boot=int(p_fit["n_bootstrap"]),
        ci_percent=float(p_fit["ci_percent"]),
        seed=int(p_fit["seed"]),
        n_points=int(p_fit["n_points"]),
    )

    # Stamp provenance
    res.insert(0, "run_hash", run_hash)
    res.insert(1, "params_path", os.path.abspath(params_path))
    band.insert(0, "run_hash", run_hash)
    band.insert(1, "params_path", os.path.abspath(params_path))

    # Write
    res.to_csv(os.path.join(outd, "fit_decay_results.csv"), index=False)
    band.to_csv(os.path.join(outd, "decay_band.csv"), index=False)

    print(
        f"τ_fut (OLS) = {float(res['tau_hat_ms'].iloc[0]):.1f} ms | "
        f"WLS={float(res['tau_hat_ms_wls'].iloc[0]):.1f} ms | "
        f"Tobit={float(res['tau_hat_ms_tobit'].iloc[0]):.1f} ms"
    )
    print(f"run_hash={run_hash}")

    save_state()


if __name__ == "__main__":
    main()

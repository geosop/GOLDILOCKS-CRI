# -*- coding: utf-8 -*-
"""
qpt/qpt_fit.py

Fits:
  1) For each γ_b, linear fit to ln P(t) to recover κ0*(γ_fwd+γ_b) via slope,
     then reports gamma_sum_hat ≡ (γ_fwd+γ_b) = (-2*slope)/κ0  (dimensionless).
  2) R_mean vs λ_env → linear regression + bootstrap CI for slope.

Robustness upgrades (CI / hostile-reviewer hardening):
  A) Config override via env var CRI_QPT_CONFIG (same resolution policy as qpt_simulation.py).
  B) Require qpt/output/run_manifest.json and enforce:
       - YAML(qpt) -> run_hash matches manifest.run_hash
       - qpt_sim_data.npz run_hash (if present) matches manifest.run_hash
       - qpt_sim_data.npz params_path (if present) matches resolved params_path
  C) Stamp run_hash + params_path into output CSVs.

Writes:
  - qpt/output/qpt_pop_fit.csv
  - qpt/output/qpt_R_fit.csv
"""
from __future__ import annotations

import os
import json
import yaml
import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


# -----------------------------------------------------------------------------#
# Hash / manifest utilities (must match qpt_simulation.py)
# -----------------------------------------------------------------------------#
def _canonical_json_bytes(obj: Any) -> bytes:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def _compute_run_hash(qpt_params: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(qpt_params)).hexdigest()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------#
# Config resolution (same policy as qpt_simulation.py)
# -----------------------------------------------------------------------------#
def _resolve_config_path(config_path: Optional[str]) -> str:
    """
    Resolve config path deterministically.
    Relative paths are resolved against repo root first, then qpt/.

    Priority:
      1) explicit argument
      2) env var CRI_QPT_CONFIG
      3) qpt/default_params.yml (fallback)
    """
    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, ".."))

    if config_path is None:
        config_path = os.getenv("CRI_QPT_CONFIG")

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
        f"QPT config not found: {config_path}\n"
        f"Tried:\n  - {candidates[0]}\n  - {candidates[1]}\n  - {candidates[2]}"
    )


def load_params(config_path: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """Load YAML and return (qpt_dict, resolved_path)."""
    cfg_path = _resolve_config_path(config_path)
    with open(cfg_path, "r", encoding="utf-8-sig", errors="replace") as f:
        cfg = yaml.safe_load(f) or {}
    p = (cfg.get("qpt") or cfg or {})
    if not isinstance(p, dict):
        raise ValueError("QPT YAML did not parse to a dict under key 'qpt'.")
    return p, cfg_path


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


def _enforce_manifest_and_hashes(
    qpt_out: str,
    params: Dict[str, Any],
    params_path: str,
    sim_npz: np.lib.npyio.NpzFile,
    strict_npz: bool = True,
) -> str:
    """
    Enforce:
      - run_manifest.json exists and contains run_hash
      - YAML(qpt) hash == manifest hash
      - NPZ run_hash (if present) == manifest hash (strict by default)
      - NPZ params_path (if present) matches resolved params_path (strict by default)
    Returns: run_hash (manifest)
    """
    man_path = os.path.join(qpt_out, "run_manifest.json")
    if not os.path.exists(man_path):
        raise RuntimeError(
            "Missing qpt/output/run_manifest.json. "
            "Rerun qpt/qpt_simulation.py before qpt_fit.py."
        )
    manifest = _read_json(man_path)
    if "run_hash" not in manifest:
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
        msg = (
            "run_hash mismatch between qpt_sim_data.npz and run_manifest.json.\n"
            f"  npz.run_hash={npz_hash}\n"
            f"  manifest.run_hash={run_hash}\n"
            "Fix: delete qpt/output/* and regenerate."
        )
        if strict_npz:
            raise RuntimeError(msg)

    npz_path = _npz_get_str(sim_npz, "params_path")
    if npz_path is not None and os.path.abspath(npz_path) != os.path.abspath(params_path):
        msg = (
            "params_path mismatch between qpt_sim_data.npz and resolved config path.\n"
            f"  npz.params_path={npz_path}\n"
            f"  resolved params_path={params_path}\n"
            "Fix: delete qpt/output/* and regenerate."
        )
        if strict_npz:
            raise RuntimeError(msg)

    return run_hash


# -----------------------------------------------------------------------------#
# Fit functions
# -----------------------------------------------------------------------------#
def fit_population(t_s: np.ndarray, pop_row: np.ndarray) -> Tuple[float, float]:
    """
    Fit ln P(t) = a + b t (OLS), return (slope b, stderr of slope).
    """
    x = np.asarray(t_s, dtype=float)
    y = np.log(np.clip(np.asarray(pop_row, dtype=float), 1e-12, None))
    slope, _, _, _, stderr = linregress(x, y)
    return float(slope), float(stderr)


def fit_R(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    OLS y = a + b x; return (slope b, intercept a, stderr of slope).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept, _, _, stderr = linregress(x, y)
    return float(slope), float(intercept), float(stderr)


def bootstrap_R(x: np.ndarray, y: np.ndarray, n_boot: int, ci: float, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    slopes = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        s, _, _ = fit_R(x[idx], y[idx])
        slopes.append(s)
    low, high = np.percentile(slopes, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(low), float(high)


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main() -> None:
    here = os.path.dirname(__file__)
    qpt_out = os.path.join(here, "output")
    npz_path = os.path.join(qpt_out, "qpt_sim_data.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}. Run qpt/qpt_simulation.py first.")

    # Load YAML (for bootstrap settings) + resolved path
    p_yaml, params_path = load_params()

    # Load simulated data
    dat = np.load(npz_path, allow_pickle=True)
    t_s          = dat["t"]
    gamma_b_vals = dat["gamma_b_vals"]
    pops         = dat["pops"]          # 2D float array (n_gb × n_t)
    lambda_env   = dat["lambda_env"]    # s^-1
    R_mean       = dat["R_mean"]        # dimensionless
    kappa0       = float(dat["kappa0"]) # s^-1

    # Enforce manifest + hash consistency (requires simulator wrote manifest)
    run_hash = _enforce_manifest_and_hashes(qpt_out, p_yaml, params_path, dat, strict_npz=True)

    # Resolve gamma_fwd (for reporting gamma_sum_hat)
    gamma_fwd = p_yaml.get("gamma_fwd", None)
    if gamma_fwd is None:
        gamma_fwd = p_yaml.get("gamma_f", 1.0)
    gamma_fwd = float(gamma_fwd)

    # ---------------- Population fits ----------------
    # ln P(t) slope = -0.5 * kappa0 * (gamma_fwd + gamma_b)
    # -> gamma_sum_hat = (gamma_fwd+gamma_b) = (-2*slope)/kappa0
    rows = []
    for gb, pop_row in zip(gamma_b_vals, pops):
        slope, slope_se = fit_population(t_s, pop_row)
        gamma_sum_hat = (-2.0 * slope) / kappa0
        gamma_sum_se  = (2.0 * slope_se) / kappa0
        rows.append({
            "run_hash": run_hash,
            "params_path": os.path.abspath(params_path),
            "kappa0_s^-1": float(kappa0),
            "gamma_fwd": float(gamma_fwd),
            "gamma_b": float(gb),
            "gamma_sum_hat": float(gamma_sum_hat),
            "gamma_sum_se": float(gamma_sum_se),
            "slope_s^-1": float(slope),
            "slope_se_s^-1": float(slope_se),
        })
    pd.DataFrame(rows).to_csv(os.path.join(qpt_out, "qpt_pop_fit.csv"), index=False)

    # ---------------- R fit + bootstrap CI for slope ----------------
    # Here we keep absolute x = λ_env (s^-1); slope has units of 1/(s^-1)=s.
    # make_tomography_figure.py can normalize by κ0_hat for a dimensionless slope=1 view.
    slope, intercept, s_err = fit_R(lambda_env, R_mean)

    n_boot = int(p_yaml.get("n_bootstrap", 2000))
    ci_pct = float(p_yaml.get("ci_percent", 95))
    ci_low, ci_high = bootstrap_R(lambda_env, R_mean, n_boot=n_boot, ci=ci_pct, seed=0)

    df_R = pd.DataFrame([{
        "run_hash": run_hash,
        "params_path": os.path.abspath(params_path),
        "kappa0_s^-1": float(kappa0),
        "x": "lambda_env",
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_err": float(s_err),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_bootstrap": int(n_boot),
        "ci_percent": float(ci_pct),
    }])
    df_R.to_csv(os.path.join(qpt_out, "qpt_R_fit.csv"), index=False)

    print("Saved qpt_pop_fit.csv and qpt_R_fit.csv")
    print(f"run_hash={run_hash}")


if __name__ == "__main__":
    main()

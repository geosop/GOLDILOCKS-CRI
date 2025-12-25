# -*- coding: utf-8 -*-
"""
qpt/qpt_simulation.py

Generates synthetic data for Box-2(c) with correct units:

Left panel (ms axis in the figure):
    P(t) = exp(- (kappa0/2) * (gamma_fwd + gamma_b) * t),   t in seconds.
    gamma_fwd defaults to YAML 'gamma_f' if 'gamma_fwd' is absent.

Right panel (dimensionless):
    For each λ_env in a grid, simulate bootstrap samples around the
    theory R(λ_env) = λ_env / kappa0, then report mean and 95% CI.

Robustness upgrades (CI / hostile-reviewer hardening):
  1) Config override via env var CRI_QPT_CONFIG (relative resolved against repo root, then qpt/).
  2) Deterministic run_hash = sha256(canonical JSON of *effective* qpt params).
  3) Require qpt/output/run_manifest.json written each run (records run_hash + params_path + params).
  4) Stamp run_hash + params_path into qpt/output/qpt_sim_data.npz for downstream consistency checks.

Writes qpt/output/qpt_sim_data.npz with:
    run_hash, params_path, t [s], gamma_b_vals, pops (n_gb × n_t), lambda_env [s^-1],
    R_mean, R_ci_low, R_ci_high, and kappa0 (for reference).
"""
from __future__ import annotations

import os
import sys
import json
import yaml
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

# Ensure utilities on path (optional)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)
try:
    from utilities.seed_manager import load_state, save_state
except Exception:
    def load_state():
        pass
    def save_state():
        pass

DEFAULTS: Dict[str, Any] = {
    # Simulation parameters (CRI-friendly defaults)
    "gamma_f": 1.0,                 # kept for backward compat
    "gamma_fwd": None,              # preferred name; if None → use gamma_f
    "gamma_b_vals": [0.2, 0.8],
    "kappa0": None,                 # s^-1; if None → derive from tau_mem_* or fallback 8.0
    "tau_mem_s": None,              # optional
    "tau_mem_ms": None,             # optional

    "t_max": 0.2,                   # seconds (200 ms window)
    "n_t": 200,

    # Environmental coupling scan
    "lambda_env_min": 0.0,          # s^-1
    "lambda_env_max": 1.0,          # s^-1
    "n_lambda_env": 11,

    # Bootstrap around theory R = λ_env / kappa0
    "noise_R": 0.05,                # stdev of noise in R-units (dimensionless)
    "n_trials_per_lambda": None,    # if None, fall back to n_bootstrap
    "n_bootstrap": 1000,            # kept for backward compat
    "ci_percent": 95,

    # RNG
    "seed": 52,
}


# -----------------------------------------------------------------------------#
# Hash / manifest utilities
# -----------------------------------------------------------------------------#
def _canonical_json_bytes(obj: Any) -> bytes:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def _compute_run_hash(qpt_params: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(qpt_params)).hexdigest()


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _safe_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return float(default) if default is not None else None


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


def load_params(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Merge YAML with DEFAULTS; supports CRI_QPT_CONFIG override."""
    cfg_path = _resolve_config_path(config_path)
    with open(cfg_path, "r", encoding="utf-8-sig", errors="replace") as f:
        cfg = yaml.safe_load(f) or {}
    user = (cfg.get("qpt") or cfg or {})
    if not isinstance(user, dict):
        raise ValueError("QPT YAML did not parse to a dict under key 'qpt'.")

    p: Dict[str, Any] = {**DEFAULTS, **user}
    p["_params_path"] = cfg_path  # provenance only

    # Resolve gamma_fwd
    if p.get("gamma_fwd", None) is None:
        p["gamma_fwd"] = _safe_float(p.get("gamma_f", 1.0), 1.0)

    # Resolve kappa0 (s^-1): explicit → tau_mem_s → tau_mem_ms → fallback 8.0
    k0 = p.get("kappa0", None)
    if k0 is not None:
        p["kappa0"] = _safe_float(k0, 8.0)
    else:
        tau_s = _safe_float(p.get("tau_mem_s", None), None)
        tau_ms = _safe_float(p.get("tau_mem_ms", None), None)
        if tau_s and tau_s > 0:
            p["kappa0"] = 1.0 / tau_s
        elif tau_ms and tau_ms > 0:
            p["kappa0"] = 1000.0 / tau_ms
        else:
            p["kappa0"] = 8.0  # ≈ 125 ms

    # Trials per λ
    if p.get("n_trials_per_lambda", None) is None:
        p["n_trials_per_lambda"] = int(p.get("n_bootstrap", 1000))

    # Coerce numerics / types
    p["gamma_fwd"] = float(_safe_float(p["gamma_fwd"], 1.0))
    p["gamma_b_vals"] = [float(g) for g in p["gamma_b_vals"]]
    p["t_max"] = float(_safe_float(p.get("t_max", 0.2), 0.2))
    p["n_t"] = int(p["n_t"])
    p["lambda_env_min"] = float(_safe_float(p.get("lambda_env_min", 0.0), 0.0))
    p["lambda_env_max"] = float(_safe_float(p.get("lambda_env_max", 1.0), 1.0))
    p["n_lambda_env"] = int(p["n_lambda_env"])
    p["noise_R"] = float(_safe_float(p.get("noise_R", 0.05), 0.05))
    p["ci_percent"] = float(_safe_float(p.get("ci_percent", 95.0), 95.0))
    p["seed"] = int(p["seed"])

    return p


def simulate_populations(t_s: np.ndarray, gamma_fwd: float, gamma_b_vals: list[float], kappa0: float) -> np.ndarray:
    """
    Return a 2D array (n_gb × n_t) with:
        P(t) = exp( - (kappa0/2) * (gamma_fwd + gamma_b) * t ).
    """
    curves = []
    for gb in gamma_b_vals:
        rate = 0.5 * kappa0 * (gamma_fwd + gb)  # s^-1
        curves.append(np.exp(-rate * t_s))
    return np.vstack(curves).astype(float)


def main() -> None:
    load_state()  # no-op if utilities not present
    p = load_params()
    rng = np.random.default_rng(p["seed"])
    save_state()

    # --- Left panel data (time in seconds) ---
    t = np.linspace(0.0, p["t_max"], p["n_t"])
    pops = simulate_populations(t, p["gamma_fwd"], p["gamma_b_vals"], p["kappa0"])

    # --- Right panel data: R = λ_env / kappa0 + noise ---
    lambda_env = np.linspace(p["lambda_env_min"], p["lambda_env_max"], p["n_lambda_env"])
    R_true = lambda_env / p["kappa0"]  # dimensionless
    n_trials = int(p["n_trials_per_lambda"])
    sigma = float(p["noise_R"])
    alpha = (100.0 - float(p["ci_percent"])) / 100.0

    R_mean, R_low, R_high = [], [], []
    for r0 in R_true:
        samples = r0 + rng.normal(0.0, sigma, size=n_trials)
        samples = np.clip(samples, 0.0, None)
        R_mean.append(float(samples.mean()))
        R_low.append(float(np.percentile(samples, 100 * alpha / 2)))
        R_high.append(float(np.percentile(samples, 100 * (1 - alpha / 2))))

    # --- Provenance / manifest (anti-contradiction) ---
    params_effective = {k: v for k, v in p.items() if not str(k).startswith("_")}
    run_hash = _compute_run_hash(params_effective)

    out = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out, exist_ok=True)

    manifest = {
        "run_hash": run_hash,
        "params_path": p.get("_params_path", "NA"),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "qpt": params_effective,
    }
    _write_json(os.path.join(out, "run_manifest.json"), manifest)

    # --- Save ---
    np.savez(
        os.path.join(out, "qpt_sim_data.npz"),
        run_hash=np.array(run_hash),
        params_path=np.array(p.get("_params_path", "NA")),
        t=t.astype(float),
        gamma_b_vals=np.array(p["gamma_b_vals"], dtype=float),
        pops=pops,                              # 2D (n_gb × n_t)
        lambda_env=lambda_env.astype(float),    # s^-1
        R_mean=np.array(R_mean, dtype=float),   # dimensionless
        R_ci_low=np.array(R_low, dtype=float),
        R_ci_high=np.array(R_high, dtype=float),
        kappa0=float(p["kappa0"]),              # for reference
    )

    print(
        "Saved qpt_sim_data.npz + run_manifest.json "
        f"(pops shape: {pops.shape}, κ0={p['kappa0']:.3f} s^-1, run_hash={run_hash[:12]})"
    )


if __name__ == "__main__":
    main()

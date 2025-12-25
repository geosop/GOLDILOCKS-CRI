# -*- coding: utf-8 -*-
"""
decay/simulate_decay.py  •  CRI v0.3-SIM (robust)

Generates:
  - decay/output/decay_data.csv         (delta, lnA_pre, se_lnA)       <-- used for WLS/OLS/Tobit
  - decay/output/decay_curve.csv        (delta_cont, lnA_pre_cont)     <-- used to draw the line
  - decay/output/decay_data_raw.csv     (delta, lnA_pre_raw)           <-- replicates (SI)
  - decay/output/run_manifest.json      provenance: run_hash, params_path, effective params

Robustness / correctness upgrades:
  1) Deterministic config resolution (explicit arg / env / default) with repo-root + local fallback.
  2) CRI_SEED override applied correctly (fixes bug in previous script).
  3) Deterministic run_hash = sha256(canonical JSON of *effective* decay params).
  4) Parameter validation (positive scales; delta grid sanity; n_rep>=2; etc.).
  5) Optional log-domain censoring (detection floor) applied consistently.
  6) Writes run_manifest.json and stamps run_hash + params_path into NPZ-equivalent provenance (JSON only).
"""
from __future__ import annotations

import os
import json
import yaml
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------#
# Hash / manifest utilities
# -----------------------------------------------------------------------------#
def _canonical_json_bytes(obj: Any) -> bytes:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def _compute_run_hash(decay_params: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(decay_params)).hexdigest()


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------#
# Config resolution (CI-safe; mirrors QPT policy)
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
    Load YAML and return (effective_decay_params, resolved_path).

    Supports either:
      - top-level decay: {...}
      - or the file itself is the decay dict
    """
    cfg_path = _resolve_config_path(config_path)
    with open(cfg_path, "r", encoding="utf-8-sig", errors="replace") as f:
        y = yaml.safe_load(f) or {}

    p_raw = y.get("decay") if isinstance(y, dict) and "decay" in y else y
    if not isinstance(p_raw, dict):
        raise ValueError("Decay YAML did not parse to a dict (either top-level or under key 'decay').")

    # Build effective params with defaults + manuscript-friendly aliases
    p: Dict[str, Any] = {
        "seed": _safe_int(p_raw.get("seed", 52), 52),
        "A0": _safe_float(p_raw.get("A0", 1.0), 1.0),
        # accept tau_fut as alias
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

    # CRI_SEED override (correctly applied)
    env_seed = os.getenv("CRI_SEED", None)
    if env_seed is not None:
        try:
            p["seed"] = int(env_seed)
        except Exception:
            pass

    return p, cfg_path


# -----------------------------------------------------------------------------#
# Validation
# -----------------------------------------------------------------------------#
def _validate_params(p: Dict[str, Any]) -> None:
    if p["A0"] <= 0:
        raise ValueError(f"A0 must be >0 (got {p['A0']}).")
    if p["tau_f"] <= 0:
        raise ValueError(f"tau_f must be >0 seconds (got {p['tau_f']}).")
    if p["noise_log"] < 0:
        raise ValueError(f"noise_log must be >=0 (got {p['noise_log']}).")
    if p["delta_step"] <= 0:
        raise ValueError(f"delta_step must be >0 seconds (got {p['delta_step']}).")
    if p["delta_end"] < p["delta_start"]:
        raise ValueError("delta_end must be >= delta_start.")
    if p["n_cont"] < 10:
        raise ValueError("n_cont must be >=10 for a smooth curve.")
    if p["n_rep"] < 2:
        raise ValueError("n_rep must be >=2 to compute SE.")
    if p.get("apply_censoring", False) and p["A_min"] <= 0:
        raise ValueError("A_min must be >0 when apply_censoring=True.")


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main(config_path: Optional[str] = None) -> None:
    p, params_path = load_params(config_path=config_path)
    _validate_params(p)

    # Provenance: hash only the effective decay params (no private keys)
    params_effective = dict(p)
    run_hash = _compute_run_hash(params_effective)

    print(
        "simulate_decay.py VERSION: CRI v0.3-SIM",
        f"(seed={p['seed']}, n_rep={p['n_rep']}, noise_log={p['noise_log']}, "
        f"delta=[{p['delta_start']},{p['delta_end']}] step={p['delta_step']}, "
        f"tau_f={p['tau_f']} s, censor={p['apply_censoring']})"
    )
    print(f"run_hash={run_hash}")

    rng = np.random.default_rng(int(p["seed"]))

    here = os.path.dirname(__file__)
    outd = os.path.join(here, "output")
    os.makedirs(outd, exist_ok=True)

    # Write manifest early (helps CI debugging)
    manifest = {
        "run_hash": run_hash,
        "params_path": os.path.abspath(params_path),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decay": params_effective,
    }
    _write_json(os.path.join(outd, "run_manifest.json"), manifest)

    # Discrete delays (seconds)
    deltas = np.arange(p["delta_start"], p["delta_end"] + 1e-12, p["delta_step"]).astype(float)
    if deltas.size < 2:
        raise ValueError("delta grid has <2 points; check delta_start/end/step.")

    # True mean in log-domain: ln A_true(delta) = ln(A0) - delta / tau_f
    lnA_true = np.log(float(p["A0"])) - deltas / float(p["tau_f"])

    # Simulate replicates in log-domain (homoscedastic log-noise)
    raw_rows = []
    c = float(np.log(p["A_min"])) if p.get("apply_censoring", False) else None
    for d, mu in zip(deltas, lnA_true):
        y = mu + rng.normal(0.0, float(p["noise_log"]), size=int(p["n_rep"]))
        if c is not None:
            y = np.maximum(y, c)
        for v in y:
            raw_rows.append({"delta": float(d), "lnA_pre_raw": float(v)})

    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv(os.path.join(outd, "decay_data_raw.csv"), index=False)

    # Aggregate to mean ± SE per delay (WLS uses se_lnA)
    agg = df_raw.groupby("delta", as_index=False).agg(
        lnA_pre=("lnA_pre_raw", "mean"),
        sd=("lnA_pre_raw", "std"),
        n=("lnA_pre_raw", "size"),
    )
    # Use ddof=1 std from pandas; SE = sd/sqrt(n); guard n=1 (already prevented by n_rep>=2)
    agg["se_lnA"] = agg["sd"] / np.sqrt(agg["n"].astype(float))
    agg_out = agg[["delta", "lnA_pre", "se_lnA"]].copy()
    agg_out.to_csv(os.path.join(outd, "decay_data.csv"), index=False)

    # Dense noiseless curve for drawing the central line
    x_cont = np.linspace(float(p["delta_start"]), float(p["delta_end"]), int(p["n_cont"])).astype(float)
    lnA_cont = np.log(float(p["A0"])) - x_cont / float(p["tau_f"])
    pd.DataFrame({"delta_cont": x_cont, "lnA_pre_cont": lnA_cont}).to_csv(
        os.path.join(outd, "decay_curve.csv"),
        index=False,
    )

    print(f"Saved discrete data with SEs → {os.path.join(outd, 'decay_data.csv')}")
    print(f"Saved dense curve → {os.path.join(outd, 'decay_curve.csv')}")
    print(f"Saved manifest → {os.path.join(outd, 'run_manifest.json')}")


if __name__ == "__main__":
    main()

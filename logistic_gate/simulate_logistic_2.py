# -*- coding: utf-8 -*-
"""
logistic_gate/simulate_logistic.py  •  CRI (Box 2b / SI-ready)

Simulate trial-wise Bernoulli activations for the logistic “tipping point”:
    G(q|a) = expit((q - p0(a)) / alpha)

Supports two modes for p0(a):
  1) p0_mode: "explicit"
       - uses p0_a1 and p0_a2 from YAML (back-compatible with older configs)
  2) p0_mode: "gaussian"
       - uses p0(a) = p_base ± delta_p0 * exp(-(a-a0)^2/(2*sigma_a^2))
       - choose p0_shape: "dip" or "bump"

CRITICAL REPRODUCIBILITY / ANTI-CONTRADICTION:
  - Writes logistic_gate/output/run_manifest.json containing:
      * run_hash (SHA-256 of canonicalised "logistic" dict)
      * config_sha256 (SHA-256 of the YAML file bytes)
      * config_path_rel (relative to repo root where possible)
      * exact parameters used + derived p0_used(a1), p0_used(a2)
      * timestamp + optional git commit
  - Stamps run_hash into every row of CSV outputs.
  - Downstream scripts should verify run_hash matches manifest and CSVs.

Writes:
  - logistic_gate/output/logistic_curve.csv
      q, G_a1, [G_a2], run_hash
  - logistic_gate/output/logistic_trials.csv
      q, y, a, p0, run_hash   (trial-wise Bernoulli outcomes + the p0 used)
  - logistic_gate/output/logistic_bins.csv
      q_bin_center, rate_mean, n_bin, a, run_hash  (bin means for visualization ONLY)
  - logistic_gate/output/run_manifest.json
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import yaml
import hashlib
import datetime as _dt
import subprocess
import platform
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.special import expit  # stable logistic

# --- reproducibility hooks (optional) ---
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


# ----------------------------- manifest utilities ----------------------------
def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_git_commit(repo_root: str) -> Optional[str]:
    """Return current git commit hash if available; otherwise None. Never raises."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out if out else None
    except Exception:
        return None


def _safe_git_dirty(repo_root: str) -> Optional[bool]:
    """Return True if repo has uncommitted changes, False if clean, None if unknown."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return None


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_bytes(obj: Any) -> bytes:
    """
    Deterministic JSON encoding for hashing.
    - sort keys
    - remove whitespace
    """
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def _compute_run_hash(logistic_params: Dict[str, Any]) -> str:
    """
    Hash only the *logistic* section to avoid unrelated YAML edits changing Box2b outputs.
    """
    payload = _canonical_json_bytes(logistic_params)
    return hashlib.sha256(payload).hexdigest()


def _relpath_safe(path: str, root: str) -> str:
    """Best-effort relative path; falls back to basename if not a subpath."""
    try:
        root = os.path.abspath(root)
        path = os.path.abspath(path)
        common = os.path.commonpath([root, path])
        if common == root:
            return os.path.relpath(path, root)
    except Exception:
        pass
    return os.path.basename(path)


def _write_run_manifest(
    out_dir: str,
    run_hash: str,
    config_path: str,
    logistic_params: Dict[str, Any],
    derived: Dict[str, Any],
    repo_root: str,
) -> str:
    """
    Writes output/run_manifest.json and returns its path.
    """
    manifest = {
        "schema_version": 1,
        "run_hash": run_hash,
        "timestamp_utc": _utc_now_iso(),
        "script": os.path.relpath(__file__, repo_root) if os.path.exists(repo_root) else os.path.basename(__file__),
        "repo_root": os.path.abspath(repo_root),
        "config_path": os.path.abspath(config_path),
        "config_path_rel": _relpath_safe(config_path, repo_root),
        "config_sha256": _sha256_of_file(config_path),
        "git_commit": _safe_git_commit(repo_root),
        "git_dirty": _safe_git_dirty(repo_root),
        "python": {
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "packages": {
            "numpy": getattr(np, "__version__", None),
            "pandas": getattr(pd, "__version__", None),
        },
        "logistic_params": logistic_params,  # full dict for auditability
        "derived": derived,                  # includes p0_used(a1), p0_used(a2), etc.
        "outputs": {
            "logistic_curve_csv": os.path.join(out_dir, "logistic_curve.csv"),
            "logistic_trials_csv": os.path.join(out_dir, "logistic_trials.csv"),
            "logistic_bins_csv": os.path.join(out_dir, "logistic_bins.csv"),
        },
    }
    path = os.path.join(out_dir, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
    return path


# ----------------------------- config ----------------------------------------
def _resolve_params_path(config_path: str | None = None) -> str:
    """Resolve YAML path with sensible precedence.

    Precedence:
      1) explicit argument `config_path`
      2) env var CRI_LOGISTIC_CONFIG (or CRI_BOX2B_CONFIG)
      3) logistic_gate/params_box2b.yml if present
      4) logistic_gate/default_params.yml (fallback)

    Relative paths are resolved against repo root first, then logistic_gate/.
    """
    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, ".."))

    if config_path is None:
        config_path = os.getenv("CRI_LOGISTIC_CONFIG") or os.getenv("CRI_BOX2B_CONFIG")

    if config_path is None:
        cand_box2b = os.path.join(here, "params_box2b.yml")
        cand_default = os.path.join(here, "default_params.yml")
        config_path = cand_box2b if os.path.exists(cand_box2b) else cand_default

    if not os.path.isabs(config_path):
        cand_repo = os.path.join(repo, config_path)
        cand_here = os.path.join(here, config_path)
        if os.path.exists(cand_repo):
            config_path = cand_repo
        elif os.path.exists(cand_here):
            config_path = cand_here

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find logistic YAML config: {config_path}")

    return os.path.abspath(config_path)


def load_params(config_path: str | None = None) -> Tuple[dict, str]:
    """Load YAML and return (params_dict, abs_path)."""
    path = _resolve_params_path(config_path)

    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    p = obj["logistic"] if isinstance(obj, dict) and "logistic" in obj else obj
    if not isinstance(p, dict):
        raise ValueError("YAML did not parse into a dict under key 'logistic' (or as a top-level dict).")

    # Back-compat: older keys
    if "q_min" not in p and "p_min" in p:
        p["q_min"] = p["p_min"]
    if "q_max" not in p and "p_max" in p:
        p["q_max"] = p["p_max"]

    # Defaults
    p.setdefault("seed", 52)
    p.setdefault("q_min", 0.0)
    p.setdefault("q_max", 1.0)
    p.setdefault("n_points", 400)
    p.setdefault("n_bins", 15)
    p.setdefault("q_sampling", "random")  # "uniform" | "random"
    p.setdefault("use_two_conditions", True)

    # Gate params
    p.setdefault("alpha", 0.05)

    # p0 model defaults
    p.setdefault("p0_mode", "explicit")    # "explicit" | "gaussian"
    p.setdefault("p0_shape", "dip")        # "dip" | "bump" (gaussian mode only)

    # Gaussian parameters
    p.setdefault("p_base", 0.55)
    p.setdefault("a0", 0.50)
    p.setdefault("sigma_a", 0.18)
    p.setdefault("delta_p0", 0.05)

    # Explicit p0 values (used in explicit mode)
    p.setdefault("p0_a1", 0.50)
    p.setdefault("p0_a2", 0.55)

    # Arousal levels + trial counts
    p.setdefault("a1", 0.30)
    p.setdefault("a2", 0.70)
    p.setdefault("n_trials_a1", 60)
    p.setdefault("n_trials_a2", 60)

    return p, path


def _validate_params(p: Dict[str, Any]) -> None:
    """Fail fast on parameter inconsistencies that would enable figure/data contradictions."""
    # Numeric domain checks
    q_min = float(p["q_min"]); q_max = float(p["q_max"])
    if not (0.0 <= q_min < q_max <= 1.0):
        raise ValueError(f"Require 0 <= q_min < q_max <= 1. Got q_min={q_min}, q_max={q_max}.")

    n_points = int(p["n_points"])
    if n_points < 2:
        raise ValueError(f"Require n_points >= 2. Got {n_points}.")

    alpha = float(p["alpha"])
    if not (alpha > 0.0):
        raise ValueError(f"Require alpha > 0. Got {alpha}.")

    use_two = bool(p.get("use_two_conditions", True))
    a1 = float(p["a1"])
    if not (0.0 <= a1 <= 1.0):
        raise ValueError(f"Require a1 in [0,1]. Got {a1}.")
    if use_two:
        a2 = float(p["a2"])
        if not (0.0 <= a2 <= 1.0):
            raise ValueError(f"Require a2 in [0,1]. Got {a2}.")
        if np.isclose(a1, a2):
            raise ValueError(f"Two-condition mode requires a1 != a2. Got a1≈a2≈{a1}.")

    n1 = int(p["n_trials_a1"]); n2 = int(p["n_trials_a2"])
    if n1 <= 0 or (use_two and n2 <= 0):
        raise ValueError(f"Trial counts must be positive. Got n_trials_a1={n1}, n_trials_a2={n2}.")

    qs = str(p.get("q_sampling", "random")).lower()
    if qs not in {"uniform", "random"}:
        raise ValueError(f"q_sampling must be 'uniform' or 'random'. Got {qs}.")

    mode = str(p.get("p0_mode", "explicit")).lower()
    if mode not in {"explicit", "gaussian"}:
        raise ValueError(f"p0_mode must be 'explicit' or 'gaussian'. Got {mode}.")

    if mode == "explicit":
        p0_a1 = float(p["p0_a1"])
        if not (0.0 <= p0_a1 <= 1.0):
            raise ValueError(f"Explicit mode requires p0_a1 in [0,1]. Got {p0_a1}.")
        if use_two:
            p0_a2 = float(p["p0_a2"])
            if not (0.0 <= p0_a2 <= 1.0):
                raise ValueError(f"Explicit mode requires p0_a2 in [0,1]. Got {p0_a2}.")
    else:
        # gaussian mode
        p_base = float(p["p_base"])
        a0 = float(p["a0"])
        sigma_a = float(p["sigma_a"])
        delta = float(p["delta_p0"])
        shape = str(p.get("p0_shape", "dip")).lower()
        if shape not in {"dip", "bump"}:
            raise ValueError(f"p0_shape must be 'dip' or 'bump'. Got {shape}.")
        if not (0.0 <= p_base <= 1.0):
            raise ValueError(f"Gaussian mode requires p_base in [0,1]. Got {p_base}.")
        if not (0.0 <= a0 <= 1.0):
            raise ValueError(f"Gaussian mode requires a0 in [0,1]. Got {a0}.")
        if not (sigma_a > 0.0):
            raise ValueError(f"Gaussian mode requires sigma_a > 0. Got {sigma_a}.")
        if not (0.0 <= delta <= 1.0):
            raise ValueError(f"Gaussian mode requires delta_p0 in [0,1]. Got {delta}.")


def parse_args():
    ap = argparse.ArgumentParser(description="Simulate Box-2(b) logistic gate trials.")
    ap.add_argument(
        "--config",
        default=os.getenv("CRI_LOGISTIC_CONFIG") or os.getenv("CRI_BOX2B_CONFIG"),
        help="Path to YAML config (default: env CRI_LOGISTIC_CONFIG; else params_box2b.yml if present; else default_params.yml).",
    )
    return ap.parse_args()


# ---------------------------- model ------------------------------------------
def logistic(q: np.ndarray, p0: float, alpha: float) -> np.ndarray:
    """Stable logistic gate: G(q) = expit((q - p0)/alpha)."""
    alpha = max(float(alpha), 1e-12)
    return expit((np.asarray(q, dtype=float) - float(p0)) / alpha)


def p0_of_a(a: float, p: dict) -> float:
    """
    Compute p0(a) under either:
      - explicit: use p0_a1 / p0_a2 corresponding to a1 / a2
      - gaussian: p_base ± delta_p0 * exp(-(a-a0)^2/(2*sigma_a^2))
    """
    mode = str(p.get("p0_mode", "explicit")).lower()
    a = float(a)

    if mode == "explicit":
        a1 = float(p["a1"])
        a2 = float(p["a2"])
        if np.isclose(a, a1):
            return float(p["p0_a1"])
        if np.isclose(a, a2):
            return float(p["p0_a2"])
        # Fallback: choose nearest configured condition
        return float(p["p0_a1"]) if abs(a - a1) <= abs(a - a2) else float(p["p0_a2"])

    # gaussian dip/bump
    p_base = float(p.get("p_base", 0.55))
    a0 = float(p.get("a0", 0.50))
    sigma_a = max(float(p.get("sigma_a", 0.18)), 1e-6)
    delta = float(p.get("delta_p0", 0.05))
    shape = str(p.get("p0_shape", "dip")).lower()

    bump = np.exp(-0.5 * ((a - a0) / sigma_a) ** 2)
    val = p_base - delta * bump if shape == "dip" else p_base + delta * bump
    return float(np.clip(val, 0.0, 1.0))


def _sample_q(rng: np.random.Generator, n: int, q_min: float, q_max: float, mode: str) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "uniform":
        return np.linspace(q_min, q_max, int(n))
    return rng.uniform(q_min, q_max, size=int(n))


# ---------------------------- main -------------------------------------------
def main() -> None:
    load_state()
    args = parse_args()

    p, params_path = load_params(args.config)
    _validate_params(p)

    rng = np.random.default_rng(int(p["seed"]))
    save_state()

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    # --- compute run hash from the logistic params dict (canonicalised) ---
    run_hash = _compute_run_hash(p)

    # Grid for dense (noiseless) curves
    q_min = float(p["q_min"])
    q_max = float(p["q_max"])
    n_points = int(p["n_points"])
    q_grid = np.linspace(q_min, q_max, n_points)

    a1 = float(p["a1"])
    a2 = float(p["a2"])
    alpha = float(p["alpha"])

    # Compute p0 per condition
    p0_1 = p0_of_a(a1, p)
    use_two = bool(p.get("use_two_conditions", True))
    p0_2 = p0_of_a(a2, p) if use_two else None

    # Consistency guard: if explicit mode, p0_used must equal the explicit YAML values
    mode = str(p.get("p0_mode", "explicit")).lower()
    tol = float(p.get("consistency_tol", 1e-9))
    if mode == "explicit":
        if abs(p0_1 - float(p["p0_a1"])) > tol:
            raise RuntimeError("Explicit mode inconsistency: p0_used(a1) != p0_a1 from YAML.")
        if use_two and p0_2 is not None and abs(float(p0_2) - float(p["p0_a2"])) > tol:
            raise RuntimeError("Explicit mode inconsistency: p0_used(a2) != p0_a2 from YAML.")
    else:
        # If user left explicit p0_a1/p0_a2 in YAML while in gaussian mode, warn if they conflict.
        # This prevents silent “parameterisation drift” accusations later.
        if "p0_a1" in p and abs(p0_1 - float(p["p0_a1"])) > 1e-6:
            print("[WARN] gaussian mode: p0_a1 in YAML differs from derived p0_used(a1). "
                  "Consider removing p0_a1/p0_a2 keys or switching to explicit mode for Box2b.")
        if use_two and p0_2 is not None and "p0_a2" in p and abs(float(p0_2) - float(p["p0_a2"])) > 1e-6:
            print("[WARN] gaussian mode: p0_a2 in YAML differs from derived p0_used(a2). "
                  "Consider removing p0_a1/p0_a2 keys or switching to explicit mode for Box2b.")

    derived: Dict[str, Any] = {
        "q_min": q_min,
        "q_max": q_max,
        "n_points": n_points,
        "a1": a1,
        "a2": (a2 if use_two else None),
        "alpha": alpha,
        "p0_mode": str(p.get("p0_mode", "explicit")),
        "p0_shape": (str(p.get("p0_shape", "")) if str(p.get("p0_mode", "explicit")).lower() == "gaussian" else None),
        "p0_used_a1": float(p0_1),
        "p0_used_a2": (float(p0_2) if p0_2 is not None else None),
    }

    # Write run manifest FIRST
    repo_root = ROOT  # logistic_gate/.. is repository root in your layout
    manifest_path = _write_run_manifest(
        out_dir=out_dir,
        run_hash=run_hash,
        config_path=params_path,
        logistic_params=p,
        derived=derived,
        repo_root=repo_root,
    )

    # Ground-truth curves
    G_a1 = logistic(q_grid, p0_1, alpha)
    curve_data = {
        "q": q_grid,
        "G_a1": G_a1,
        "run_hash": np.full(q_grid.shape[0], run_hash, dtype=object),
    }
    if use_two:
        G_a2 = logistic(q_grid, float(p0_2), alpha)
        curve_data["G_a2"] = G_a2

    # Simulate trials
    trials = []

    # Condition a1
    q_a1 = _sample_q(rng, int(p["n_trials_a1"]), q_min, q_max, p.get("q_sampling", "random"))
    prob_a1 = logistic(q_a1, p0_1, alpha)
    y_a1 = rng.binomial(1, prob_a1)
    trials.append(pd.DataFrame({
        "q": q_a1,
        "y": y_a1.astype(int),
        "a": np.full_like(q_a1, a1, dtype=float),
        "p0": np.full_like(q_a1, p0_1, dtype=float),
        "run_hash": np.full(q_a1.shape[0], run_hash, dtype=object),
    }))

    # Optional condition a2
    if use_two:
        q_a2 = _sample_q(rng, int(p["n_trials_a2"]), q_min, q_max, p.get("q_sampling", "random"))
        prob_a2 = logistic(q_a2, float(p0_2), alpha)
        y_a2 = rng.binomial(1, prob_a2)
        trials.append(pd.DataFrame({
            "q": q_a2,
            "y": y_a2.astype(int),
            "a": np.full_like(q_a2, a2, dtype=float),
            "p0": np.full_like(q_a2, float(p0_2), dtype=float),
            "run_hash": np.full(q_a2.shape[0], run_hash, dtype=object),
    }))

    # Write dense noiseless curves
    pd.DataFrame(curve_data).to_csv(os.path.join(out_dir, "logistic_curve.csv"), index=False)

    # Write trial-wise data
    df_trials = pd.concat(trials, ignore_index=True).sort_values("q").reset_index(drop=True)
    df_trials.to_csv(os.path.join(out_dir, "logistic_trials.csv"), index=False)

    # Bin means (visualization only)
    bins = int(p["n_bins"])
    edges = np.linspace(q_min, q_max, bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    rows = []
    for a_val, df_a in df_trials.groupby("a"):
        counts, _ = np.histogram(df_a["q"].values, bins=edges)
        sums, _ = np.histogram(df_a["q"].values, bins=edges, weights=df_a["y"].astype(float).values)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, sums / counts, np.nan)

        for c, m, n in zip(bin_centers, means, counts):
            if int(n) > 0:
                rows.append({
                    "q_bin_center": float(c),
                    "rate_mean": float(m),
                    "n_bin": int(n),
                    "a": float(a_val),
                    "run_hash": run_hash,
                })

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "logistic_bins.csv"), index=False)

    print(f"Saved logistic_curve.csv, logistic_trials.csv, logistic_bins.csv → {out_dir}")
    print(f"Wrote run manifest → {manifest_path}")
    print(f"run_hash={run_hash}")
    print(f"p0(a1={a1:.3f})={p0_1:.4f}" + ("" if p0_2 is None else f" | p0(a2={a2:.3f})={float(p0_2):.4f}"))


if __name__ == "__main__":
    main()

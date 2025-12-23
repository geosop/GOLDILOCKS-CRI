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

Writes:
  - logistic_gate/output/logistic_curve.csv
      q, G_a1, [G_a2]
  - logistic_gate/output/logistic_trials.csv
      q, y, a, p0   (trial-wise Bernoulli outcomes + the p0 used)
  - logistic_gate/output/logistic_bins.csv
      q_bin_center, rate_mean, n_bin, a  (bin means for visualization ONLY)
"""
from __future__ import annotations

import os
import sys
import yaml
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


# ----------------------------- config ----------------------------------------
def load_params() -> dict:
    here = os.path.dirname(__file__)
    path = os.path.join(here, "default_params.yml")
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    p = obj["logistic"] if isinstance(obj, dict) and "logistic" in obj else obj
    if not isinstance(p, dict):
        raise ValueError("default_params.yml did not parse into a dict under key 'logistic'.")

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

    # p0 model defaults (kept back-compatible)
    p.setdefault("p0_mode", "explicit")    # "explicit" | "gaussian"
    p.setdefault("p0_shape", "dip")        # "dip" | "bump" (gaussian mode only)
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

    return p


# ---------------------------- model -----------------------------------------
def logistic(q: np.ndarray, p0: float, alpha: float) -> np.ndarray:
    """
    Stable logistic gate:
        G(q) = expit((q - p0)/alpha)
    """
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
        # evenly spaced values (deterministic given n)
        return np.linspace(q_min, q_max, int(n))
    # default: random uniform samples
    return rng.uniform(q_min, q_max, size=int(n))


# ---------------------------- main ------------------------------------------
def main() -> None:
    load_state()
    p = load_params()
    rng = np.random.default_rng(int(p["seed"]))
    save_state()

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    # Grid for dense (noiseless) curves
    q_min = float(p["q_min"])
    q_max = float(p["q_max"])
    n_points = int(p["n_points"])
    q_grid = np.linspace(q_min, q_max, n_points)

    a1 = float(p["a1"])
    a2 = float(p["a2"])
    alpha = float(p["alpha"])

    # Compute p0 per condition (supports gaussian mode)
    p0_1 = p0_of_a(a1, p)
    p0_2 = p0_of_a(a2, p) if bool(p.get("use_two_conditions", True)) else None

    # Ground-truth curves
    G_a1 = logistic(q_grid, p0_1, alpha)
    curve_data = {"q": q_grid, "G_a1": G_a1}
    if bool(p.get("use_two_conditions", True)):
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
    }))

    # Optional condition a2
    if bool(p.get("use_two_conditions", True)):
        q_a2 = _sample_q(rng, int(p["n_trials_a2"]), q_min, q_max, p.get("q_sampling", "random"))
        prob_a2 = logistic(q_a2, float(p0_2), alpha)
        y_a2 = rng.binomial(1, prob_a2)
        trials.append(pd.DataFrame({
            "q": q_a2,
            "y": y_a2.astype(int),
            "a": np.full_like(q_a2, a2, dtype=float),
            "p0": np.full_like(q_a2, float(p0_2), dtype=float),
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
                })

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "logistic_bins.csv"), index=False)

    print(f"Saved logistic_curve.csv, logistic_trials.csv, logistic_bins.csv → {out_dir}")
    print(f"p0(a1={a1:.3f})={p0_1:.4f}" + ("" if p0_2 is None else f" | p0(a2={a2:.3f})={float(p0_2):.4f}"))


if __name__ == "__main__":
    main()


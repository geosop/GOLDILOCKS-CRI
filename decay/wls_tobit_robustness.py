#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decay/wls_tobit_robustness.py  •  CRI v0.3-SIM (robust; provenance-visible)

Robust artifacts showing WLS/Tobit consistency with OLS bootstrap CIs
on the SAME synthetic dataset produced by simulate_decay.py (Box-2a).

Figure:
  x-axis  →  Estimated τ_fut (ms)
  y-axis  →  Estimator (–)  [categorical: OLS / WLS / Tobit]

Provenance (FIX in this version):
  - Enforces manifest/run_hash consistency as before.
  - Renders run_hash + config path visibly on the figure (footer).
  - Embeds run_hash + config path into PDF metadata and PNG metadata (Pillow text chunks when available).

Outputs:
  - decay/output/wls_tobit_robustness.csv
  - figures/output/decay_wls_tobit_robustness.[png|pdf]
  - (optional) decay/output/wls_tobit_replicates.csv + wls_tobit_coverage.csv
"""
from __future__ import annotations

# stdlib
import os
import sys
import math
import argparse
import json
import hashlib
from typing import Any, Dict, Optional, Tuple

# third-party
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml

# Optional: Pillow for PNG metadata text chunks
try:
    from PIL import PngImagePlugin  # type: ignore
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

# --- make the repo root importable when running this file by path -------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Use the same fitting helpers as the pipeline for consistency
from decay.fit_decay import _ols_fit, _wls_fit, _tobit_fit, _fit_all  # noqa: E402


HERE = os.path.dirname(__file__)
OUTD = os.path.join(HERE, "output")
FIGD = os.path.join(os.path.dirname(HERE), "figures", "output")
os.makedirs(OUTD, exist_ok=True)
os.makedirs(FIGD, exist_ok=True)

# Embed TrueType in PDF (no Type 3 fonts)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# -----------------------------------------------------------------------------#
# Hash / manifest utilities (must match simulate_decay.py and fit_decay.py)
# -----------------------------------------------------------------------------#
def _canonical_json_bytes(obj: Any) -> bytes:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def _compute_run_hash(decay_params: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(decay_params)).hexdigest()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _short_path(path: str, keep: int = 2) -> str:
    """
    Shorten a path for on-figure provenance while keeping tail segments.
    Example: /a/b/c/d.yml -> …/c/d.yml (keep=2)
    """
    p = os.path.abspath(path)
    parts = p.replace("\\", "/").split("/")
    if len(parts) <= keep:
        return p
    return "…/" + "/".join(parts[-keep:])


# -----------------------------------------------------------------------------#
# Config resolution (same policy as simulate_decay.py / fit_decay.py)
# -----------------------------------------------------------------------------#
def _resolve_config_path(config_path: Optional[str]) -> str:
    """
    Resolve config path deterministically.
    Relative paths resolved against repo root first, then decay/.

    Priority:
      1) explicit argument
      2) env var CRI_DECAY_CONFIG
      3) decay/default_params.yml (fallback)
    """
    repo = ROOT
    here = HERE

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


def _load_yaml_decay_block(config_path: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """Load YAML and return (decay_dict, resolved_path)."""
    cfg_path = _resolve_config_path(config_path)
    with open(cfg_path, "r", encoding="utf-8-sig", errors="replace") as f:
        y = yaml.safe_load(f) or {}
    p = y["decay"] if isinstance(y, dict) and "decay" in y else y
    if not isinstance(p, dict):
        raise ValueError("Decay YAML did not parse to a dict (either top-level or under key 'decay').")
    return p, cfg_path


def _effective_sim_params_from_yaml(p_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruct the *simulation-effective* dict that simulate_decay.py hashes.
    Must stay aligned with simulate_decay.py.
    """
    seed = _safe_int(p_raw.get("seed", 52), 52)
    env_seed = os.getenv("CRI_SEED", None)
    if env_seed is not None:
        try:
            seed = int(env_seed)
        except Exception:
            pass

    return {
        "seed": int(seed),
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


def _load_fit_settings_from_yaml(p_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract fit/CI settings; separate from sim-effective dict."""
    n_boot = _safe_int(p_raw.get("n_boot", p_raw.get("n_bootstrap", 2000)), 2000)
    ci_percent = _safe_float(p_raw.get("ci_percent", 95.0), 95.0)
    A_min = _safe_float(p_raw.get("A_min", p_raw.get("epsilon_detection", 0.01)), 0.01)
    seed = _safe_int(p_raw.get("seed", 52), 52)
    env_seed = os.getenv("CRI_SEED", None)
    if env_seed is not None:
        try:
            seed = int(env_seed)
        except Exception:
            pass
    return {
        "seed": int(seed),
        "n_boot": int(n_boot),
        "ci_percent": float(ci_percent),
        "A_min": float(A_min),
    }


def _enforce_manifest(decay_out: str, sim_effective: Dict[str, Any], params_path: str) -> str:
    """
    Enforce:
      - decay/output/run_manifest.json exists and contains run_hash
      - hash(sim_effective) == manifest.run_hash
      - manifest.params_path matches resolved params_path
    Returns run_hash.
    """
    man_path = os.path.join(decay_out, "run_manifest.json")
    if not os.path.exists(man_path):
        raise RuntimeError(
            "Missing decay/output/run_manifest.json. "
            "Rerun decay/simulate_decay.py before running wls_tobit_robustness.py."
        )
    manifest = _read_json(man_path)
    if "run_hash" not in manifest:
        raise RuntimeError("run_manifest.json missing key 'run_hash'.")
    run_hash = str(manifest["run_hash"])

    yaml_hash = _compute_run_hash(sim_effective)
    if yaml_hash != run_hash:
        raise RuntimeError(
            "Simulation-effective decay params hash does not match run_manifest.json.\n"
            f"  YAML-derived hash: {yaml_hash}\n"
            f"  manifest.run_hash: {run_hash}\n"
            "Fix: delete decay/output/* and regenerate from the intended YAML."
        )

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
# Local simulator (used only for optional Monte-Carlo repeats)
# -----------------------------------------------------------------------------#
def _simulate_once(
    seed: int,
    A0: float,
    tau_f: float,
    noise_log: float,
    delta_start: float,
    delta_end: float,
    delta_step: float,
    n_rep: int,
    apply_censoring: bool = False,
    A_min: float = 0.01,
):
    """Local simulator matching simulate_decay.py semantics; used only for Monte-Carlo repeats."""
    rng = np.random.default_rng(int(seed))
    deltas = np.arange(delta_start, delta_end + 1e-12, delta_step)
    mu = np.log(A0) - deltas / tau_f
    rows = []
    c = float(np.log(A_min)) if apply_censoring else None

    for d, m in zip(deltas, mu):
        y = m + rng.normal(0.0, noise_log, size=int(n_rep))
        if c is not None:
            y = np.maximum(y, c)
        rows.extend({"delta": float(d), "lnA_pre_raw": float(v)} for v in y)

    df_raw = pd.DataFrame(rows)
    agg = df_raw.groupby("delta", as_index=False).agg(
        lnA_pre=("lnA_pre_raw", "mean"),
        sd=("lnA_pre_raw", "std"),
        n=("lnA_pre_raw", "size"),
    )
    agg["se_lnA"] = agg["sd"] / np.sqrt(agg["n"])
    return agg[["delta", "lnA_pre", "se_lnA"]], df_raw


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _bootstrap_slope_ci_ols(x: np.ndarray, y: np.ndarray, B: int = 2000, ci: float = 95.0, seed: int = 52):
    """Nonparametric bootstrap for OLS slope CI (percentile)."""
    rng = np.random.default_rng(int(seed))
    n = len(x)
    b1s = np.empty(int(B), dtype=float)
    for i in range(int(B)):
        idx = rng.integers(0, n, n)
        _, b1s[i] = _ols_fit(x[idx], y[idx])
    a = (100.0 - float(ci)) / 2.0
    lo = float(np.percentile(b1s, a))
    hi = float(np.percentile(b1s, 100.0 - a))
    return lo, hi


def _b1_to_tau_ms(b1: float) -> float:
    """Convert slope of ln(A) vs Δ (b1) to τ in ms. (Δ in seconds)"""
    if not np.isfinite(b1) or b1 >= 0:
        return float("nan")
    return (-1.0 / b1) * 1000.0


def _fit_all_from_dataset(df: pd.DataFrame, A_min: float, n_boot: int, ci_percent: float, seed: int):
    """
    Fit OLS/WLS/Tobit on provided aggregated dataset (delta, lnA_pre, se_lnA),
    and compute the OLS τ CI using the same pipeline helper (_fit_all).
    """
    x = df["delta"].astype(float).to_numpy()
    y = df["lnA_pre"].astype(float).to_numpy()
    se = df["se_lnA"].astype(float).to_numpy()

    # Point estimates (slopes)
    _, b1_ols = _ols_fit(x, y)
    _, b1_wls = _wls_fit(x, y, se)
    _, b1_tob = _tobit_fit(x, y, math.log(float(A_min)), seed=int(seed))

    tau_ols_ms = _b1_to_tau_ms(b1_ols)
    tau_wls_ms = _b1_to_tau_ms(b1_wls)
    tau_tob_ms = _b1_to_tau_ms(b1_tob)

    # OLS τ CI via the same helper used in the pipeline
    res_df, _ = _fit_all(df, A_min=A_min, n_boot=n_boot, ci_percent=ci_percent, seed=seed, n_points=200)
    ols_ci_lo_ms = float(res_df["ci_lo_ms"].iloc[0])
    ols_ci_hi_ms = float(res_df["ci_hi_ms"].iloc[0])

    # OLS slope CI (for the “slope within CI” claim)
    slo_lo, slo_hi = _bootstrap_slope_ci_ols(x, y, B=n_boot, ci=ci_percent, seed=seed)

    return {
        "b1_ols": float(b1_ols),
        "b1_wls": float(b1_wls),
        "b1_tobit": float(b1_tob),
        "slope_CI_lo_OLS": float(slo_lo),
        "slope_CI_hi_OLS": float(slo_hi),
        "tau_ols_ms": float(tau_ols_ms),
        "tau_wls_ms": float(tau_wls_ms),
        "tau_tobit_ms": float(tau_tob_ms),
        "ols_ci_lo_ms": float(ols_ci_lo_ms),
        "ols_ci_hi_ms": float(ols_ci_hi_ms),
        "wls_in_OLS_tau_CI": bool(ols_ci_lo_ms <= tau_wls_ms <= ols_ci_hi_ms),
        "tobit_in_OLS_tau_CI": bool(ols_ci_lo_ms <= tau_tob_ms <= ols_ci_hi_ms),
        "wls_in_OLS_slope_CI": bool(slo_lo <= b1_wls <= slo_hi),
        "tobit_in_OLS_slope_CI": bool(slo_lo <= b1_tob <= slo_hi),
    }


def _render_provenance_footer(
    fig: plt.Figure,
    run_hash: str,
    params_path: str,
    *,
    loc: str = "bottom-right",
    fontsize: float = 7.0,
    hash_chars: int = 12,
    path_keep: int = 2,
) -> None:
    """Render a small provenance footer inside the figure canvas."""
    shown_hash = run_hash[: max(4, int(hash_chars))]
    shown_cfg = _short_path(params_path, keep=max(1, int(path_keep)))
    prov_text = f"run_hash={shown_hash}…  |  cfg={shown_cfg}"

    loc = (loc or "bottom-right").lower()
    x = 0.99 if "right" in loc else 0.01
    y = 0.02 if "bottom" in loc else 0.98
    ha = "right" if "right" in loc else "left"
    va = "bottom" if "bottom" in loc else "top"

    fig.text(
        x, y, prov_text,
        transform=fig.transFigure,
        ha=ha, va=va,
        fontsize=float(fontsize),
        color="black",
        zorder=30,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.85, edgecolor="none"),
    )


def _plot_tau_panel(
    out: dict,
    out_png: str,
    out_pdf: str,
    run_hash: str,
    params_path: str,
    *,
    show_provenance: bool = True,
    prov_loc: str = "bottom-right",
    prov_fontsize: float = 7.0,
    prov_hash_chars: int = 12,
    prov_path_keep: int = 2,
):
    """
    Horizontal CI panel with categorical y-axis.
    - x-axis: Estimated τ_fut (ms)
    - y-axis: Estimator (–): {OLS, WLS, Tobit}
    """
    fig, ax = plt.subplots(figsize=(6.4, 2.8))

    # Leave room for footer provenance (bbox_inches="tight" can otherwise clip)
    fig.subplots_adjust(bottom=0.18)

    # Categorical rows (top→bottom)
    rows = [("OLS", 2), ("WLS", 1), ("Tobit", 0)]

    # OLS 95% CI on its own row, and as dashed guide on others
    ax.hlines(rows[0][1], out["ols_ci_lo_ms"], out["ols_ci_hi_ms"], linewidth=4, zorder=1)
    for _, y in rows[1:]:
        ax.hlines(y, out["ols_ci_lo_ms"], out["ols_ci_hi_ms"], linewidth=2, alpha=0.25, linestyle="--", zorder=0)

    # Point estimates (no explicit colors set; matplotlib defaults)
    ax.plot(out["tau_ols_ms"], rows[0][1], marker="o", ms=7, linestyle="None",
            label=f"OLS {out['tau_ols_ms']:.1f} ms", zorder=3)
    ax.plot(out["tau_wls_ms"], rows[1][1], marker="^", ms=7, linestyle="None",
            label=f"WLS {out['tau_wls_ms']:.1f} ms", zorder=3)
    ax.plot(out["tau_tobit_ms"], rows[2][1], marker="s", ms=7, linestyle="None",
            label=f"Tobit {out['tau_tobit_ms']:.1f} ms", zorder=3)

    ax.set_xlabel(r"Estimated $\tau_{\mathrm{fut}}$ (ms)")
    ax.set_ylabel("Estimator (–)")
    ax.set_yticks([r[1] for r in rows], labels=[r[0] for r in rows])

    x_vals = [
        out["ols_ci_lo_ms"], out["ols_ci_hi_ms"],
        out["tau_ols_ms"], out["tau_wls_ms"], out["tau_tobit_ms"],
    ]
    xmin, xmax = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
    pad = 0.04 * (xmax - xmin if xmax > xmin else 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)

    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_ylim(-0.6, 2.6)

    # Compact legend (single row if possible)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.0),
        mode="expand", ncol=4, frameon=False, fontsize=8,
        handlelength=1.2, handletextpad=0.5, columnspacing=0.9, labelspacing=0.2,
    )

    # Visible provenance footer (FIX)
    if bool(show_provenance):
        _render_provenance_footer(
            fig,
            run_hash=run_hash,
            params_path=params_path,
            loc=prov_loc,
            fontsize=prov_fontsize,
            hash_chars=prov_hash_chars,
            path_keep=prov_path_keep,
        )

    # Metadata
    fig.canvas.manager.set_window_title("decay_wls_tobit_robustness")
    meta_pdf = {
        "Title": "Decay robustness: WLS/Tobit vs OLS CI",
        "Author": "CRI pipeline",
        "Subject": f"run_hash={run_hash}; params_path={os.path.abspath(params_path)}",
        "Keywords": "CRI; decay; robustness; WLS; Tobit; bootstrap",
    }

    # PNG metadata text chunks (when Pillow available)
    pnginfo = None
    if _HAVE_PIL:
        try:
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("Software", "CRI pipeline")
            pnginfo.add_text("run_hash", str(run_hash))
            pnginfo.add_text("params_path", os.path.abspath(params_path))
        except Exception:
            pnginfo = None

    plt.tight_layout()

    fig.savefig(out_png, dpi=600, bbox_inches="tight", pil_kwargs=({"pnginfo": pnginfo} if pnginfo else None))
    fig.savefig(out_pdf, bbox_inches="tight", metadata=meta_pdf)
    plt.close(fig)


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser(
        description="WLS/Tobit robustness vs OLS CIs on the current synthetic dataset."
    )
    ap.add_argument(
        "--data",
        default=os.path.join(HERE, "output", "decay_data.csv"),
        help="Aggregated dataset with columns: delta, lnA_pre, se_lnA",
    )
    ap.add_argument(
        "--out-csv",
        default=os.path.join(HERE, "output", "wls_tobit_robustness.csv"),
        help="Output CSV path",
    )
    ap.add_argument(
        "--out-png",
        default=os.path.join(FIGD, "decay_wls_tobit_robustness.png"),
        help="Output figure (PNG) path",
    )
    ap.add_argument(
        "--out-pdf",
        default=os.path.join(FIGD, "decay_wls_tobit_robustness.pdf"),
        help="Output figure (PDF) path",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=0,
        help="Monte-Carlo repeats over NEW datasets (0=off; does not alter primary artifact)",
    )
    ap.add_argument("--seed", type=int, default=None, help="Base RNG seed (default from YAML / CRI_SEED)")
    ap.add_argument("--n-boot", type=int, default=None, help="Bootstrap B for CIs (default from YAML)")
    ap.add_argument("--ci", type=float, default=None, help="CI percent (default from YAML)")
    ap.add_argument(
        "--config",
        default=None,
        help="Optional YAML path; overrides CRI_DECAY_CONFIG and default_params.yml resolution.",
    )

    # Provenance rendering controls (new)
    ap.add_argument("--show-provenance", action="store_true", default=True,
                    help="Render run_hash + config path visibly on the figure (default: on).")
    ap.add_argument("--prov-loc", default=os.getenv("CRI_PROV_LOC", "bottom-right"),
                    choices=["bottom-right", "bottom-left", "top-right", "top-left"],
                    help="Where to place the provenance footer.")
    ap.add_argument("--prov-fontsize", type=float, default=float(os.getenv("CRI_PROV_FONTSIZE", 7.0)),
                    help="Font size for provenance footer.")
    ap.add_argument("--prov-hash-chars", type=int, default=int(os.getenv("CRI_PROV_HASH_CHARS", 12)),
                    help="Number of run_hash characters to show visibly.")
    ap.add_argument("--prov-path-keep", type=int, default=int(os.getenv("CRI_PROV_PATH_KEEP", 2)),
                    help="How many trailing path components to show (…/tail).")

    args = ap.parse_args()

    # Load YAML + resolve path
    p_raw, params_path = _load_yaml_decay_block(args.config)

    # Reconstruct sim-effective params and enforce manifest hash
    sim_effective = _effective_sim_params_from_yaml(p_raw)
    run_hash = _enforce_manifest(OUTD, sim_effective, params_path)

    # Fit settings (can be overridden by args)
    fit_settings = _load_fit_settings_from_yaml(p_raw)
    base_seed = fit_settings["seed"] if args.seed is None else int(args.seed)
    n_boot = fit_settings["n_boot"] if args.n_boot is None else int(args.n_boot)
    ci_percent = fit_settings["ci_percent"] if args.ci is None else float(args.ci)
    A_min = fit_settings["A_min"]

    # -------- Single-dataset check on the SAME synthetic data already produced
    if not os.path.isfile(args.data):
        raise FileNotFoundError(
            f"Expected dataset not found: {args.data}\nRun decay/simulate_decay.py first."
        )

    df = pd.read_csv(args.data)
    needed = {"delta", "lnA_pre", "se_lnA"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{args.data} must contain columns {needed}, got {df.columns.tolist()}")

    out = _fit_all_from_dataset(df, A_min=A_min, n_boot=n_boot, ci_percent=ci_percent, seed=base_seed)

    # Stamp provenance into CSV
    out_row = {
        "run_hash": run_hash,
        "params_path": os.path.abspath(params_path),
        **out,
        "n_bootstrap": int(n_boot),
        "ci_percent": float(ci_percent),
        "seed": int(base_seed),
    }
    pd.DataFrame([out_row]).to_csv(args.out_csv, index=False)

    print(f"Wrote {args.out_csv}")
    print(pd.DataFrame([out_row]).to_string(index=False))

    _plot_tau_panel(
        out,
        args.out_png,
        args.out_pdf,
        run_hash=run_hash,
        params_path=params_path,
        show_provenance=bool(args.show_provenance),
        prov_loc=str(args.prov_loc),
        prov_fontsize=float(args.prov_fontsize),
        prov_hash_chars=int(args.prov_hash_chars),
        prov_path_keep=int(args.prov_path_keep),
    )
    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_pdf}")

    # -------- Optional Monte-Carlo coverage over NEW datasets (fresh sims)
    R = int(args.repeats)
    if R > 0:
        rows = []
        for r in range(R):
            df_r, _ = _simulate_once(
                seed=base_seed + 101 + r,
                A0=sim_effective["A0"],
                tau_f=sim_effective["tau_f"],
                noise_log=sim_effective["noise_log"],
                delta_start=sim_effective["delta_start"],
                delta_end=sim_effective["delta_end"],
                delta_step=sim_effective["delta_step"],
                n_rep=sim_effective["n_rep"],
                apply_censoring=bool(sim_effective["apply_censoring"]),
                A_min=float(sim_effective["A_min"]),
            )
            out_r = _fit_all_from_dataset(
                df_r, A_min=A_min, n_boot=n_boot, ci_percent=ci_percent, seed=base_seed + 202 + r
            )
            rows.append(
                {
                    "run_hash": run_hash,
                    "params_path": os.path.abspath(params_path),
                    "replicate": int(r),
                    **out_r,
                }
            )

        MC = pd.DataFrame(rows)
        MC_path = os.path.join(OUTD, "wls_tobit_replicates.csv")
        MC.to_csv(MC_path, index=False)

        summary = pd.DataFrame(
            [
                {
                    "run_hash": run_hash,
                    "params_path": os.path.abspath(params_path),
                    "repeats": int(R),
                    "WLS_in_OLS_tau_CI_rate": float(np.mean(MC["wls_in_OLS_tau_CI"])),
                    "Tobit_in_OLS_tau_CI_rate": float(np.mean(MC["tobit_in_OLS_tau_CI"])),
                    "WLS_in_OLS_slope_CI_rate": float(np.mean(MC["wls_in_OLS_slope_CI"])),
                    "Tobit_in_OLS_slope_CI_rate": float(np.mean(MC["tobit_in_OLS_slope_CI"])),
                }
            ]
        )
        SUM_path = os.path.join(OUTD, "wls_tobit_coverage.csv")
        summary.to_csv(SUM_path, index=False)

        print("\nMonte-Carlo coverage summary:")
        print(summary.to_string(index=False))
        print(f"Wrote {MC_path}")
        print(f"Wrote {SUM_path}")


if __name__ == "__main__":
    main()

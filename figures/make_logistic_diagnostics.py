# -*- coding: utf-8 -*-
"""
figures/make_logistic_diagnostics.py  •  CRI (SI diagnostics)

Panel (top): model-free kernel smoother (faint) overlaid on fitted logistic curves.
Panel (bottom): calibration curves (observed vs predicted) with Wilson intervals,
plus Brier and ECE annotations per condition.

Anti-contradiction / robustness upgrades:
  A) Require logistic_gate/output/run_manifest.json.
  B) Enforce YAML(effective logistic params) -> run_hash matches manifest.run_hash.
  C) Enforce all input CSV run_hash matches manifest.run_hash (strict by default).
  D) Enforce arousal levels in calibration match YAML a1/a2 (two-condition).
  E) Kernel is optional; if present must match same run_hash and required columns.
  F) Stamp provenance in footer + PDF metadata.

Axes labels include dimensionless units:
  - Top: q (–), G(q) (–)
  - Bottom: Predicted probability (–), Observed rate (–)

Outputs:
  - figures/output/Box2b_logistic_diagnostics.pdf
  - figures/output/Box2b_logistic_diagnostics.png
"""
from __future__ import annotations

import os
import json
import argparse
import yaml
import hashlib
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   9,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "legend.fontsize": 7,     # global default; ax2 uses smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

EPS = 1e-12


# -----------------------------------------------------------------------------#
# Hash / manifest utilities (must match simulate_logistic.py / fit_logistic.py)
# -----------------------------------------------------------------------------#
def _canonical_json_bytes(obj: Any) -> bytes:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def _compute_run_hash(logistic_params_effective: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(logistic_params_effective)).hexdigest()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------#
# Config resolution (mirror simulate_logistic.py)
# -----------------------------------------------------------------------------#
def _resolve_params_path(config_path: Optional[str] = None) -> str:
    """
    Precedence:
      1) explicit --config
      2) env var CRI_LOGISTIC_CONFIG (or CRI_BOX2B_CONFIG)
      3) logistic_gate/params_box2b.yml if present
      4) logistic_gate/default_params.yml (fallback)

    Relative paths are resolved against repo root first, then logistic_gate/.
    """
    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, os.pardir))
    gate = os.path.join(repo, "logistic_gate")

    if config_path is None:
        config_path = os.getenv("CRI_LOGISTIC_CONFIG") or os.getenv("CRI_BOX2B_CONFIG")

    if config_path is None:
        cand_box2b = os.path.join(gate, "params_box2b.yml")
        cand_default = os.path.join(gate, "default_params.yml")
        config_path = cand_box2b if os.path.exists(cand_box2b) else cand_default

    if not os.path.isabs(config_path):
        cand_repo = os.path.join(repo, config_path)
        cand_gate = os.path.join(gate, config_path)
        if os.path.exists(cand_repo):
            config_path = cand_repo
        elif os.path.exists(cand_gate):
            config_path = cand_gate

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find logistic YAML config: {config_path}")

    return os.path.abspath(config_path)


def _load_yaml_logistic_section(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    p = obj["logistic"] if isinstance(obj, dict) and "logistic" in obj else obj
    if not isinstance(p, dict):
        raise ValueError("YAML did not parse into a dict (expected either top-level dict or key 'logistic').")
    return p


def _apply_simulate_defaults(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply EXACT simulate_logistic.py defaults, and nothing else.
    This dict is what must be hashed to reproduce manifest.run_hash.
    """
    p = dict(p)  # copy

    # Back-compat aliases
    if "q_min" not in p and "p_min" in p:
        p["q_min"] = p["p_min"]
    if "q_max" not in p and "p_max" in p:
        p["q_max"] = p["p_max"]

    p.setdefault("seed", 52)
    p.setdefault("q_min", 0.0)
    p.setdefault("q_max", 1.0)
    p.setdefault("n_points", 400)
    p.setdefault("n_bins", 15)
    p.setdefault("q_sampling", "random")
    p.setdefault("use_two_conditions", True)

    p.setdefault("alpha", 0.05)

    p.setdefault("p0_mode", "explicit")
    p.setdefault("p0_shape", "dip")
    p.setdefault("p_base", 0.55)
    p.setdefault("a0", 0.50)
    p.setdefault("sigma_a", 0.18)
    p.setdefault("delta_p0", 0.05)

    p.setdefault("p0_a1", 0.50)
    p.setdefault("p0_a2", 0.55)

    p.setdefault("a1", 0.30)
    p.setdefault("a2", 0.70)
    p.setdefault("n_trials_a1", 60)
    p.setdefault("n_trials_a2", 60)

    # Export default (used for PNG dpi)
    p.setdefault("figure_dpi", 1200)

    return p


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _require(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _dedupe_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if not l or l.strip() == "" or l in seen:
            continue
        seen.add(l)
        h2.append(h)
        l2.append(l)
    return h2, l2


def _get_df_run_hash(df: pd.DataFrame, *, strict: bool, name: str) -> Optional[str]:
    if "run_hash" not in df.columns:
        if strict:
            raise RuntimeError(f"{name} is missing required run_hash column (strict mode). Regenerate outputs.")
        return None
    uniq = df["run_hash"].dropna().astype(str).unique()
    if len(uniq) == 0:
        if strict:
            raise RuntimeError(f"{name} has run_hash column but it is empty.")
        return None
    if len(uniq) != 1:
        raise RuntimeError(f"{name} has multiple run_hash values: {uniq}. Mixed outputs detected.")
    return str(uniq[0])


def _extract_unique_arousal(df: pd.DataFrame) -> np.ndarray:
    return np.sort(np.unique(np.round(df["a"].values.astype(float), 6)))


def _enforce_manifest_and_hashes(
    gate_out: str,
    params_effective: dict,
    params_path: str,
    dfs: Dict[str, pd.DataFrame],
    *,
    strict_csv_hash: bool,
    atol_a: float = 1e-6,
) -> str:
    """
    Enforce:
      - run_manifest.json exists and contains run_hash
      - YAML(effective) run_hash == manifest.run_hash
      - (optional) manifest.config_sha256 matches YAML bytes (if present)
      - all CSV run_hash == manifest.run_hash (strict by default)
      - arousal levels in calibration match YAML (two-condition)
    Returns: run_hash (manifest)
    """
    man_path = os.path.join(gate_out, "run_manifest.json")
    if not os.path.exists(man_path):
        raise RuntimeError(
            "Missing logistic_gate/output/run_manifest.json. "
            "Rerun logistic_gate/simulate_logistic.py (and fit_logistic.py) before diagnostics."
        )
    manifest = _read_json(man_path)
    if "run_hash" not in manifest:
        raise RuntimeError("run_manifest.json missing key 'run_hash'.")
    run_hash = str(manifest["run_hash"])

    # Internal manifest self-consistency (if logistic_params exists)
    if "logistic_params" in manifest and isinstance(manifest["logistic_params"], dict):
        rh_internal = _compute_run_hash(manifest["logistic_params"])
        if rh_internal != run_hash:
            raise RuntimeError(
                "run_manifest.json is internally inconsistent:\n"
                f"  compute_hash(manifest.logistic_params)={rh_internal}\n"
                f"  manifest.run_hash={run_hash}\n"
                "Fix: rerun simulate_logistic.py to regenerate outputs."
            )

    # Optional: verify YAML bytes hash if present
    if "config_sha256" in manifest and isinstance(manifest["config_sha256"], str):
        sha_now = _sha256_of_file(params_path)
        if sha_now != str(manifest["config_sha256"]):
            raise RuntimeError(
                "YAML file bytes do not match run_manifest.json config_sha256.\n"
                f"  current YAML sha256: {sha_now}\n"
                f"  manifest sha256:     {manifest['config_sha256']}\n"
                "Fix: delete logistic_gate/output/* and regenerate from the intended YAML."
            )

    yaml_hash = _compute_run_hash(params_effective)
    if yaml_hash != run_hash:
        raise RuntimeError(
            "YAML-derived run_hash does not match run_manifest.json.\n"
            f"  YAML run_hash:     {yaml_hash}\n"
            f"  manifest run_hash: {run_hash}\n"
            "Fix: delete logistic_gate/output/* and regenerate from the intended YAML."
        )

    # CSV -> hash checks
    for name, df in dfs.items():
        h = _get_df_run_hash(df, strict=strict_csv_hash, name=name)
        if h is None:
            continue
        if h != run_hash:
            raise RuntimeError(
                f"run_hash mismatch for {name}:\n"
                f"  {name}.run_hash={h}\n"
                f"  manifest.run_hash={run_hash}\n"
                "Fix: delete logistic_gate/output/* and regenerate all outputs in one run."
            )

    # Arousal-level checks (two-condition): enforce against calibration arousal levels
    use_two = bool(params_effective.get("use_two_conditions", True))
    if use_two:
        a1_y, a2_y = float(params_effective["a1"]), float(params_effective["a2"])
        if "logistic_calibration.csv" not in dfs:
            raise RuntimeError("Diagnostics expects logistic_calibration.csv for arousal checks.")

        dfc = dfs["logistic_calibration.csv"]
        if "a" not in dfc.columns:
            raise RuntimeError("Calibration file missing 'a' column (required for two-condition checks).")

        a_vals = _extract_unique_arousal(dfc)
        if len(a_vals) < 2:
            raise RuntimeError("Two-condition YAML but calibration contains <2 arousal values.")

        a1_d = float(a_vals[np.argmin(np.abs(a_vals - a1_y))])
        a2_d = float(a_vals[np.argmin(np.abs(a_vals - a2_y))])
        if (abs(a1_d - a1_y) > atol_a) or (abs(a2_d - a2_y) > atol_a) or np.isclose(a1_d, a2_d):
            raise RuntimeError(
                "Arousal levels in calibration do not match YAML a1/a2 within tolerance.\n"
                f"  YAML a1,a2: {a1_y:.6f}, {a2_y:.6f}\n"
                f"  Calibration levels: {a_vals}\n"
                "Fix: delete logistic_gate/output/* and regenerate."
            )

    return run_hash


def _clean_calib(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize calibration rows:
      - clip to [0,1]
      - drop non-finite
      - ensure lo<=hi and obs_rate within [lo,hi]
      - compute non-negative asymmetric errors
    """
    df = df.copy()
    needed = {"pred_mean", "obs_rate", "lo", "hi", "n_bin", "a"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"logistic_calibration.csv missing columns: {sorted(missing)}")

    for col in ("pred_mean", "obs_rate", "lo", "hi"):
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0.0, 1.0)
    df["n_bin"] = pd.to_numeric(df["n_bin"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["pred_mean", "obs_rate", "lo", "hi", "n_bin", "a"])

    lo = np.minimum(df["lo"].values, df["hi"].values)
    hi = np.maximum(df["lo"].values, df["hi"].values)
    df["lo"], df["hi"] = lo, hi

    df["obs_rate"] = np.maximum(df["obs_rate"].values, df["lo"].values)
    df["obs_rate"] = np.minimum(df["obs_rate"].values, df["hi"].values)

    df["err_lo"] = np.maximum(df["obs_rate"].values - df["lo"].values, 0.0)
    df["err_hi"] = np.maximum(df["hi"].values - df["obs_rate"].values, 0.0)

    df = df[(df["n_bin"] > 0) & (df["err_lo"] >= 0) & (df["err_hi"] >= 0)]
    return df


def _metric_val(df_metrics: pd.DataFrame, name: str, a_label: str) -> float:
    row = df_metrics[(df_metrics["metric"] == name) & (df_metrics["a"] == a_label)]
    return float(row["value"].iloc[0]) if not row.empty else float("nan")


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def parse_args():
    p = argparse.ArgumentParser(description="Create SI logistic diagnostics figure (kernel smoother + calibration).")

    p.add_argument(
        "--config",
        default=os.getenv("CRI_LOGISTIC_CONFIG") or os.getenv("CRI_BOX2B_CONFIG"),
        help="Path to YAML config (default: env; else params_box2b.yml if present; else default_params.yml).",
    )

    p.add_argument("--panel-label", default=os.getenv("CRI_PANEL_LABEL", ""))  # e.g. "(SI)"
    p.add_argument("--panel-x", type=float, default=float(os.getenv("CRI_PANEL_X", 0.008)))
    p.add_argument("--panel-y", type=float, default=float(os.getenv("CRI_PANEL_Y", 0.985)))

    p.add_argument("--out-stem", default=os.getenv("CRI_BOX2B_DIAG_STEM", "Box2b_logistic_diagnostics"))
    p.add_argument("--no-footer", action="store_true", help="Disable provenance footer.")

    p.add_argument("--allow-missing-run-hash", action="store_true",
                   help="Backward-compat: allow CSVs without run_hash columns (not recommended).")
    return p.parse_args()


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main():
    args = parse_args()

    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, os.pardir))
    gate = os.path.join(repo, "logistic_gate")
    gate_out = os.path.join(gate, "output")

    # Load YAML with same precedence, then apply simulate defaults for hashing
    params_path = _resolve_params_path(args.config)
    p_raw = _load_yaml_logistic_section(params_path)
    params = _apply_simulate_defaults(p_raw)

    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    # Required inputs
    band  = pd.read_csv(_require(os.path.join(gate_out, "logistic_band.csv"), "logistic_band.csv"))
    calib = pd.read_csv(_require(os.path.join(gate_out, "logistic_calibration.csv"), "logistic_calibration.csv"))
    metr  = pd.read_csv(_require(os.path.join(gate_out, "logistic_calibration_metrics.csv"),
                                 "logistic_calibration_metrics.csv"))

    # Optional
    kern_path = os.path.join(gate_out, "logistic_kernel.csv")
    kern = pd.read_csv(kern_path) if os.path.exists(kern_path) else None

    fit_path = os.path.join(gate_out, "fit_logistic_results.csv")
    fit = pd.read_csv(fit_path) if os.path.exists(fit_path) else None

    # Enforce manifest + hash consistency
    dfs_for_check = {
        "logistic_band.csv": band,
        "logistic_calibration.csv": calib,
        "logistic_calibration_metrics.csv": metr,
    }
    if kern is not None:
        dfs_for_check["logistic_kernel.csv"] = kern
    if fit is not None:
        dfs_for_check["fit_logistic_results.csv"] = fit

    run_hash = _enforce_manifest_and_hashes(
        gate_out=gate_out,
        params_effective=params,
        params_path=params_path,
        dfs=dfs_for_check,
        strict_csv_hash=(not args.allow_missing_run_hash),
    )

    neff_min = float(params.get("kernel_neff_min", 25))
  
    two = ("G_central_a2" in band.columns)

    # Clean calibration rows (also ensures needed cols exist)
    calib_clean = _clean_calib(calib)

    # Colours
    col_teal1 = "#6EC5B8"; col_blue1 = "#1f77b4"
    col_teal2 = "#9ADBD2"; col_purp2 = "#7f3c8d"
    col_kern  = "#888888"
    
    # Figure layout (HORIZONTAL: 1 row × 2 cols)
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(180/25.4, 70/25.4),         # ~180mm wide × 70mm tall (tune as needed)
        gridspec_kw={"wspace": 0.32},         # spacing between left/right panels
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.18)

    # Optional panel label
    if isinstance(args.panel_label, str) and args.panel_label.strip():
        lab = args.panel_label.strip()
        if lab.startswith("(") and lab.endswith(")"):
            lab_inner = lab.strip("()")
            panel_text = rf"$({lab_inner})$"
        else:
            panel_text = lab
        fig.text(float(args.panel_x), float(args.panel_y),
                 panel_text,
                 transform=fig.transFigure, ha="left", va="top",
                 fontsize=9, color="black", clip_on=False, zorder=10)

    # --- TOP: fitted curves + kernel smoother ---
    req_a1 = {"q", "G_low_a1", "G_high_a1", "G_central_a1"}
    if not req_a1.issubset(band.columns):
        raise ValueError(f"logistic_band.csv missing required a1 columns: {sorted(req_a1.difference(band.columns))}")

    ax1.fill_between(band["q"], band["G_low_a1"], band["G_high_a1"],
                     facecolor=col_teal1, alpha=0.25, edgecolor="none")
    ax1.plot(band["q"], band["G_central_a1"], color=col_blue1, lw=1.2,
             label=r"Fitted $G(q\mid a_1)$")

    if kern is not None:
        req_k1 = {"q", "Gk_central_a1", "Gk_low_a1", "Gk_high_a1"}
        if req_k1.issubset(kern.columns):
            neff_min = float(params.get("kernel_neff_min", 25))
            if "n_eff_a1" in kern.columns:
                m = (kern["n_eff_a1"].values.astype(float) >= neff_min)
            else:
                m = np.ones(len(kern), dtype=bool)

            qk = kern["q"].values
            y  = kern["Gk_central_a1"].values.astype(float).copy()
            lo = kern["Gk_low_a1"].values.astype(float).copy()
            hi = kern["Gk_high_a1"].values.astype(float).copy()
            y[~m]  = np.nan
            lo[~m] = np.nan
            hi[~m] = np.nan

            ax1.plot(qk, y, color=col_kern, lw=1.0, alpha=0.9,
                     label=f"Kernel smoother (a1; $n_\\mathrm{{eff}}\\geq{int(neff_min)}$)")
            ax1.fill_between(qk, lo, hi, color=col_kern, alpha=0.12, edgecolor="none")

    if two:
        req_a2 = {"G_low_a2", "G_high_a2", "G_central_a2"}
        if not req_a2.issubset(band.columns):
            raise ValueError(f"logistic_band.csv indicates a2 but missing columns: {sorted(req_a2.difference(band.columns))}")

        ax1.fill_between(band["q"], band["G_low_a2"], band["G_high_a2"],
                         facecolor=col_teal2, alpha=0.25, edgecolor="none")
        ax1.plot(band["q"], band["G_central_a2"], color=col_purp2, lw=1.2,
                 label=r"Fitted $G(q\mid a_2)$")

        if kern is not None:
            req_k2 = {"q", "Gk_central_a2", "Gk_low_a2", "Gk_high_a2"}
            if req_k2.issubset(kern.columns):
                neff_min = float(params.get("kernel_neff_min", 25))
                if "n_eff_a2" in kern.columns:
                    m = (kern["n_eff_a2"].values.astype(float) >= neff_min)
                else:
                    m = np.ones(len(kern), dtype=bool)

                qk = kern["q"].values
                y  = kern["Gk_central_a2"].values.astype(float).copy()
                lo = kern["Gk_low_a2"].values.astype(float).copy()
                hi = kern["Gk_high_a2"].values.astype(float).copy()
                y[~m]  = np.nan
                lo[~m] = np.nan
                hi[~m] = np.nan

                ax1.plot(qk, y, color=col_kern, lw=1.0, alpha=0.9,
                         linestyle="--",
                         label=f"Kernel smoother (a2; $n_\\mathrm{{eff}}\\geq{int(neff_min)}$)")
                ax1.fill_between(qk, lo, hi, color=col_kern, alpha=0.12, edgecolor="none")             


    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(r"$\xi$ (–) (here $\xi=q$)")
    ax1.set_ylabel(r"$G(\xi)$ (–)")
    h1, l1 = _dedupe_legend(ax1)
    ax1.legend(h1, l1, loc="lower right", frameon=True, fontsize=7)

    # --- BOTTOM: calibration curves ---
    for a_val, df_a in calib_clean.groupby("a", sort=False):
        yerr = np.vstack([df_a["err_lo"].values, df_a["err_hi"].values])
        ax2.errorbar(
            df_a["pred_mean"].values,
            df_a["obs_rate"].values,
            yerr=yerr,
            fmt="o", ms=3.5, capsize=1.5,
            label=f"Calibration (a={a_val})"
        )

    # Metrics annotation (expects canonical labels a1/a2/all from fit_logistic.py)
    text_lines = []
    if two:
        text_lines.append(f"a1: Brier={_metric_val(metr,'Brier','a1'):.3f}, ECE={_metric_val(metr,'ECE','a1'):.3f}")
        text_lines.append(f"a2: Brier={_metric_val(metr,'Brier','a2'):.3f}, ECE={_metric_val(metr,'ECE','a2'):.3f}")
    else:
        text_lines.append(f"a1: Brier={_metric_val(metr,'Brier','a1'):.3f}, ECE={_metric_val(metr,'ECE','a1'):.3f}")

    ax2.text(
        0.02, 0.98, "\n".join(text_lines),
        transform=ax2.transAxes,
        va="top", ha="left",
        fontsize=7, linespacing=1.0,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.85, edgecolor="none"),
        zorder=4
    )

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(r"Predicted $G(q\mid a)$ (–)")
    ax2.set_ylabel(r"Observed rate (–)")
    h2, l2 = _dedupe_legend(ax2)
    leg2 = ax2.legend(
        h2, l2,
        loc="lower right",
        frameon=True, fancybox=True,
        fontsize=6,
        handlelength=1.3, labelspacing=0.25, borderpad=0.25
    )
    leg2.get_frame().set_alpha(0.95)
    leg2.get_frame().set_edgecolor("0.80")
    leg2.set_zorder(5)

    # --- Provenance footer + PDF metadata ---
    seed = params.get("seed", "NA")
    alpha = params.get("alpha", "NA")
    a1_y = params.get("a1", "NA")
    a2_y = params.get("a2", "NA")
    rh_short = run_hash[:12]

    p0_txt = ""
    if fit is not None and "p0_hat_a1" in fit.columns:
        try:
            p0a1 = float(fit["p0_hat_a1"].iloc[0])
            p0_txt += f"  p0_hat(a1)={p0a1:.3f}"
            if two and "p0_hat_a2" in fit.columns and pd.notna(fit["p0_hat_a2"].iloc[0]):
                p0a2 = float(fit["p0_hat_a2"].iloc[0])
                p0_txt += f"  p0_hat(a2)={p0a2:.3f}"
        except Exception:
            pass

    footer = f"run_hash={rh_short}  seed={seed}  alpha={alpha}  a1={a1_y} a2={a2_y}{p0_txt}"
    if not args.no_footer:
        fig.text(0.01, 0.02, footer, ha="left", va="bottom", fontsize=5, color="0.25")

    out_pdf = os.path.join(out_dir, f"{args.out_stem}.pdf")
    out_png = os.path.join(out_dir, f"{args.out_stem}.png")

    pdf_meta = {
        "Title": "CRI Logistic Diagnostics (Kernel + Calibration)",
        "Author": "GOLDILOCKS-CRI pipeline",
        "Subject": f"run_hash={run_hash}",
        "Keywords": f"run_hash={run_hash}, seed={seed}, a1={a1_y}, a2={a2_y}",
    }

    fig.savefig(out_pdf, bbox_inches="tight", metadata=pdf_meta)
    fig.savefig(out_png, dpi=int(params.get("figure_dpi", 1200)), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved SI diagnostics figure → {out_pdf} and {out_png}")
    print(f"run_hash={run_hash}")
    print(f"config={params_path}")


if __name__ == "__main__":
    main()

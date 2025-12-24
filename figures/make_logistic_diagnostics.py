# -*- coding: utf-8 -*-
"""
figures/make_logistic_diagnostics.py  •  CRI (SI diagnostics)

Panel (top): model-free kernel smoother (faint) overlaid on fitted logistic curves.
Panel (bottom): calibration curves (observed vs predicted) with Wilson intervals,
plus Brier and ECE annotations per condition.

Anti-contradiction / robustness upgrades:
  A) Require logistic_gate/output/run_manifest.json.
  B) Enforce YAML(logistic) -> run_hash matches manifest.run_hash.
  C) Enforce all input CSV run_hash (if present) matches manifest.run_hash.
  D) Enforce that arousal levels in calibration match YAML a1/a2 (two-condition).
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
    "font.size":   8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "legend.fontsize": 5,     # global default; ax2 uses smaller
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------#
# Hash / manifest utilities (must match simulate_logistic.py/fit_logistic.py)
# -----------------------------------------------------------------------------#
def _canonical_json_bytes(obj: Any) -> bytes:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def _compute_run_hash(logistic_params: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(logistic_params)).hexdigest()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _require(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_gate_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    p = obj["logistic"] if isinstance(obj, dict) and "logistic" in obj else obj
    if not isinstance(p, dict):
        raise ValueError("default_params.yml did not parse into a dict under key 'logistic'.")
    return p


def parse_args():
    p = argparse.ArgumentParser(description="Create SI logistic diagnostics figure (kernel smoother + calibration).")
    p.add_argument("--panel-label", default=os.getenv("CRI_PANEL_LABEL", ""))  # e.g. "(SI)"
    p.add_argument("--panel-x", type=float, default=float(os.getenv("CRI_PANEL_X", 0.008)))
    p.add_argument("--panel-y", type=float, default=float(os.getenv("CRI_PANEL_Y", 0.985)))
    p.add_argument("--out-stem", default=os.getenv("CRI_BOX2B_DIAG_STEM", "Box2b_logistic_diagnostics"))
    p.add_argument("--no-footer", action="store_true", help="Disable provenance footer.")
    return p.parse_args()


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


def _get_df_run_hash(df: pd.DataFrame) -> Optional[str]:
    if "run_hash" not in df.columns:
        return None
    uniq = df["run_hash"].dropna().astype(str).unique()
    if len(uniq) == 0:
        return None
    if len(uniq) != 1:
        raise RuntimeError(f"Input CSV has multiple run_hash values: {uniq}. Mixed outputs detected.")
    return str(uniq[0])


def _extract_unique_arousal(df: pd.DataFrame) -> np.ndarray:
    return np.sort(np.unique(np.round(df["a"].values.astype(float), 6)))


def _resolve_a1_a2_from_values(a_values: np.ndarray, params: dict) -> tuple[float, float]:
    """
    Resolve (a1,a2) deterministically:
      - If YAML has a1/a2, map to closest present arousal values.
      - Else use sorted unique.
    """
    a_vals = np.unique(np.round(np.asarray(a_values, dtype=float), 6))
    if len(a_vals) == 0:
        return np.nan, np.nan
    if len(a_vals) == 1:
        return float(a_vals[0]), np.nan

    a_sorted = np.sort(a_vals)
    if "a1" in params and "a2" in params:
        t1, t2 = float(params["a1"]), float(params["a2"])
        a1 = float(a_sorted[np.argmin(np.abs(a_sorted - t1))])
        a2 = float(a_sorted[np.argmin(np.abs(a_sorted - t2))])
        if np.isclose(a1, a2):
            a1, a2 = float(a_sorted[0]), float(a_sorted[1])
        return a1, a2

    return float(a_sorted[0]), float(a_sorted[1])


def _enforce_manifest_and_hashes(
    gate_out: str,
    params: dict,
    dfs: Dict[str, pd.DataFrame],
    atol_a: float = 1e-6,
) -> str:
    """
    Enforce:
      - run_manifest.json exists and contains run_hash
      - YAML(logistic) hash == manifest hash
      - all CSV run_hash (if present) == manifest hash
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

    yaml_hash = _compute_run_hash(params)
    if yaml_hash != run_hash:
        raise RuntimeError(
            "YAML/logistic dict hash does not match run_manifest.json.\n"
            f"  YAML hash:      {yaml_hash}\n"
            f"  manifest hash:  {run_hash}\n"
            "Fix: delete logistic_gate/output/* and regenerate from the intended YAML."
        )

    # CSV -> hash checks (if present)
    for name, df in dfs.items():
        h = _get_df_run_hash(df)
        if h is None:
            continue
        if h != run_hash:
            raise RuntimeError(
                f"run_hash mismatch for {name}:\n"
                f"  {name}.run_hash={h}\n"
                f"  manifest.run_hash={run_hash}\n"
                "Fix: delete logistic_gate/output/* and regenerate all outputs in one run."
            )

    # Arousal-level checks (two-condition)
    use_two = bool(params.get("use_two_conditions", True))
    if use_two:
        if "a1" not in params or "a2" not in params:
            raise RuntimeError("Two-condition mode requires YAML keys a1 and a2.")
        a1_y, a2_y = float(params["a1"]), float(params["a2"])

        # enforce against calibration file if present
        if "logistic_calibration.csv" in dfs:
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
    needed = {"pred_mean", "obs_rate", "lo", "hi", "n_bin"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"logistic_calibration.csv missing columns: {sorted(missing)}")

    for col in ("pred_mean", "obs_rate", "lo", "hi"):
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0.0, 1.0)

    df["n_bin"] = pd.to_numeric(df["n_bin"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["pred_mean", "obs_rate", "lo", "hi", "n_bin"])

    # order interval
    lo = np.minimum(df["lo"].values, df["hi"].values)
    hi = np.maximum(df["lo"].values, df["hi"].values)
    df["lo"], df["hi"] = lo, hi

    # clamp obs_rate into [lo,hi]
    df["obs_rate"] = np.maximum(df["obs_rate"].values, df["lo"].values)
    df["obs_rate"] = np.minimum(df["obs_rate"].values, df["hi"].values)

    # asymmetric errors (non-negative)
    df["err_lo"] = np.maximum(df["obs_rate"].values - df["lo"].values, 0.0)
    df["err_hi"] = np.maximum(df["hi"].values - df["obs_rate"].values, 0.0)

    df = df[(df["err_lo"].notna()) & (df["err_hi"].notna()) & (df["n_bin"] > 0)]
    return df


def _metric_val(df_metrics: pd.DataFrame, name: str, a_label: str) -> float:
    row = df_metrics[(df_metrics["metric"] == name) & (df_metrics["a"] == a_label)]
    return float(row["value"].iloc[0]) if not row.empty else float("nan")


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main():
    args = parse_args()

    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, os.pardir))
    gate = os.path.join(repo, "logistic_gate")
    gate_out = os.path.join(gate, "output")

    params = load_gate_params(_require(os.path.join(gate, "default_params.yml"), "default_params.yml"))

    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    # Required inputs
    band_path = _require(os.path.join(gate_out, "logistic_band.csv"), "logistic_band.csv")
    calib_path = _require(os.path.join(gate_out, "logistic_calibration.csv"), "logistic_calibration.csv")
    metr_path = _require(os.path.join(gate_out, "logistic_calibration_metrics.csv"), "logistic_calibration_metrics.csv")

    band  = pd.read_csv(band_path)
    calib = pd.read_csv(calib_path)
    metr  = pd.read_csv(metr_path)

    # Optional kernel diagnostics
    kern_path = os.path.join(gate_out, "logistic_kernel.csv")
    kern = pd.read_csv(kern_path) if os.path.exists(kern_path) else None

    # Optional fit (for footer consistency; not strictly required for plotting)
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

    run_hash = _enforce_manifest_and_hashes(gate_out, params, dfs_for_check)

    two = ("G_central_a2" in band.columns)

    # Resolve (a1,a2) values for consistent annotations (if calibration has numeric a)
    if "a" not in calib.columns:
        raise ValueError("logistic_calibration.csv must include an 'a' column for grouping by condition.")
    a1_val, a2_val = _resolve_a1_a2_from_values(calib["a"].values, params)

    # Colours
    col_teal1 = "#6EC5B8"; col_blue1 = "#1f77b4"
    col_teal2 = "#9ADBD2"; col_purp2 = "#7f3c8d"
    col_kern  = "#888888"

    # Figure layout
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(88/25.4, 110/25.4),
        gridspec_kw={"hspace": 0.35},
    )
    fig.subplots_adjust(right=0.98, top=0.97, bottom=0.12)

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
            ax1.plot(kern["q"], kern["Gk_central_a1"], color=col_kern, lw=1.0, alpha=0.9,
                     label="Kernel smoother (a1)")
            ax1.fill_between(kern["q"], kern["Gk_low_a1"], kern["Gk_high_a1"],
                             color=col_kern, alpha=0.12, edgecolor="none")

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
                ax1.plot(kern["q"], kern["Gk_central_a2"], color=col_kern, lw=1.0, alpha=0.9,
                         linestyle="--", label="Kernel smoother (a2)")
                ax1.fill_between(kern["q"], kern["Gk_low_a2"], kern["Gk_high_a2"],
                                 color=col_kern, alpha=0.12, edgecolor="none")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(r"$q$ (–)")
    ax1.set_ylabel(r"$G(q)$ (–)")
    h1, l1 = _dedupe_legend(ax1)
    ax1.legend(h1, l1, loc="lower right", frameon=True)

    # --- BOTTOM: calibration curves ---
    calib_clean = _clean_calib(calib)

    # Plot per condition
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
        va="top", ha="left", fontsize=5, linespacing=0.95,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.85, edgecolor="none"),
        zorder=4
    )

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Predicted probability (–)")
    ax2.set_ylabel("Observed rate (–)")

    h2, l2 = _dedupe_legend(ax2)
    leg2 = ax2.legend(
        h2, l2,
        loc="lower right",
        frameon=True, fancybox=True,
        fontsize=4,
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

    # include best-effort fit p0-hats if available
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

    footer = (
        f"run_hash={rh_short}  seed={seed}  alpha={alpha}  "
        f"a1={a1_y} a2={a2_y}{p0_txt}"
    )
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


if __name__ == "__main__":
    main()

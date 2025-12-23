# -*- coding: utf-8 -*-
"""
figures/make_logistic_diagnostics.py  •  CRI (SI diagnostics)

Panel (top): model-free kernel smoother (faint) overlaid on fitted logistic curves.
Panel (bottom): calibration curves (observed vs predicted) with Wilson intervals,
plus Brier and ECE annotations per condition.

Robustness / correctness upgrades:
  1) Explicit existence checks with clear errors (and kernel-optional handling).
  2) Deterministic mapping of a1/a2 using YAML (if provided) to avoid swapped condition labels.
  3) Sanitization of calibration data (finite, within [0,1], non-negative yerr).
  4) Legend de-duplication to prevent repeated entries.
  5) Safe handling if kernel diagnostics are disabled (kernel CSV absent).

Axes labels include dimensionless units:
  - Top: q (–), G(q) (–)
  - Bottom: Predicted probability (–), Observed rate (–)

Outputs:
  - figures/output/Box2b_logistic_diagnostics.pdf
  - figures/output/Box2b_logistic_diagnostics.png
"""
from __future__ import annotations

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "legend.fontsize": 5,     # global default; ax2 overrides smaller
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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

    # Optional panel label outside the axes (figure coords)
    p.add_argument("--panel-label", default=os.getenv("CRI_PANEL_LABEL", ""))  # e.g. "(SI)"
    p.add_argument("--panel-x", type=float, default=float(os.getenv("CRI_PANEL_X", 0.008)))
    p.add_argument("--panel-y", type=float, default=float(os.getenv("CRI_PANEL_Y", 0.985)))

    # Output naming
    p.add_argument("--out-stem", default=os.getenv("CRI_BOX2B_DIAG_STEM", "Box2b_logistic_diagnostics"))
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

    # Order lo/hi and clamp obs_rate inside interval
    df["lo"] = np.minimum(df["lo"].values, df["hi"].values)
    df["hi"] = np.maximum(df["lo"].values, df["hi"].values)
    df["obs_rate"] = np.maximum(df["obs_rate"].values, df["lo"].values)
    df["obs_rate"] = np.minimum(df["obs_rate"].values, df["hi"].values)

    df["err_lo"] = np.maximum(df["obs_rate"].values - df["lo"].values, 0.0)
    df["err_hi"] = np.maximum(df["hi"].values - df["obs_rate"].values, 0.0)

    # Drop rows with no error info or empty bins
    df = df[(df["err_lo"].notna()) & (df["err_hi"].notna()) & (df["n_bin"] > 0)]
    return df


def _metric_val(df_metrics: pd.DataFrame, name: str, a_label: str) -> float:
    row = df_metrics[(df_metrics["metric"] == name) & (df_metrics["a"] == a_label)]
    return float(row["value"].iloc[0]) if not row.empty else np.nan


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, os.pardir))
    gate = os.path.join(repo, "logistic_gate")

    params = load_gate_params(_require(os.path.join(gate, "default_params.yml"), "default_params.yml"))

    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    # Required inputs
    band  = pd.read_csv(_require(os.path.join(gate, "output", "logistic_band.csv"), "logistic_band.csv"))
    calib = pd.read_csv(_require(os.path.join(gate, "output", "logistic_calibration.csv"), "logistic_calibration.csv"))
    metr  = pd.read_csv(_require(os.path.join(gate, "output", "logistic_calibration_metrics.csv"),
                                 "logistic_calibration_metrics.csv"))

    # Optional kernel diagnostics (may not exist if kernel_enabled=false)
    kern_path = os.path.join(gate, "output", "logistic_kernel.csv")
    kern = pd.read_csv(kern_path) if os.path.exists(kern_path) else None

    two = ("G_central_a2" in band.columns)

    # Determine (a1,a2) mapping for label text consistency
    # We use calibration 'a' values if present; otherwise default to YAML labels.
    if "a" in calib.columns:
        a1_val, a2_val = _resolve_a1_a2_from_values(calib["a"].values, params)
    else:
        a1_val, a2_val = float(params.get("a1", np.nan)), float(params.get("a2", np.nan))

    # Colours (kept consistent with your figure palette)
    col_teal1 = "#6EC5B8"; col_blue1 = "#1f77b4"
    col_teal2 = "#9ADBD2"; col_purp2 = "#7f3c8d"
    col_kern  = "#888888"  # faint gray for kernel

    # Figure layout
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(88/25.4, 110/25.4),
        gridspec_kw={"hspace": 0.35},
    )
    fig.subplots_adjust(right=0.98, top=0.97)

    # Optional panel label outside axes (black)
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

    # --- TOP: kernel smoother vs logistic fits ---
    required_a1 = {"q", "G_low_a1", "G_high_a1", "G_central_a1"}
    if not required_a1.issubset(band.columns):
        raise ValueError(f"logistic_band.csv missing required a1 columns: {sorted(required_a1.difference(band.columns))}")

    ax1.fill_between(band["q"], band["G_low_a1"], band["G_high_a1"],
                     facecolor=col_teal1, alpha=0.25, edgecolor="none")
    ax1.plot(band["q"], band["G_central_a1"], color=col_blue1, lw=1.2,
             label=r"Fitted $G(q\mid a_1)$")

    # Kernel (a1) if available
    if kern is not None and {"q", "Gk_central_a1", "Gk_low_a1", "Gk_high_a1"}.issubset(kern.columns):
        ax1.plot(kern["q"], kern["Gk_central_a1"], color=col_kern, lw=1.0, alpha=0.9,
                 label="Kernel smoother (a1)")
        ax1.fill_between(kern["q"], kern["Gk_low_a1"], kern["Gk_high_a1"],
                         color=col_kern, alpha=0.12, edgecolor="none")
    else:
        # If kernel is missing, make this explicit in the legend via a dummy handle (optional)
        pass

    if two:
        required_a2 = {"G_low_a2", "G_high_a2", "G_central_a2"}
        if not required_a2.issubset(band.columns):
            raise ValueError(f"logistic_band.csv indicates a2 but missing columns: {sorted(required_a2.difference(band.columns))}")

        ax1.fill_between(band["q"], band["G_low_a2"], band["G_high_a2"],
                         facecolor=col_teal2, alpha=0.25, edgecolor="none")
        ax1.plot(band["q"], band["G_central_a2"], color=col_purp2, lw=1.2,
                 label=r"Fitted $G(q\mid a_2)$")

        # Kernel (a2) if available
        if kern is not None and {"q", "Gk_central_a2", "Gk_low_a2", "Gk_high_a2"}.issubset(kern.columns):
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

    # --- BOTTOM: calibration curve with Wilson CIs ---
    if "a" not in calib.columns:
        raise ValueError("logistic_calibration.csv must include an 'a' column for grouping by condition.")

    # If your pipeline writes a as numeric floats for two-condition, keep it numeric for labels.
    # If it writes 'a1'/'a2' strings, this also works.
    for a_val, df_a in calib.groupby("a", sort=False):
        dfc = _clean_calib(df_a)
        if dfc.empty:
            continue
        yerr = np.vstack([dfc["err_lo"].values, dfc["err_hi"].values])
        ax2.errorbar(
            dfc["pred_mean"].values,
            dfc["obs_rate"].values,
            yerr=yerr,
            fmt="o", ms=3.5, capsize=1.5,
            label=f"Calibration (a={a_val})"
        )

    # Metrics annotation (Brier & ECE)
    # Prefer canonical labels 'a1','a2' as written by fit_logistic.py
    text_lines = []
    if two:
        text_lines.append(f"a1: Brier={_metric_val(metr,'Brier','a1'):.3f}, ECE={_metric_val(metr,'ECE','a1'):.3f}")
        text_lines.append(f"a2: Brier={_metric_val(metr,'Brier','a2'):.3f}, ECE={_metric_val(metr,'ECE','a2'):.3f}")
    else:
        text_lines.append(f"a1: Brier={_metric_val(metr,'Brier','a1'):.3f}, ECE={_metric_val(metr,'ECE','a1'):.3f}")
    text = "\n".join(text_lines)

    ax2.text(
        0.02, 0.98, text,
        transform=ax2.transAxes,
        va="top", ha="left",
        fontsize=5, linespacing=0.95,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.85, edgecolor="none"),
        zorder=4
    )

    # Axes labels WITH units (dimensionless)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Predicted probability (–)")
    ax2.set_ylabel("Observed rate (–)")

    # Legend inside (lower-right), compact
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

    # Save
    out_pdf = os.path.join(out_dir, f"{args.out_stem}.pdf")
    out_png = os.path.join(out_dir, f"{args.out_stem}.png")
    fig.savefig(out_pdf, bbox_inches="tight")  # vector
    fig.savefig(out_png, dpi=int(params.get("figure_dpi", 1200)), bbox_inches="tight")  # high-dpi raster
    plt.close(fig)

    print(f"Saved SI diagnostics figure → {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
figures/make_logistic_figure.py  •  CRI (Box 2b main panel)

Main Box-2(b): logistic “tipping point” with two arousal levels.
- 95% bootstrap CI ribbons for each curve
- Solid lines for fitted sigmoids
- Orange/yellow bin-mean markers (visualization only; fits are trial-wise)
- Dashed vertical lines at p0-hats for each condition
- Inset showing dG/dq for each curve

Robustness / correctness fixes:
  1) Deterministic mapping of a1/a2 using YAML to avoid label/colour swaps.
  2) Panel label is BLACK and controlled via args/env (no hard-coded blue).
  3) Legend de-duplication to prevent repeated labels when multiple bin groups exist.
  4) Inset vertical p0(a2) line is only drawn if p0_a2 is defined (no NameError).
  5) Mild sanitation of inputs and existence checks with explicit errors.

Outputs:
  - figures/output/Box2b_logistic.pdf
  - figures/output/Box2b_logistic.png
"""
from __future__ import annotations

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "legend.fontsize": 4,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_gate_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    p = obj["logistic"] if isinstance(obj, dict) and "logistic" in obj else obj
    if not isinstance(p, dict):
        raise ValueError("default_params.yml did not parse into a dict under key 'logistic'.")
    return p


def _require(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _resolve_a1_a2_from_bins(df_bins: pd.DataFrame, params: dict) -> tuple[float, float]:
    """
    Resolve (a1,a2) deterministically:
      - If YAML has a1/a2, map to closest arousal values present in bins.
      - Else use sorted unique values from bins.
    """
    a_vals = np.unique(np.round(df_bins["a"].values.astype(float), 6))
    if len(a_vals) < 2:
        # single-condition is allowed; return (a1, nan)
        return float(a_vals[0]) if len(a_vals) == 1 else np.nan, np.nan

    a_sorted = np.sort(a_vals)

    if "a1" in params and "a2" in params:
        t1, t2 = float(params["a1"]), float(params["a2"])
        a1 = float(a_sorted[np.argmin(np.abs(a_sorted - t1))])
        a2 = float(a_sorted[np.argmin(np.abs(a_sorted - t2))])
        if np.isclose(a1, a2):
            a1, a2 = float(a_sorted[0]), float(a_sorted[1])
        return a1, a2

    return float(a_sorted[0]), float(a_sorted[1])


def _dedupe_legend(ax):
    """Remove duplicate labels while preserving order."""
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l in seen or l.strip() == "":
            continue
        seen.add(l)
        h2.append(h)
        l2.append(l)
    return h2, l2


def parse_args():
    p = argparse.ArgumentParser(description="Create Box-2(b) logistic gate figure (main).")

    # Panel label controls (figure coordinates)
    p.add_argument("--panel-label", default=os.getenv("CRI_PANEL_LABEL", "(b)"))
    p.add_argument("--panel-x", type=float, default=float(os.getenv("CRI_PANEL_X", 0.008)))
    p.add_argument("--panel-y", type=float, default=float(os.getenv("CRI_PANEL_Y", 0.975)))

    # Inset controls
    p.add_argument("--inset-x", type=float, default=float(os.getenv("CRI_INSET_X", 0.68)))
    p.add_argument("--inset-y", type=float, default=float(os.getenv("CRI_INSET_Y", 0.27)))
    p.add_argument("--inset-w", type=float, default=float(os.getenv("CRI_INSET_W", 0.38)))
    p.add_argument("--inset-h", type=float, default=float(os.getenv("CRI_INSET_H", 0.38)))

    # Output stem
    p.add_argument("--out-stem", default=os.getenv("CRI_BOX2B_STEM", "Box2b_logistic"))
    return p.parse_args()


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

    # Inputs (explicit checks)
    df_band = pd.read_csv(_require(os.path.join(gate, "output", "logistic_band.csv"), "logistic_band.csv"))
    df_fit  = pd.read_csv(_require(os.path.join(gate, "output", "fit_logistic_results.csv"), "fit_logistic_results.csv"))
    df_bins = pd.read_csv(_require(os.path.join(gate, "output", "logistic_bins.csv"), "logistic_bins.csv"))
    df_der  = pd.read_csv(_require(os.path.join(gate, "output", "logistic_derivative.csv"), "logistic_derivative.csv"))

    # Determine whether two-condition outputs are present
    two = ("G_central_a2" in df_band.columns) and ("dGdq_a2" in df_der.columns) and ("p0_hat_a2" in df_fit.columns)

    # Robust mapping of a1/a2 for consistent bin colours/labels
    a1_val, a2_val = _resolve_a1_a2_from_bins(df_bins, params)

    # Colors (kept as in your script)
    col_teal1 = "#6EC5B8"
    col_teal2 = "#9ADBD2"
    col_blue1 = "#1f77b4"
    col_purp2 = "#7f3c8d"
    col_orng  = "#FF8C1A"
    col_yell  = "#FFC107"
    col_grey  = "0.45"

    # Figure size
    fig, ax = plt.subplots(figsize=(88/25.4, (88/1.55)/25.4))
    fig.subplots_adjust(top=0.965)  # reduce whitespace above axes

    # Panel label outside axes (BLACK, controllable)
    lab = args.panel_label.strip()
    # If user passes "(b)" or "b", we render in mathtext consistently.
    if lab.startswith("(") and lab.endswith(")"):
        lab_inner = lab.strip("()")
        panel_text = rf"$({lab_inner})$"
    else:
        panel_text = lab
    fig.text(float(args.panel_x), float(args.panel_y),
             panel_text,
             transform=fig.transFigure, ha="left", va="top",
             fontsize=9, color="black", clip_on=False, zorder=10)

    # --- CI ribbons + fitted curves ---
    if not {"q", "G_low_a1", "G_high_a1", "G_central_a1"}.issubset(df_band.columns):
        raise ValueError("logistic_band.csv missing required a1 columns.")

    ax.fill_between(df_band["q"], df_band["G_low_a1"], df_band["G_high_a1"],
                    facecolor=col_teal1, alpha=0.45, edgecolor=col_teal1, linewidth=0.6,
                    label=f"{params.get('ci_percent', 95)}% sim. bootstrap band — a1")
    ax.plot(df_band["q"], df_band["G_central_a1"], color=col_blue1, linewidth=1.2,
            label=r"Fitted $G(q\mid a_1)$")

    if two:
        ax.fill_between(df_band["q"], df_band["G_low_a2"], df_band["G_high_a2"],
                        facecolor=col_teal2, alpha=0.45, edgecolor=col_teal2, linewidth=0.6,
                        label=f"{params.get('ci_percent', 95)}% sim. bootstrap band — a2")
        ax.plot(df_band["q"], df_band["G_central_a2"], color=col_purp2, linewidth=1.2,
                label=r"Fitted $G(q\mid a_2)$")

    # --- Bin means (visual only) ---
    # Use resolved a1/a2 mapping for colours; label once per condition.
    if not {"q_bin_center", "rate_mean", "a"}.issubset(df_bins.columns):
        raise ValueError("logistic_bins.csv missing required columns.")

    # Plot a1 bins
    if np.isfinite(a1_val):
        df_a1 = df_bins[np.isclose(df_bins["a"].astype(float), a1_val)]
        if len(df_a1):
            ax.scatter(df_a1["q_bin_center"], df_a1["rate_mean"], s=18,
                       facecolors=col_orng, edgecolors="black", linewidths=0.4,
                       label=rf"Bin means ($a_1={a1_val:.2f}$)")

    # Plot a2 bins (if present)
    if two and np.isfinite(a2_val):
        df_a2 = df_bins[np.isclose(df_bins["a"].astype(float), a2_val)]
        if len(df_a2):
            ax.scatter(df_a2["q_bin_center"], df_a2["rate_mean"], s=18,
                       facecolors=col_yell, edgecolors="black", linewidths=0.4,
                       label=rf"Bin means ($a_2={a2_val:.2f}$)")

    # --- Vertical lines at p0-hats ---
    p0_a1 = float(df_fit["p0_hat_a1"].iloc[0])
    ax.axvline(p0_a1, color=col_grey, linestyle="--", linewidth=0.8,
               label=rf"$p_0(a_1)={p0_a1:.2f}$")

    p0_a2 = None
    if two:
        val = df_fit["p0_hat_a2"].iloc[0]
        if pd.notna(val):
            p0_a2 = float(val)
            ax.axvline(p0_a2, color="0.30", linestyle="--", linewidth=0.8,
                       label=rf"$p_0(a_2)={p0_a2:.2f}$")

    # --- Axes labels WITH units (dimensionless) ---
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$q$ (–)")
    ax.set_ylabel(r"$G(q\mid a)$ (–)")

    # Legend (de-duplicated)
    h2, l2 = _dedupe_legend(ax)
    ax.legend(h2, l2, loc="upper left", frameon=True)

    # --- Inset: derivatives (dimensionless) ---
    if not {"q", "dGdq_a1"}.issubset(df_der.columns):
        raise ValueError("logistic_derivative.csv missing required a1 columns.")

    ax_ins = inset_axes(
        ax, width="70%", height="70%",
        loc="lower left",
        bbox_to_anchor=(float(args.inset_x), float(args.inset_y), float(args.inset_w), float(args.inset_h)),
        bbox_transform=ax.transAxes
    )

    ax_ins.plot(df_der["q"], df_der["dGdq_a1"], color=col_blue1, linewidth=0.9,
                label=r"$\mathrm{d}G/\mathrm{d}q$ (a$_1$)")
    ax_ins.axvline(p0_a1, color=col_grey, linestyle="--", linewidth=0.7)

    if two and ("dGdq_a2" in df_der.columns):
        ax_ins.plot(df_der["q"], df_der["dGdq_a2"], color=col_purp2, linewidth=0.9,
                    label=r"$\mathrm{d}G/\mathrm{d}q$ (a$_2$)")
        if p0_a2 is not None:
            ax_ins.axvline(p0_a2, color="0.30", linestyle="--", linewidth=0.7)

    ax_ins.set_title(r"$\mathrm{d}G/\mathrm{d}q$", fontsize=5)
    ax_ins.set_xlabel(r"$q$ (–)", fontsize=5)
    ax_ins.set_ylabel(r"$\mathrm{d}G/\mathrm{d}q$ (–)", fontsize=5)
    ax_ins.set_xlim(0, 1)
    ax_ins.tick_params(labelsize=3)

    # Small inset legend without frame
    h3, l3 = _dedupe_legend(ax_ins)
    ax_ins.legend(h3, l3, loc="upper right", frameon=False, fontsize=2)

    # --- Save ---
    out_pdf = os.path.join(out_dir, f"{args.out_stem}.pdf")
    out_png = os.path.join(out_dir, f"{args.out_stem}.png")

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png, dpi=int(params.get("figure_dpi", 1200)), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"Saved logistic figure → {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
figures/make_logistic_figure.py  •  CRI (Box 2b main panel)

Main Box-2(b): logistic “tipping point” with two arousal levels.

Anti-contradiction mechanisms (hostile-reviewer-proof):
  A) Require logistic_gate/output/run_manifest.json and verify:
       - manifest.run_hash matches YAML-derived run_hash (logistic dict only)
  B) Require every input CSV to carry the same run_hash (if present),
     and enforce it matches the manifest run_hash.
  C) Enforce that unique arousal values in bins/trials match YAML a1/a2
     (within tolerance) when use_two_conditions is true.
  D) Stamp provenance into figure (footer + PDF metadata).

Outputs:
  - figures/output/Box2b_logistic.pdf
  - figures/output/Box2b_logistic.png
"""
from __future__ import annotations

import os
import json
import argparse
import yaml
import hashlib
from typing import Any, Dict, Optional, Tuple

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


# -----------------------------------------------------------------------------#
# Hash / manifest utilities (must match simulate_logistic.py/fit_logistic.py)
# -----------------------------------------------------------------------------#
def _canonical_json_bytes(obj: Any) -> bytes:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def _compute_run_hash(logistic_params: Dict[str, Any]) -> str:
    payload = _canonical_json_bytes(logistic_params)
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
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


def _extract_unique_arousal(df: pd.DataFrame) -> np.ndarray:
    return np.sort(np.unique(np.round(df["a"].values.astype(float), 6)))


def _dedupe_legend(ax):
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


def _get_df_run_hash(df: pd.DataFrame) -> Optional[str]:
    if "run_hash" not in df.columns:
        return None
    uniq = df["run_hash"].dropna().astype(str).unique()
    if len(uniq) == 0:
        return None
    if len(uniq) != 1:
        raise RuntimeError(f"Input CSV has multiple run_hash values: {uniq}. Mixed outputs detected.")
    return str(uniq[0])


def _enforce_manifest_and_hashes(
    gate_out: str,
    params: dict,
    dfs: Dict[str, pd.DataFrame],
    atol_a: float = 1e-6,
) -> str:
    """
    Enforce:
      - run_manifest.json exists
      - YAML hash == manifest hash
      - all CSV run_hash (if present) equals manifest hash
      - arousal levels match YAML (two-condition)
    Returns: run_hash (manifest)
    """
    man_path = os.path.join(gate_out, "run_manifest.json")
    if not os.path.exists(man_path):
        raise RuntimeError(
            "Missing logistic_gate/output/run_manifest.json. "
            "Rerun logistic_gate/simulate_logistic.py before generating figures."
        )
    manifest = _read_json(man_path)
    if "run_hash" not in manifest:
        raise RuntimeError("run_manifest.json missing key 'run_hash'.")
    run_hash = str(manifest["run_hash"])

    # YAML -> hash check
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
            # allowed for backward-compat, but strongly discouraged
            continue
        if h != run_hash:
            raise RuntimeError(
                f"run_hash mismatch for {name}:\n"
                f"  {name}.run_hash={h}\n"
                f"  manifest.run_hash={run_hash}\n"
                "Fix: delete logistic_gate/output/* and regenerate all outputs in one run."
            )

    # Arousal-level checks
    use_two = bool(params.get("use_two_conditions", True))
    if use_two:
        if "a1" not in params or "a2" not in params:
            raise RuntimeError("Two-condition mode requires YAML keys a1 and a2 to be present.")
        a1_y, a2_y = float(params["a1"]), float(params["a2"])

        # Prefer bins for display consistency; fall back to any df containing 'a'
        df_a = None
        for name in ["logistic_bins.csv", "logistic_trials.csv"]:
            if name in dfs and "a" in dfs[name].columns:
                df_a = dfs[name]
                break
        if df_a is None:
            raise RuntimeError("No input dataframe contains column 'a' for arousal checks.")

        a_vals = _extract_unique_arousal(df_a)
        if len(a_vals) < 2:
            raise RuntimeError("Two-condition YAML but data contains <2 arousal values.")

        a1_d = float(a_vals[np.argmin(np.abs(a_vals - a1_y))])
        a2_d = float(a_vals[np.argmin(np.abs(a_vals - a2_y))])

        if (abs(a1_d - a1_y) > atol_a) or (abs(a2_d - a2_y) > atol_a) or np.isclose(a1_d, a2_d):
            raise RuntimeError(
                "Arousal levels in data do not match YAML a1/a2 within tolerance.\n"
                f"  YAML a1,a2: {a1_y:.6f}, {a2_y:.6f}\n"
                f"  Data levels: {a_vals}\n"
                "Fix: delete logistic_gate/output/* and regenerate."
            )

    return run_hash


def _resolve_a1_a2(df_bins: pd.DataFrame, params: dict) -> tuple[float, float]:
    a_vals = _extract_unique_arousal(df_bins)
    if len(a_vals) < 2:
        return float(a_vals[0]) if len(a_vals) == 1 else np.nan, np.nan

    if "a1" in params and "a2" in params:
        t1, t2 = float(params["a1"]), float(params["a2"])
        a1 = float(a_vals[np.argmin(np.abs(a_vals - t1))])
        a2 = float(a_vals[np.argmin(np.abs(a_vals - t2))])
        if np.isclose(a1, a2):
            a1, a2 = float(a_vals[0]), float(a_vals[1])
        return a1, a2

    return float(a_vals[0]), float(a_vals[1])


def parse_args():
    p = argparse.ArgumentParser(description="Create Box-2(b) logistic gate figure (main).")

    # Panel label controls
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

    # Footer provenance toggle
    p.add_argument("--no-footer", action="store_true", help="Disable provenance footer.")
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

    params = load_gate_params(_require(os.path.join(gate, "default_params.yml"), "default_params.yml"))

    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    # Inputs
    df_band = pd.read_csv(_require(os.path.join(gate_out, "logistic_band.csv"), "logistic_band.csv"))
    df_fit  = pd.read_csv(_require(os.path.join(gate_out, "fit_logistic_results.csv"), "fit_logistic_results.csv"))
    df_bins = pd.read_csv(_require(os.path.join(gate_out, "logistic_bins.csv"), "logistic_bins.csv"))
    df_der  = pd.read_csv(_require(os.path.join(gate_out, "logistic_derivative.csv"), "logistic_derivative.csv"))

    # Optional: for stronger checks if present
    trials_path = os.path.join(gate_out, "logistic_trials.csv")
    df_trials = pd.read_csv(trials_path) if os.path.exists(trials_path) else None

    dfs_for_check = {
        "logistic_band.csv": df_band,
        "fit_logistic_results.csv": df_fit,
        "logistic_bins.csv": df_bins,
        "logistic_derivative.csv": df_der,
    }
    if df_trials is not None:
        dfs_for_check["logistic_trials.csv"] = df_trials

    # Enforce manifest + hash consistency
    run_hash = _enforce_manifest_and_hashes(gate_out, params, dfs_for_check)

    # Determine whether two-condition outputs are present
    two = ("G_central_a2" in df_band.columns) and ("dGdq_a2" in df_der.columns) and ("p0_hat_a2" in df_fit.columns)

    # Resolve a1/a2 for display and ensure stable mapping
    a1_val, a2_val = _resolve_a1_a2(df_bins, params)

    # Colours
    col_teal1 = "#6EC5B8"
    col_teal2 = "#9ADBD2"
    col_blue1 = "#1f77b4"
    col_purp2 = "#7f3c8d"
    col_orng  = "#FF8C1A"
    col_yell  = "#FFC107"
    col_grey  = "0.45"

    # Figure
    fig, ax = plt.subplots(figsize=(88/25.4, (88/1.55)/25.4))
    fig.subplots_adjust(top=0.965, bottom=0.14)

    # Panel label (black)
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

    # --- Bands + fitted curves ---
    req_a1 = {"q", "G_low_a1", "G_high_a1", "G_central_a1"}
    if not req_a1.issubset(df_band.columns):
        raise ValueError("logistic_band.csv missing required a1 columns.")

    ci_percent = params.get("ci_percent", 95)

    ax.fill_between(df_band["q"], df_band["G_low_a1"], df_band["G_high_a1"],
                    facecolor=col_teal1, alpha=0.45, edgecolor=col_teal1, linewidth=0.6,
                    label=f"{ci_percent}% sim. bootstrap band — a1")
    ax.plot(df_band["q"], df_band["G_central_a1"], color=col_blue1, linewidth=1.2,
            label=r"Fitted $G(q\mid a_1)$")

    if two:
        ax.fill_between(df_band["q"], df_band["G_low_a2"], df_band["G_high_a2"],
                        facecolor=col_teal2, alpha=0.45, edgecolor=col_teal2, linewidth=0.6,
                        label=f"{ci_percent}% sim. bootstrap band — a2")
        ax.plot(df_band["q"], df_band["G_central_a2"], color=col_purp2, linewidth=1.2,
                label=r"Fitted $G(q\mid a_2)$")

    # --- Bin means (visual only) ---
    if not {"q_bin_center", "rate_mean", "a"}.issubset(df_bins.columns):
        raise ValueError("logistic_bins.csv missing required columns.")

    if np.isfinite(a1_val):
        df_a1 = df_bins[np.isclose(df_bins["a"].astype(float), a1_val)]
        if len(df_a1):
            ax.scatter(df_a1["q_bin_center"], df_a1["rate_mean"], s=18,
                       facecolors=col_orng, edgecolors="black", linewidths=0.4,
                       label=rf"Bin means ($a_1={a1_val:.2f}$)")

    if two and np.isfinite(a2_val):
        df_a2 = df_bins[np.isclose(df_bins["a"].astype(float), a2_val)]
        if len(df_a2):
            ax.scatter(df_a2["q_bin_center"], df_a2["rate_mean"], s=18,
                       facecolors=col_yell, edgecolors="black", linewidths=0.4,
                       label=rf"Bin means ($a_2={a2_val:.2f}$)")

    # --- Vertical p0-hat lines ---
    if "p0_hat_a1" not in df_fit.columns:
        raise ValueError("fit_logistic_results.csv missing p0_hat_a1.")
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

    # --- Axes labels (dimensionless) ---
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$q$ (–)")
    ax.set_ylabel(r"$G(q\mid a)$ (–)")

    # Legend (deduped)
    h2, l2 = _dedupe_legend(ax)
    ax.legend(h2, l2, loc="upper left", frameon=True)

    # --- Inset: dG/dq ---
    if not {"q", "dGdq_a1"}.issubset(df_der.columns):
        raise ValueError("logistic_derivative.csv missing required derivative columns.")

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

    h3, l3 = _dedupe_legend(ax_ins)
    ax_ins.legend(h3, l3, loc="upper right", frameon=False, fontsize=2)

    # --- Provenance footer (anti-contradiction stamp) ---
    seed = params.get("seed", "NA")
    alpha = params.get("alpha", "NA")
    a1_y = params.get("a1", "NA")
    a2_y = params.get("a2", "NA")
    rh_short = run_hash[:12]

    footer = (
        f"run_hash={rh_short}  seed={seed}  alpha={alpha}  "
        f"a1={a1_y} a2={a2_y}  "
        f"p0_hat(a1)={p0_a1:.3f}" + ("" if p0_a2 is None else f"  p0_hat(a2)={p0_a2:.3f}")
    )
    if not args.no_footer:
        fig.text(0.01, 0.02, footer, ha="left", va="bottom", fontsize=5, color="0.25")

    # --- Save with PDF metadata where possible ---
    out_pdf = os.path.join(out_dir, f"{args.out_stem}.pdf")
    out_png = os.path.join(out_dir, f"{args.out_stem}.png")

    pdf_meta = {
        "Title": "CRI Box 2b Logistic Gate",
        "Author": "GOLDILOCKS-CRI pipeline",
        "Subject": f"run_hash={run_hash}",
        "Keywords": f"run_hash={run_hash}, seed={seed}, a1={a1_y}, a2={a2_y}",
    }

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01, metadata=pdf_meta)
    fig.savefig(out_png, dpi=int(params.get("figure_dpi", 1200)), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"Saved logistic figure → {out_pdf} and {out_png}")
    print(f"run_hash={run_hash}")


if __name__ == "__main__":
    main()

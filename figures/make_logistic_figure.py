# -*- coding: utf-8 -*-
"""
figures/make_logistic_figure.py  •  CRI (Box 2b main panel)

Main Box-2(b): logistic “tipping point” with two arousal levels.

Anti-contradiction mechanisms (hostile-reviewer-proof):
  A) Require logistic_gate/output/run_manifest.json and verify:
       - manifest.run_hash matches YAML-derived run_hash (computed with SAME defaulting as simulate_logistic.py)
       - (optional) manifest.config_sha256 matches YAML file bytes (if present)
       - (optional) manifest is internally consistent if it contains manifest.logistic_params
  B) Require every input CSV to carry the same run_hash (strict by default),
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
from typing import Any, Dict, Optional

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
# Hash / manifest utilities (must match simulate_logistic.py / fit_logistic.py)
# -----------------------------------------------------------------------------#
def _canonical_json_bytes(obj: Any) -> bytes:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def _compute_run_hash(logistic_params_effective: Dict[str, Any]) -> str:
    payload = _canonical_json_bytes(logistic_params_effective)
    return hashlib.sha256(payload).hexdigest()


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
# Config resolution (mirror simulate_logistic.py / fit_logistic.py)
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

    # Defaults used by simulate_logistic.py
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

    return p


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
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
      - run_manifest.json exists
      - YAML(effective)->run_hash == manifest.run_hash
      - (optional) manifest.config_sha256 matches YAML file bytes (if present)
      - CSV run_hash equals manifest.run_hash (strict by default)
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

    # Optional: verify YAML bytes hash if manifest provides it
    if "config_sha256" in manifest and isinstance(manifest["config_sha256"], str):
        sha_now = _sha256_of_file(params_path)
        if sha_now != str(manifest["config_sha256"]):
            raise RuntimeError(
                "YAML file bytes do not match run_manifest.json config_sha256.\n"
                f"  current YAML sha256: {sha_now}\n"
                f"  manifest sha256:     {manifest['config_sha256']}\n"
                "Fix: delete logistic_gate/output/* and regenerate from the intended YAML."
            )

    # YAML(effective) -> hash check
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

    # Arousal-level checks
    use_two = bool(params_effective.get("use_two_conditions", True))
    if use_two:
        a1_y, a2_y = float(params_effective["a1"]), float(params_effective["a2"])

        # Prefer bins; fall back to trials
        df_a = None
        for key in ("logistic_bins.csv", "logistic_trials.csv"):
            if key in dfs and "a" in dfs[key].columns:
                df_a = dfs[key]
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


def _resolve_a1_a2(df_bins: pd.DataFrame, params_effective: dict) -> tuple[float, float]:
    a_vals = _extract_unique_arousal(df_bins)
    if len(a_vals) < 2:
        return float(a_vals[0]) if len(a_vals) == 1 else np.nan, np.nan

    t1, t2 = float(params_effective["a1"]), float(params_effective["a2"])
    a1 = float(a_vals[np.argmin(np.abs(a_vals - t1))])
    a2 = float(a_vals[np.argmin(np.abs(a_vals - t2))])
    if np.isclose(a1, a2):
        a1, a2 = float(a_vals[0]), float(a_vals[1])
    return a1, a2


def parse_args():
    p = argparse.ArgumentParser(description="Create Box-2(b) logistic gate figure (main).")

    # YAML config (match simulate/fit precedence)
    p.add_argument(
        "--config",
        default=os.getenv("CRI_LOGISTIC_CONFIG") or os.getenv("CRI_BOX2B_CONFIG"),
        help="Path to YAML config (default: env; else params_box2b.yml if present; else default_params.yml).",
    )

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

    # Provenance footer toggle
    p.add_argument("--no-footer", action="store_true", help="Disable provenance footer.")

    # Strictness: require run_hash column in all input CSVs
    p.add_argument("--allow-missing-run-hash", action="store_true",
                   help="Backward-compat: allow CSVs without run_hash columns (not recommended).")

    # --- NEW layout knobs (to prevent footer/xlabel collisions) ---
    p.add_argument("--fig-bottom", type=float, default=float(os.getenv("CRI_BOX2B_FIG_BOTTOM", 0.18)),
                   help="Bottom margin for fig.subplots_adjust (fraction of figure).")
    p.add_argument("--xlabel-y", type=float, default=float(os.getenv("CRI_BOX2B_XLABEL_Y", -0.06)),
                   help="Y position of xlabel in AXES coordinates (0=axis line; negative is below).")
    p.add_argument("--footer-y", type=float, default=float(os.getenv("CRI_BOX2B_FOOTER_Y", 0.004)),
                   help="Y position of provenance footer in FIGURE coordinates (0=bottom edge).")

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

    # Load YAML (same precedence as simulate/fit), then apply simulate defaults for hashing
    params_path = _resolve_params_path(args.config)
    p_raw = _load_yaml_logistic_section(params_path)
    params = _apply_simulate_defaults(p_raw)

    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    # Inputs
    df_band = pd.read_csv(_require(os.path.join(gate_out, "logistic_band.csv"), "logistic_band.csv"))
    df_fit  = pd.read_csv(_require(os.path.join(gate_out, "fit_logistic_results.csv"), "fit_logistic_results.csv"))
    df_bins = pd.read_csv(_require(os.path.join(gate_out, "logistic_bins.csv"), "logistic_bins.csv"))
    df_der  = pd.read_csv(_require(os.path.join(gate_out, "logistic_derivative.csv"), "logistic_derivative.csv"))

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
    run_hash = _enforce_manifest_and_hashes(
        gate_out=gate_out,
        params_effective=params,
        params_path=params_path,
        dfs=dfs_for_check,
        strict_csv_hash=(not args.allow_missing_run_hash),
    )

    # Determine whether two-condition outputs are present
    two = ("G_central_a2" in df_band.columns) and ("dGdq_a2" in df_der.columns) and ("p0_hat_a2" in df_fit.columns)

    # Resolve a1/a2 for display
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
    fig.subplots_adjust(top=0.965, bottom=float(args.fig_bottom))

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

    # Bands + fitted curves
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

    # Bin means (visual only)
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

    # Vertical p0-hat lines
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

    # Axes labels (dimensionless)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$q$ (–)")
    ax.set_ylabel(r"$G(q\mid a)$ (–)")

    # --- FIX: move xlabel upward (axes coords) to clear the footer ---
    try:
        ax.xaxis.set_label_coords(0.5, float(args.xlabel_y))
    except Exception:
        pass

    # Legend (deduped)
    h2, l2 = _dedupe_legend(ax)
    ax.legend(h2, l2, loc="upper left", frameon=True)

    # Inset: dG/dq
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

    # Provenance footer + PDF metadata
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
        # --- FIX: move footer downward (figure coords) into the freed space ---
        fig.text(0.01, float(args.footer_y), footer, ha="left", va="bottom", fontsize=5, color="0.25")

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
    print(f"config={params_path}")


if __name__ == "__main__":
    main()

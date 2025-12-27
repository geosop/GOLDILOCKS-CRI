# -*- coding: utf-8 -*-
"""
figures/make_decay_figure.py  •  CRI v0.3-SIM (robust; provenance-visible)

FIX (layout):
  - Provenance footer is anchored bottom-left (so it does not collide with xlabel).
  - X-axis label is right-aligned and shifted right.
  - Adds explicit --prov-x/--prov-y and --xlabel-x knobs for fine positioning.

Other provenance behavior unchanged:
  - Enforces run_hash against decay/output/run_manifest.json AND YAML-effective params.
  - Embeds run_hash + config path into PDF/PNG metadata.
"""
from __future__ import annotations

import os
import json
import yaml
import hashlib
import argparse
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
from matplotlib import transforms as mtransforms

# Optional: Pillow for PNG metadata text chunks
try:
    from PIL import PngImagePlugin  # type: ignore
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False


# -----------------------------------------------------------------------------
# Matplotlib defaults (RSOS-friendly)
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 0.9,
    "legend.fontsize": 5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DECAY_DIR = os.path.join(REPO_ROOT, "decay")
DECAY_OUT = os.path.join(DECAY_DIR, "output")
FIG_OUT = os.path.join(HERE, "output")
os.makedirs(FIG_OUT, exist_ok=True)


# -----------------------------------------------------------------------------
# Hash / manifest utilities (must match decay/simulate_decay.py)
# -----------------------------------------------------------------------------
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


def _resolve_config_path(config_path: Optional[str]) -> str:
    """
    Resolve decay YAML path deterministically.
    Relative paths resolved against repo root first, then decay/.

    Priority:
      1) explicit argument
      2) env var CRI_DECAY_CONFIG
      3) decay/default_params.yml
    """
    if config_path is None:
        config_path = os.getenv("CRI_DECAY_CONFIG")

    if config_path is None:
        return os.path.join(DECAY_DIR, "default_params.yml")

    candidates = [
        config_path,
        os.path.join(REPO_ROOT, config_path),
        os.path.join(DECAY_DIR, config_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        f"Decay config not found: {config_path}\n"
        f"Tried:\n  - {candidates[0]}\n  - {candidates[1]}\n  - {candidates[2]}"
    )


def _load_yaml_decay_block(config_path: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
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
    Must stay aligned with decay/simulate_decay.py.
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


def _enforce_manifest_and_get_run_hash(params_path: str, sim_effective: Dict[str, Any]) -> Tuple[str, str, dict]:
    """
    Require decay/output/run_manifest.json and enforce YAML hash match.
    Returns (run_hash, manifest_params_path, manifest_dict).
    """
    man_path = os.path.join(DECAY_OUT, "run_manifest.json")
    if not os.path.exists(man_path):
        raise RuntimeError(
            "Missing decay/output/run_manifest.json.\n"
            "Run decay/simulate_decay.py first (it must write the manifest)."
        )

    manifest = _read_json(man_path)
    run_hash = str(manifest.get("run_hash", ""))
    if not run_hash:
        raise RuntimeError("run_manifest.json missing key 'run_hash' (or empty).")

    yaml_hash = _compute_run_hash(sim_effective)
    if yaml_hash != run_hash:
        raise RuntimeError(
            "Decay simulation hash mismatch: YAML-derived effective params do not match the manifest.\n"
            f"  YAML-derived hash: {yaml_hash}\n"
            f"  manifest.run_hash: {run_hash}\n"
            "Fix: delete decay/output/* and re-run decay/simulate_decay.py with the intended YAML."
        )

    man_params_path = str(manifest.get("params_path", "NA"))
    if man_params_path not in ("", "NA", None):
        if os.path.abspath(man_params_path) != os.path.abspath(params_path):
            raise RuntimeError(
                "params_path mismatch between run_manifest.json and resolved config.\n"
                f"  manifest.params_path={man_params_path}\n"
                f"  resolved params_path={params_path}\n"
                "Fix: delete decay/output/* and regenerate."
            )

    return run_hash, man_params_path, manifest


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def _require_csv(path: str, required_cols: Optional[set] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    if required_cols is not None:
        missing = required_cols.difference(df.columns)
        if missing:
            raise RuntimeError(f"{path} missing required columns: {sorted(missing)}")
    return df


def _load_tau_fut_seconds(decay_output_dir: str) -> Tuple[float, float, float]:
    f1 = os.path.join(decay_output_dir, "fit_decay_results.csv")
    df = _require_csv(f1, required_cols={"tau_hat_ms", "ci_lo_ms", "ci_hi_ms"})
    tau_s = float(df["tau_hat_ms"].iloc[0]) / 1000.0
    lo_s = float(df["ci_lo_ms"].iloc[0]) / 1000.0
    hi_s = float(df["ci_hi_ms"].iloc[0]) / 1000.0
    if not np.isfinite(tau_s) or tau_s <= 0:
        raise RuntimeError("Invalid τ̂_fut read from fit_decay_results.csv.")
    return tau_s, lo_s, hi_s


def _resolve_detection_threshold(p_raw: Dict[str, Any], band_df: pd.DataFrame) -> Tuple[float, float]:
    mode = str(p_raw.get("detection_mode", "param")).lower().strip()
    A_min = _safe_float(p_raw.get("A_min", p_raw.get("epsilon_detection", 0.01)), 0.01)

    if mode == "auto":
        idx = int(np.argmax(band_df["delta_ms"].to_numpy()))
        lnA_min = float(band_df["lnA_low"].iloc[idx])
        A_min = float(np.exp(lnA_min))
    else:
        lnA_min = float(np.log(A_min))

    return A_min, lnA_min


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _short_path(path: str, keep: int = 2) -> str:
    p = os.path.abspath(path)
    parts = p.replace("\\", "/").split("/")
    if len(parts) <= keep:
        return p
    return "…/" + "/".join(parts[-keep:])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args():
    ap = argparse.ArgumentParser(description="Box-2(a) decay figure (manifest-verified; provenance-visible).")
    ap.add_argument("--config", default=None, help="Optional YAML path; overrides CRI_DECAY_CONFIG and default.")
    ap.add_argument("--inset-x", type=float, default=float(os.getenv("CRI_INSET_X", 0.14)))
    ap.add_argument("--inset-y", type=float, default=float(os.getenv("CRI_INSET_Y", 0.28)))
    ap.add_argument("--inset-w", type=float, default=float(os.getenv("CRI_INSET_W", 0.22)))
    ap.add_argument("--inset-h", type=float, default=float(os.getenv("CRI_INSET_H", 0.20)))

    # Annotation placement (UPDATED defaults: up+left; right-anchored to prevent clipping)
    ap.add_argument("--ann-x", type=float, default=float(os.getenv("CRI_ANN_X", 0.92)),
                    help="Annotation x in axes fraction (0=left, 1=right).")
    ap.add_argument("--ann-yfrac", type=float, default=float(os.getenv("CRI_ANN_YFRAC", 0.62)),
                    help="Annotation y as fraction between ymin..ymax (0=bottom, 1=top).")
    ap.add_argument("--ann-ha", default=os.getenv("CRI_ANN_HA", "right"),
                    choices=["left", "center", "right"],
                    help="Horizontal alignment for the annotation text box (default: right).")

    ap.add_argument("--panel-label", default=os.getenv("CRI_PANEL_LABEL", "(a)"))
    ap.add_argument("--panel-x", type=float, default=float(os.getenv("CRI_PANEL_X", 0.008)))
    ap.add_argument("--panel-y", type=float, default=float(os.getenv("CRI_PANEL_Y", 0.975)))
    ap.add_argument("--out-stem", default=os.getenv("CRI_BOX2A_STEM", "Box2a_decay"))

    # Provenance rendering controls (layout defaults)
    ap.add_argument("--show-provenance", dest="show_provenance", action="store_true", default=True)
    ap.add_argument("--no-provenance", dest="show_provenance", action="store_false",
                    help="Disable visible provenance footer.")
    ap.add_argument("--prov-fontsize", type=float, default=float(os.getenv("CRI_PROV_FONTSIZE", 5.2)))
    ap.add_argument("--prov-hash-chars", type=int, default=int(os.getenv("CRI_PROV_HASH_CHARS", 12)))
    ap.add_argument("--prov-path-keep", type=int, default=int(os.getenv("CRI_PROV_PATH_KEEP", 2)))
    ap.add_argument("--prov-x", type=float, default=float(os.getenv("CRI_PROV_X", 0.001)),
                    help="Provenance x in figure coords (0=left edge, 1=right edge).")
    ap.add_argument("--prov-y", type=float, default=float(os.getenv("CRI_PROV_Y", 0.012)),
                    help="Provenance y in figure coords (0=bottom edge, 1=top edge).")

    # X-label shift right to avoid collisions
    ap.add_argument("--xlabel-x", type=float, default=float(os.getenv("CRI_XLABEL_X", 0.995)),
                    help="X-label x position in axes coords (0=left, 1=right).")

    # Styling knobs (requested)
    ap.add_argument("--sample-color", default=os.getenv("CRI_SAMPLE_COLOR", "tab:orange"),
                    help="Color for sampled delays points/errorbars.")
    ap.add_argument("--detbound-color", default=os.getenv("CRI_DETBOUND_COLOR", "tab:red"),
                    help="Color for detection bound line.")
    return ap.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = _parse_args()

    # --- YAML + manifest enforcement ---
    p_raw, params_path = _load_yaml_decay_block(args.config)
    sim_effective = _effective_sim_params_from_yaml(p_raw)
    run_hash, _, _ = _enforce_manifest_and_get_run_hash(params_path, sim_effective)

    # --- Inputs from decay/output ---
    pts = _require_csv(os.path.join(DECAY_OUT, "decay_data.csv"), required_cols={"delta", "lnA_pre"})
    band = _require_csv(
        os.path.join(DECAY_OUT, "decay_band.csv"),
        required_cols={"delta_cont", "lnA_central", "lnA_low", "lnA_high"},
    )

    # Convert to ms for plotting
    pts = pts.copy()
    band = band.copy()
    pts["delta_ms"] = pts["delta"].astype(float) * 1000.0
    band["delta_ms"] = band["delta_cont"].astype(float) * 1000.0

    # Raw amplitude inset
    A_pts = np.exp(pts["lnA_pre"].astype(float).to_numpy())
    A_central = np.exp(band["lnA_central"].astype(float).to_numpy())

    # τ_fut annotation from fit results
    tau_s, lo_s, hi_s = _load_tau_fut_seconds(DECAY_OUT)
    tau_ms = tau_s * 1e3
    slope_per_s = -1.0 / tau_s

    # Detection bound
    A_min, lnA_min = _resolve_detection_threshold(p_raw, band)

    # Figure
    fig, ax = plt.subplots(figsize=(88 / 25.4, 58 / 25.4))

    # More bottom room so xlabel + provenance never collide or clip
    fig.subplots_adjust(top=0.965, bottom=0.21)

    # Panel label outside axes
    panel_label = args.panel_label
    if panel_label.startswith("(") and panel_label.endswith(")"):
        panel_label = rf"$({panel_label.strip('()')})$"
    fig.text(
        float(args.panel_x), float(args.panel_y), panel_label,
        transform=fig.transFigure, ha="left", va="top",
        fontsize=9, color="black", clip_on=False, zorder=10
    )

    # CI band
    ax.fill_between(
        band["delta_ms"], band["lnA_low"], band["lnA_high"],
        alpha=0.35, zorder=1,
        label=f"{_safe_float(p_raw.get('ci_percent', 95.0), 95.0):.0f}% sim. bootstrap band",
    )

    # Central line
    ax.plot(
        band["delta_ms"], band["lnA_central"],
        color="black", linewidth=1.3, zorder=2,
        label=r"$\ln A_{\mathrm{pre}}(\tau_f)$",
    )

    # Points (requested: orange)
    sample_color = str(args.sample_color)
    if {"lnA_pre", "se_lnA"}.issubset(pts.columns):
        ax.errorbar(
            pts["delta_ms"], pts["lnA_pre"], yerr=pts["se_lnA"],
            fmt="o", markersize=3.8, elinewidth=0.75, capsize=1.8,
            color=sample_color, ecolor=sample_color,
            markerfacecolor=sample_color, markeredgecolor=sample_color,
            zorder=3, label="Sampled delays",
        )
    else:
        ax.scatter(
            pts["delta_ms"], pts["lnA_pre"],
            s=16, color=sample_color, edgecolors=sample_color,
            zorder=3, label="Sampled delays"
        )

    # Detection bound (requested: red; dotted)
    det_color = str(args.detbound_color)
    ax.axhline(
        lnA_min, linestyle=":", linewidth=1.2, color=det_color, zorder=0,
        label=rf"Detection bound: $\ln A_{{\min}}={lnA_min:.2f}$",
    )

    # Axes labels
    ax.set_xlabel(r"$\tau_f$ (ms)")
    ax.set_ylabel(r"$\ln A_{\mathrm{pre}}(\tau_f)$ (–)")

    # MOVE xlabel right (and right-align) to avoid provenance collision
    try:
        lab = ax.xaxis.get_label()
        _, y0 = lab.get_position()
        lab.set_position((float(args.xlabel_x), y0))  # axes coords
        lab.set_horizontalalignment("right")
    except Exception:
        pass

    # x-limits from YAML
    x_lo = float(sim_effective["delta_start"]) * 1000.0 - 0.5
    x_hi = float(sim_effective["delta_end"]) * 1000.0 + 0.5
    ax.set_xlim(x_lo, x_hi)

    # Annotation block (UPDATED: moved up/left defaults; right-anchored to avoid clipping)
    ymin, ymax = ax.get_ylim()
    y_ann = ymin + _clamp(float(args.ann_yfrac), 0.0, 1.0) * (ymax - ymin)
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    ci_line = (
        rf"\n$[{lo_s*1e3:.1f},{hi_s*1e3:.1f}]\,\mathrm{{ms}}$"
        if (np.isfinite(lo_s) and np.isfinite(hi_s))
        else ""
    )
    ann_text = (
        rf"$\mathrm{{slope}}={slope_per_s:.1f}\,\mathrm{{s}}^{{-1}}$"
        + "\n"
        + rf"$\hat{{\tau}}_{{\mathrm{{fut}}}}={tau_ms:.1f}\,\mathrm{{ms}}$"
        + ci_line
    )
    ax.text(
        float(args.ann_x), y_ann, ann_text,
        transform=trans, fontsize=6.0, va="top", ha=str(args.ann_ha),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.80, edgecolor="none"),
        clip_on=True,
    )

    # Inset (raw amplitude)
    inset_w = float(args.inset_w)
    inset_h = float(args.inset_h)
    ix = _clamp(float(args.inset_x), 0.0, 1.0 - inset_w - 0.01)
    iy = _clamp(float(args.inset_y), 0.0, 1.0 - inset_h - 0.01)

    ax_ins = inset_axes(
        ax,
        width=f"{inset_w*100:.0f}%",
        height=f"{inset_h*100:.0f}%",
        loc="lower left",
        bbox_to_anchor=(ix, iy, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0.2,
    )
    ax_ins.plot(band["delta_ms"], A_central, color="black", linewidth=0.9, zorder=2)
    ax_ins.scatter(pts["delta_ms"], A_pts, s=12, color=sample_color, edgecolors=sample_color, zorder=3)
    ax_ins.axhline(A_min, linestyle=":", linewidth=0.9, color=det_color, zorder=1)
    ax_ins.set_title(r"Raw $A_{\mathrm{pre}}(\tau_f)$", fontsize=6.5, pad=1.5)
    ax_ins.set_xlabel(r"$\tau_f$ (ms)", fontsize=6.5)
    ax_ins.set_ylabel(r"$A_{\mathrm{pre}}$ (–)", fontsize=6.5)
    ax_ins.tick_params(labelsize=6)
    ax_ins.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
    ax_ins.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_ins.set_xlim(x_lo, x_hi)

    # Legend
    leg = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, fancybox=True)
    fr = leg.get_frame()
    fr.set_alpha(1.0)
    fr.set_linewidth(0.6)

    # Visible provenance footer (left-anchored; pushed further left by default)
    if bool(args.show_provenance):
        shown_hash = run_hash[: max(4, int(args.prov_hash_chars))]
        shown_cfg = _short_path(params_path, keep=max(1, int(args.prov_path_keep)))
        prov_text = f"run_hash={shown_hash}…  |  cfg={shown_cfg}"

        fig.text(
            float(args.prov_x), float(args.prov_y), prov_text,
            transform=fig.transFigure,
            ha="left", va="bottom",
            fontsize=float(args.prov_fontsize),
            color="black",
            zorder=20,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85, edgecolor="none"),
        )

    # Outputs + metadata
    out_pdf = os.path.join(FIG_OUT, f"{args.out_stem}.pdf")
    out_png = os.path.join(FIG_OUT, f"{args.out_stem}.png")

    meta_pdf = {
        "Title": "CRI Box-2(a): decay calibration",
        "Author": "CRI pipeline",
        "Subject": f"run_hash={run_hash}; params_path={os.path.abspath(params_path)}",
        "Keywords": "CRI; decay; bootstrap; reproducibility",
    }

    pnginfo = None
    if _HAVE_PIL:
        try:
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("Software", "CRI pipeline")
            pnginfo.add_text("run_hash", str(run_hash))
            pnginfo.add_text("params_path", os.path.abspath(params_path))
        except Exception:
            pnginfo = None

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01, metadata=meta_pdf)
    fig.savefig(
        out_png,
        dpi=int(_safe_int(p_raw.get("figure_dpi", 1200), 1200)),
        bbox_inches="tight",
        pad_inches=0.01,
        pil_kwargs=({"pnginfo": pnginfo} if pnginfo is not None else None),
    )

    plt.close(fig)
    print(
        "Saved Box-2(a) to",
        out_pdf,
        "and",
        out_png,
        f"\nrun_hash={run_hash}\nconfig={os.path.abspath(params_path)}",
    )


if __name__ == "__main__":
    main()

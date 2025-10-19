# -*- coding: utf-8 -*-
"""
figures/make_decay_figure.py

Changes vs prior:
- Use DejaVu Sans (silences Arial warnings on Linux runners).
- Read τ̂_fut robustly from decay/output/* and annotate it.
- Overlay CRI reference line with slope = -1 / τ̂_fut on the log plot.
- Keep wide 95% CI band, orange samples, red detection bound.
- Legend moved to upper-right with a light gray background.
- Inset moved to lower-left by default and reduced to ~1/3 of prior area.
- Inset position & size are adjustable via CLI flags or env vars.
- X-axis fixed to 0–20 ms.
- Avoid tight_layout() warning (save with bbox_inches='tight').
- Detection bound: A_min (log plot shows ln A_min). Backward-compatible
  with legacy epsilon_detection; optional 'auto' mode from bootstrap band.

Inset controls (axes coordinates / size fractions):
    --inset-x 0.08  --inset-y 0.10  --inset-w 0.19  --inset-h 0.17
or environment variables:
    CRI_INSET_X=0.08 CRI_INSET_Y=0.10 CRI_INSET_W=0.19 CRI_INSET_H=0.17
"""
import os
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
from matplotlib import transforms as mtransforms

# --- Matplotlib defaults (portable on CI) ------------------------------------
mpl.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        8,
    "axes.linewidth":   0.6,
    "lines.linewidth":  0.9,
    "legend.fontsize":  5.5,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ---------------------------------------------------------------------------

def load_params(path):
    with open(path, 'r', encoding='utf-8') as f:
        y = yaml.safe_load(f)
    # Expect 'decay' root; tolerate flat YAML
    return y['decay'] if isinstance(y, dict) and 'decay' in y else y

def _load_tau_fut_seconds(decay_output_dir):
    """
    Try multiple schemas to obtain τ̂_fut (in seconds) and its CI.
    Returns (tau_s, lo_s, hi_s) where any of them may be np.nan if unavailable.
    """
    tau_s = lo_s = hi_s = np.nan

    # 1) fit_decay_results.csv  (preferred if present)
    f1 = os.path.join(decay_output_dir, 'fit_decay_results.csv')
    if os.path.exists(f1):
        try:
            df = pd.read_csv(f1)
            if 'tau_hat_ms' in df.columns:
                tau_s = float(df['tau_hat_ms'].iloc[0]) / 1000.0
            if 'ci_lo_ms' in df.columns:
                lo_s = float(df['ci_lo_ms'].iloc[0]) / 1000.0
            if 'ci_hi_ms' in df.columns:
                hi_s = float(df['ci_hi_ms'].iloc[0]) / 1000.0
        except Exception:
            pass

    # 2) decay_band.csv with a "tau" row (center_s/lo_s/hi_s)
    if not np.isfinite(tau_s):
        f2 = os.path.join(decay_output_dir, 'decay_band.csv')
        if os.path.exists(f2):
            try:
                band_meta = pd.read_csv(f2)
                if 'name' in band_meta.columns and 'center_s' in band_meta.columns:
                    row = band_meta.loc[band_meta['name'].str.contains('tau', case=False)]
                    if not row.empty:
                        r0 = row.iloc[0]
                        tau_s = float(r0['center_s'])
                        lo_s  = float(r0.get('lo_s', np.nan))
                        hi_s  = float(r0.get('hi_s', np.nan))
            except Exception:
                pass

    # 3) decay_fit.csv / decay_summary.csv fallbacks (ms columns)
    for fname in ('decay_fit.csv', 'decay_summary.csv'):
        if not np.isfinite(tau_s):
            f = os.path.join(decay_output_dir, fname)
            if os.path.exists(f):
                try:
                    df = pd.read_csv(f)
                    if 'tau_ms' in df.columns:
                        tau_s = float(df['tau_ms'].iloc[0]) / 1000.0
                    if 'ci_lo_ms' in df.columns:
                        lo_s = float(df['ci_lo_ms'].iloc[0]) / 1000.0
                    if 'ci_hi_ms' in df.columns:
                        hi_s = float(df['ci_hi_ms'].iloc[0]) / 1000.0
                except Exception:
                    pass

    if not np.isfinite(tau_s):
        raise RuntimeError("Could not find τ̂_fut in decay/output/.")

    return tau_s, lo_s, hi_s

def resolve_detection_threshold(p, band):
    """
    Returns (A_min, lnA_min) for the detection bound.

    Modes:
      - 'param' (default): use p['A_min'] (or legacy p['epsilon_detection']) as a constant floor.
      - 'auto': pick lnA_min from the bootstrap lower band at the largest delay (conservative).

    Keys in YAML (under 'decay'):
      detection_mode: 'param' | 'auto'
      A_min: 0.01
      epsilon_detection: 0.01  # legacy; used if A_min not present
    """
    mode = str(p.get('detection_mode', 'param')).lower()
    A_min = p.get('A_min', p.get('epsilon_detection', 0.01))
    try:
        A_min = float(A_min)
    except Exception:
        A_min = 0.01

    if mode == 'auto':
        idx = int(np.argmax(band['delta_ms'].values))
        lnA_min = float(band['lnA_low'].iloc[idx])
        A_min = float(np.exp(lnA_min))
    else:
        lnA_min = float(np.log(A_min))

    return A_min, lnA_min

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CRI decay figure with adjustable inset.")
    # Inset position (axes coords)
    parser.add_argument("--inset-x", type=float, default=float(os.getenv("CRI_INSET_X", 0.08)),
                        help="Inset lower-left x in axes coords [0..1].")
    parser.add_argument("--inset-y", type=float, default=float(os.getenv("CRI_INSET_Y", 0.10)),
                        help="Inset lower-left y in axes coords [0..1].")
    # Inset size (fractions of axes; converted to percentages for inset_axes)
    parser.add_argument("--inset-w", type=float, default=float(os.getenv("CRI_INSET_W", 0.19)),
                        help="Inset width as fraction of main axes (e.g., 0.19 ≈ 19%).")
    parser.add_argument("--inset-h", type=float, default=float(os.getenv("CRI_INSET_H", 0.17)),
                        help="Inset height as fraction of main axes (e.g., 0.17 ≈ 17%).")
    return parser.parse_args()

def main():
    args = parse_args()

    here         = os.path.dirname(__file__)
    repo_root    = os.path.abspath(os.path.join(here, os.pardir))
    decay_folder = os.path.join(repo_root, 'decay')
    out_folder   = os.path.join(here, 'output')
    out_dir_data = os.path.join(decay_folder, 'output')

    p = load_params(os.path.join(decay_folder, 'default_params.yml'))

    # Core series
    pts  = pd.read_csv(os.path.join(out_dir_data, 'decay_data.csv'))
    band = pd.read_csv(os.path.join(out_dir_data, 'decay_band.csv'))

    # x in ms
    pts['delta_ms']  = pts['delta'] * 1000.0
    band['delta_ms'] = band['delta_cont'] * 1000.0

    # Convert central curve to linear for the inset
    A_pts     = np.exp(pts['lnA_pre'].values)
    A_central = np.exp(band['lnA_central'].values)

    # Get τ̂_fut (seconds) + CI
    tau_s, lo_s, hi_s = _load_tau_fut_seconds(out_dir_data)  # raises if not found
    tau_ms = tau_s * 1e3
    slope_per_s = -1.0 / tau_s

    # Resolve detection threshold (param or auto)
    A_min, lnA_min = resolve_detection_threshold(p, band)

    # Figure
    fig, ax = plt.subplots(figsize=(88/25.4, 58/25.4))

    # 95% CI band (wide/visible)
    ax.fill_between(
        band['delta_ms'], band['lnA_low'], band['lnA_high'],
        facecolor='#5B8FD9', alpha=0.58, zorder=1,
        edgecolor='#3E6FB8', linewidth=0.9,
        label=f"{p.get('ci_percent', 95)}% CI (bootstrap)"
    )

    # Central fitted curve (black)
    ax.plot(
        band['delta_ms'], band['lnA_central'],
        color='black', linewidth=1.3, zorder=2,
        label=r"$\ln A_{\mathrm{pre}}(\tau_f)$"
    )

    # Orange samples + SE bars
    if {'lnA_pre', 'se_lnA'}.issubset(pts.columns):
        ax.errorbar(
            pts['delta_ms'], pts['lnA_pre'], yerr=pts['se_lnA'],
            fmt='o', color='#FF8C1A', markersize=3.8, elinewidth=0.75,
            capsize=1.8, zorder=3, label="Sampled delays"
        )
    else:
        ax.scatter(
            pts['delta_ms'], pts['lnA_pre'],
            s=16, color='#FF8C1A', zorder=3, label="Sampled delays"
        )

    # Detection bound (log plot shows ln A_min)
    ax.axhline(lnA_min, linestyle='--', color='#D62728', linewidth=1.0,
               label=r"Detection bound: $\ln A_{\min}$")

    # Axes/limits
    ax.set_xlabel(r"$\tau_f$ (ms)")
    ax.set_ylabel(r"$\ln A_{\mathrm{pre}}(\tau_f)$")
    ax.set_xlim(-0.5, 20.5)

    # Legend: upper-right with light gray background
    leg = ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98),
                    frameon=True, fancybox=True)
    frame = leg.get_frame()
    frame.set_facecolor('#f2f2f2')
    frame.set_edgecolor('0.80')
    frame.set_alpha(1.0)
    frame.set_linewidth(0.6)

    # --- CRI reference line with slope = -1/τ̂_fut --------------------------
    i0 = int(np.argmin(band['delta_ms'].values))
    x0_ms = float(band['delta_ms'].values[i0])
    y0    = float(band['lnA_central'].values[i0])

    x_line_ms = np.linspace(0.0, 20.0, 200)
    x_line_s  = x_line_ms / 1000.0
    x0_s      = x0_ms / 1000.0
    y_cri     = y0 + slope_per_s * (x_line_s - x0_s)

    ax.plot(x_line_ms, y_cri, ls='--', lw=1.1, color='0.25',
            label=rf"CRI slope −1/τ$_{{\mathrm{{fut}}}}$ ({tau_ms:.1f} ms)")

    # Slope/τ annotation
    ymin, ymax = ax.get_ylim()
    y_ann = ymin + 0.15 * (ymax - ymin)  # 15% above bottom; bump to 0.12–0.15 if needed
    x_ann_axes = 0.60
    ann_txt = (r"$\mathrm{slope} = -1/\tau_{\mathrm{fut}}$"
               + "\n" + rf"$\hat{{\tau}}_{{\mathrm{{fut}}}}={tau_ms:.1f}\,\mathrm{{ms}}$")
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        x_ann_axes, y_ann, ann_txt, transform=trans, fontsize=6.0, va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.80, edgecolor="none")
    )

    # ---------------- Inset (adjustable) ------------------------------------
    #   - Position: args.inset_x / args.inset_y (axes coords, 0..1)
    #   - Size:     args.inset_w / args.inset_h (fractions of axes; turned into "%")
    inset_w_pct = f"{args.inset_w*100:.0f}%"
    inset_h_pct = f"{args.inset_h*100:.0f}%"
    
    args.inset_x = min(args.inset_x + 0.05, 1.0 - args.inset_w - 0.01)
    
    ax_ins = inset_axes(
        ax, width=inset_w_pct, height=inset_h_pct,
        loc='lower left',
        bbox_to_anchor=(args.inset_x, args.inset_y, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0.2
    )
    ax_ins.plot(band['delta_ms'], A_central, color='black', linewidth=0.9, zorder=2)
    ax_ins.scatter(pts['delta_ms'], A_pts, s=12, color='#FF8C1A', zorder=3)
    ax_ins.axhline(A_min, linestyle='--', color='#D62728', linewidth=0.7)

    ax_ins.set_title(r"Raw $A_{\mathrm{pre}}(\tau_f)$", fontsize=6.5, pad=1.5)
    ax_ins.set_xlabel(r"$\tau_f$ (ms)", fontsize=6.5)
    ax_ins.set_ylabel(r"$A_{\mathrm{pre}}$", fontsize=6.5)
    ax_ins.tick_params(labelsize=6)
    ax_ins.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
    ax_ins.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_ins.set_xlim(-0.5, 20.5)

    # Save (avoid tight_layout to prevent warning with inset)
    os.makedirs(out_folder, exist_ok=True)
    out_pdf = os.path.join(out_folder, 'Box2a_decay_refined.pdf')
    out_png = os.path.join(out_folder, 'Box2a_decay_refined.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=int(p.get('figure_dpi', 1200)), bbox_inches='tight')
    plt.close(fig)
    print("Saved Box-2(a) to", out_pdf, "and", out_png)

if __name__ == "__main__":
    main()

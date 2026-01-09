# -*- coding: utf-8 -*-
"""
EEG preprocessing flowchart → clean, numbered left-to-right pipeline (PNG + PDF)
Saves into figures/output/.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ----------------------------
# Style / export robustness
# ----------------------------
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType (better portability)
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.facecolor"] = "white"


# Define colors (TikZ RGB → matplotlib 0–1)
COLORS = {
    "rawcol":    (255/255, 102/255, 102/255),
    "bpcol":     (102/255, 178/255, 255/255),
    "notchcol":  (102/255, 204/255, 102/255),
    "icacol":    (255/255, 178/255, 102/255),
    "interpcol": (178/255, 102/255, 255/255),
    "epochcol":  (255/255, 255/255, 102/255),
    "basecol":   (102/255, 255/255, 178/255),
    "detrcol":   (255/255, 102/255, 178/255),
    "burstcol":  (178/255, 255/255, 102/255),
    "statcol":   (102/255, 255/255, 255/255),
}

# Sequential pipeline steps 
STEPS = [
    ("Raw EEG", "rawcol"),
    ("Band-pass\n0.5–45 Hz", "bpcol"),
    ("Notch\n50/60 Hz", "notchcol"),
    ("ICA & Artifact\nRejection", "icacol"),
    ("Channel\nInterpolation", "interpcol"),
    ("Epoch Segmentation\n(–30 s to 0 s)", "epochcol"),
    ("Baseline\nCorrection", "basecol"),
    ("Detrending", "detrcol"),
    ("Burst Detection\n(12–15 Hz)", "burstcol"),
    ("Statistical Validation\n(Permutation)", "statcol"),
]


def _draw_box(ax, x, y, w, h, text, color_rgb, step_idx,
              face_alpha=0.18, edge_lw=2.2, rounding=0.10):
    """Rounded rectangle node with step number and centered label."""
    face = (*color_rgb, face_alpha)
    patch = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.03,rounding_size={rounding}",
        linewidth=edge_lw,
        edgecolor=color_rgb,
        facecolor=face,
        zorder=2
    )
    ax.add_patch(patch)

    # Step number (top-left within the box)
    ax.text(
        x - w/2 + 0.14, y + h/2 - 0.14, f"{step_idx}.",
        ha="left", va="top",
        fontsize=12, fontweight="bold",
        fontname="DejaVu Sans",
        color="black", zorder=4
    )

    # Main label (center)
    ax.text(
        x, y - 0.03, text,
        ha="center", va="center",
        fontsize=11.5, fontweight="bold",
        fontname="DejaVu Sans",
        color="black", zorder=4,
        linespacing=1.15
    )


def _arrow(ax, start, end, lw=2.2, color="0.20", mscale=14):
    """Simple arrow with small head; avoids the giant filled-triangle issue."""
    ax.annotate(
        "",
        xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=mscale
        ),
        zorder=3
    )


def make_clean_flowchart(out_dir: Path, stem_name: str = "SI_Fig1_EEG_flowchart"):
    """
    Generates a two-row serpentine layout:
      Row 1: steps 1→5 left-to-right
      Row 2: steps 6→10 right-to-left
    This keeps connectors minimal and avoids crossings while remaining unambiguous
    due to numbering + arrow direction.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / stem_name

    n_steps = len(STEPS)
    n_cols = 5  # 10 steps → 2 rows × 5 columns

    # Layout geometry (axis units)
    x_spacing = 2.75
    y_top, y_bot = 1.35, 0.00

    # Node size (axis units)
    box_w, box_h = 2.35, 0.82

    # Coordinates (serpentine / "snake" order to avoid long wrap-around arrows)
    xs_top = [i * x_spacing for i in range(n_cols)]
    xs_bot = list(reversed(xs_top))

    coords = []
    for i in range(n_steps):
        if i < n_cols:
            coords.append((xs_top[i], y_top))
        else:
            coords.append((xs_bot[i - n_cols], y_bot))

    # Figure sizing
    fig_w = (n_cols - 1) * x_spacing + 5.0
    fig_h = 4.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")

    # Draw nodes
    for idx, ((label, col_key), (x, y)) in enumerate(zip(STEPS, coords), start=1):
        _draw_box(ax, x, y, box_w, box_h, label, COLORS[col_key], idx)

    # Draw arrows between successive steps
    for i in range(n_steps - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]

        same_row = abs(y2 - y1) < 1e-9

        if same_row:
            # Horizontal arrow: from box edge to next box edge
            if x2 > x1:
                start = (x1 + box_w/2, y1)
                end = (x2 - box_w/2, y2)
            else:
                start = (x1 - box_w/2, y1)
                end = (x2 + box_w/2, y2)
        else:
            # Vertical arrow: from bottom edge to top edge (step 5 → step 6)
            start = (x1, y1 - box_h/2)
            end = (x2, y2 + box_h/2)

        _arrow(ax, start, end)

    # Title & caption
    x_mid = (xs_top[0] + xs_top[-1]) / 2
    ax.text(
        x_mid, y_top + 0.80,
        "EEG Preprocessing Pipeline",
        ha="center", va="bottom",
        fontsize=22, fontweight="bold",
        fontname="DejaVu Sans",
        color="black"
    )
    ax.text(
        x_mid, y_bot - 0.82,
        "Numbered pipeline with minimal connectors (SI Fig. SI-3).",
        ha="center", va="top",
        fontsize=13,
        fontname="DejaVu Sans",
        color="black"
    )

    # Axis limits (auto-fit with margins)
    x_min = min(xs_top) - (box_w/2 + 0.45)
    x_max = max(xs_top) + (box_w/2 + 0.45)
    y_min = y_bot - 1.05
    y_max = y_top + 1.15
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Export
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", transparent=False)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")  # vector export
    plt.close(fig)

    print(f"Saved {stem.with_suffix('.png')} and {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    # Robust location handling: works in scripts and interactive sessions.
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    out_dir = here / "output"
    make_clean_flowchart(out_dir=out_dir, stem_name="SI_Fig1_EEG_flowchart")

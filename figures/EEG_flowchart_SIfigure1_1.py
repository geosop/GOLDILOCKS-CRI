# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 16:01:08 2025

@author: ADMIN

EEG preprocessing flowchart → PNG + PDF into figures/output/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe
from pathlib import Path

# Define colors (TikZ RGB to matplotlib 0-1 scale)
colors = {
    'rawcol':      (255/255,102/255,102/255),
    'bpcol':       (102/255,178/255,255/255),
    'notchcol':    (102/255,204/255,102/255),
    'icacol':      (255/255,178/255,102/255),
    'interpcol':   (178/255,102/255,255/255),
    'epochcol':    (255/255,255/255,102/255),
    'basecol':     (102/255,255/255,178/255),
    'detrcol':     (255/255,102/255,178/255),
    'burstcol':    (178/255,255/255,102/255),
    'statcol':     (102/255,255/255,255/255),
}

labels = [
    ("Raw EEG", 'rawcol', 'rect'),
    ("Band-pass\n0.5–45 Hz", 'bpcol', 'ellipse'),
    ("Notch\n50/60 Hz", 'notchcol', 'ellipse'),
    ("ICA & Artifact\nRejection", 'icacol', 'ellipse'),
    ("Channel\nInterpolation", 'interpcol', 'ellipse'),
    ("Epoch Segmentation\n(–30 s to 0 s)", 'epochcol', 'ellipse'),
    ("Baseline\nCorrection", 'basecol', 'ellipse'),
    ("Detrending", 'detrcol', 'ellipse'),
    ("Burst Detection\n(12–15 Hz)", 'burstcol', 'ellipse'),
    ("Statistical Validation\n(Permutation)", 'statcol', 'ellipse'),
]

N = len(labels)
angle = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, N, endpoint=False)
rx, ry = 4.0, 2.0
positions = [(rx * np.cos(a), ry * np.sin(a)) for a in angle]

# --- Figure ---
fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1.6, 1.6)
plt.axis('off')

# --- Draw nodes ---
node_width = 3.2    # cm
node_height = 0.8   # cm
scale = 0.4         # convert cm to approx axis units

for (text, col, shape), (x, y) in zip(labels, positions):
    if shape == 'rect':
        rect = Rectangle(
            (x - scale*node_width/2, y - scale*node_height/2),
            scale*node_width, scale*node_height,
            linewidth=2, edgecolor=colors[col], facecolor=colors[col],
            alpha=0.85, zorder=3,
            path_effects=[pe.withStroke(linewidth=4, foreground='k')]
        )
        ax.add_patch(rect)
    else:
        ell = Ellipse(
            (x, y), width=scale*node_width, height=scale*node_height,
            linewidth=2, edgecolor=colors[col], facecolor=colors[col],
            alpha=0.85, zorder=3,
            path_effects=[pe.withStroke(linewidth=4, foreground='k')]
        )
        ax.add_patch(ell)

    ax.text(
        x, y, text, ha='center', va='center', fontsize=13, fontweight='bold',
        fontname='DejaVu Sans', color='k', zorder=5, linespacing=1.1,
        path_effects=[pe.withStroke(linewidth=4, foreground='white', alpha=0.5)]
    )

# --- Draw arrows ---
for i in range(N):
    x1, y1 = positions[i]
    x2, y2 = positions[(i + 1) % N]
    color = colors[labels[i][1]]
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    shorten = 0.37
    x_start = x1 + dx * shorten / length
    y_start = y1 + dy * shorten / length
    x_end = x2 - dx * shorten / length
    y_end = y2 - dy * shorten / length

    ax.add_patch(FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        arrowstyle='->,head_length=10,head_width=6',
        mutation_scale=20, linewidth=2.5,
        color=color, alpha=0.95, zorder=10,
        path_effects=[pe.withStroke(linewidth=4, foreground='k', alpha=0.14)]
    ))

# --- Title & caption ---
ax.text(0, 2.7, "EEG Preprocessing\nFlowchart", ha='center', va='bottom',
        fontsize=22, fontweight='bold', fontname='DejaVu Sans', color='k', zorder=100)
plt.text(0, -2.7, "Flowchart of the EEG preprocessing pipeline.",
         ha='center', va='top', fontsize=16, fontname='DejaVu Sans', color='k')

ax.set_xlim(-5.2, 5.2)
ax.set_ylim(-3.5, 3.2)
plt.tight_layout()

# --- Save outputs to figures/output ---
out_dir = Path(__file__).parent / "output"
out_dir.mkdir(parents=True, exist_ok=True)
stem = out_dir / "SI_Fig1_EEG_flowchart"

plt.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches='tight', transparent=False)
plt.savefig(stem.with_suffix(".pdf"), bbox_inches='tight')  # vector export for print quality
plt.close(fig)
print(f"Saved {stem.with_suffix('.png')} and {stem.with_suffix('.pdf')}")

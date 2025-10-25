# -*- coding: utf-8 -*-
"""
figures/make_logistic_figure.py  •  CRI v0.2-SIM (two-condition Bernoulli fit)

Main Box-2(b): logistic “tipping point” with two arousal levels.
- 95% bootstrap CI ribbons for each curve
- Solid lines for fitted sigmoids
- Orange bin-mean markers (visualization only; fits are trial-wise)
- Dashed vertical lines at p0-hats for each condition
- Inset showing dG/dq for each curve
"""
import os, yaml
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

def load_gate_params(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['logistic']

def main():
    here   = os.path.dirname(__file__)
    repo   = os.path.abspath(os.path.join(here, os.pardir))
    gate   = os.path.join(repo, 'logistic_gate')
    params = load_gate_params(os.path.join(gate, 'default_params.yml'))
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Inputs
    df_band = pd.read_csv(os.path.join(gate, 'output', 'logistic_band.csv'))
    df_fit  = pd.read_csv(os.path.join(gate, 'output', 'fit_logistic_results.csv'))
    df_bins = pd.read_csv(os.path.join(gate, 'output', 'logistic_bins.csv'))
    df_der  = pd.read_csv(os.path.join(gate, 'output', 'logistic_derivative.csv'))

    two = 'G_central_a2' in df_band.columns

    # Colors
    col_teal1 = "#6EC5B8"
    col_teal2 = "#9ADBD2"
    col_blue1 = "#1f77b4"
    col_purp2 = "#7f3c8d"
    col_orng  = "#FF8C1A"
    col_grey  = "0.45"

    fig, ax = plt.subplots(figsize=(88/25.4, (88/1.55)/25.4))

    ax.fill_between(df_band['q'], df_band['G_low_a1'], df_band['G_high_a1'],
                    facecolor=col_teal1, alpha=0.45, edgecolor=col_teal1, linewidth=0.6,
                    label=f"{params['ci_percent']}% CI (bootstrap) — a1")
    ax.plot(df_band['q'], df_band['G_central_a1'], color=col_blue1, linewidth=1.2,
            label=r"Fitted $G(q\mid a_1)$")

    if two:
        ax.fill_between(df_band['q'], df_band['G_low_a2'], df_band['G_high_a2'],
                        facecolor=col_teal2, alpha=0.45, edgecolor=col_teal2, linewidth=0.6,
                        label=f"{params['ci_percent']}% CI (bootstrap) — a2")
        ax.plot(df_band['q'], df_band['G_central_a2'], color=col_purp2, linewidth=1.2,
                label=r"Fitted $G(q\mid a_2)$")

    for a_val, df_a in df_bins.groupby('a'):
        ax.scatter(df_a['q_bin_center'], df_a['rate_mean'], s=18,
                   facecolors=col_orng, edgecolors='black', linewidths=0.4,
                   label=f"Bin means (a={a_val:.2f})")

    p0_a1 = df_fit['p0_hat_a1'].iloc[0]
    ax.axvline(p0_a1, color=col_grey, linestyle='--', linewidth=0.8,
               label=rf"$p_0(a_1)={p0_a1:.2f}$")
    if two and 'p0_hat_a2' in df_fit.columns and not np.isnan(df_fit['p0_hat_a2'].iloc[0]):
        p0_a2 = df_fit['p0_hat_a2'].iloc[0]
        ax.axvline(p0_a2, color='0.30', linestyle='--', linewidth=0.8,
                   label=rf"$p_0(a_2)={p0_a2:.2f}$")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"$q$"); ax.set_ylabel(r"$G(q\mid a)$")
    ax.legend(loc='upper left', frameon=True)

    # Inset: derivatives
    ax_ins = inset_axes(ax, width='70%', height='70%',
                        loc='lower left', bbox_to_anchor=(0.62, 0.27, 0.38, 0.38),
                        bbox_transform=ax.transAxes)
    ax_ins.plot(df_der['q'], df_der['dGdq_a1'], color=col_blue1, linewidth=0.9, label=r"$\mathrm{d}G/\mathrm{d}q$ (a$_1$)")
    ax_ins.axvline(p0_a1, color=col_grey, linestyle='--', linewidth=0.7)
    if 'dGdq_a2' in df_der.columns:
        ax_ins.plot(df_der['q'], df_der['dGdq_a2'], color=col_purp2, linewidth=0.9, label=r"$\mathrm{d}G/\mathrm{d}q$ (a$_2$)")
        ax_ins.axvline(p0_a2, color='0.30', linestyle='--', linewidth=0.7)
    ax_ins.set_title(r'$\mathrm{d}G/\mathrm{d}q$', fontsize=4)
    ax_ins.set_xlabel(r"$q$", fontsize=3); ax_ins.set_ylabel("Rate", fontsize=3)
    ax_ins.set_xlim(0, 1); ax_ins.tick_params(labelsize=3)
    ax_ins.legend(loc='upper right', frameon=False, fontsize=2)

    pdf = os.path.join(out_dir, 'Box2b_logistic_refined.pdf')
    png = os.path.join(out_dir, 'Box2b_logistic_refined.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=int(params.get('figure_dpi', 1200)), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved logistic figure → {pdf} and {png}")

if __name__=='__main__':
    main()

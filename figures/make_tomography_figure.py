# -*- coding: utf-8 -*-
"""
figures/make_tomography_figure.py  •  CRI Box-2(c) — rigor version

Implements the 4 upgrades:
  1) Calibrate κ0 from left panel (no peeking at right).
  2) Use dimensionless x on right: \tilde{λ} = λ_env / κ0_hat  (theory slope = 1).
  3) Plot empirical bootstrap CI envelope (R_ci_low/high) — no fixed-width band.
  4) Show inference: κ0_hat ± CI (left); OLS on right with slope test vs 1.

CLI flags:
  --abs-x            Use absolute λ_env on right (default: normalized).
  --no-annot         Suppress stats annotations on plots.
  --dpi 1200         Export PNG dpi (PDF is vector).

Outputs:
  figures/output/Box2c_rate_refined.pdf
  figures/output/Box2c_rate_refined.png
  figures/output/Box2c_stats.csv
"""



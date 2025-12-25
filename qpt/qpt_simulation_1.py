# -*- coding: utf-8 -*-
"""
qpt/qpt_simulation.py

Generates synthetic data for Box-2(c) with correct units:

Left panel (ms axis in the figure):
    P(t) = exp(- (kappa0/2) * (gamma_fwd + gamma_b) * t),   t in seconds.
    gamma_fwd defaults to YAML 'gamma_f' if 'gamma_fwd' is absent.

Right panel (dimensionless):
    For each λ_env in a grid, simulate bootstrap samples around the
    theory R(λ_env) = λ_env / kappa0, then report mean and 95% CI.

Writes qpt/output/qpt_sim_data.npz with:
    t [s], gamma_b_vals, pops (n_gb × n_t), lambda_env [s^-1],
    R_mean, R_ci_low, R_ci_high, and kappa0 (for reference).
"""
import os, sys, yaml
import numpy as np

# Ensure utilities on path (optional)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
try:
    from utilities.seed_manager import load_state, save_state
except Exception:
    def load_state(): pass
    def save_state(): pass

DEFAULTS = {
    # Simulation parameters (CRI-friendly defaults)
    'gamma_f': 1.0,                 # kept for backward compat
    'gamma_fwd': None,              # preferred name; if None → use gamma_f
    'gamma_b_vals': [0.2, 0.8],
    'kappa0': None,                 # s^-1; if None → derive from tau_mem_* or fallback 8.0
    'tau_mem_s': None,              # optional
    'tau_mem_ms': None,             # optional

    't_max': 0.2,                   # seconds (200 ms window)
    'n_t': 200,

    # Environmental coupling scan
    'lambda_env_min': 0.0,          # s^-1
    'lambda_env_max': 1.0,          # s^-1
    'n_lambda_env': 11,

    # Bootstrap around theory R = λ_env / kappa0
    'noise_R': 0.05,                # stdev of noise in R-units (dimensionless)
    'n_trials_per_lambda': None,    # if None, fall back to n_bootstrap
    'n_bootstrap': 1000,            # kept for backward compat
    'ci_percent': 95,

    # RNG
    'seed': 52,
}

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return float(default) if default is not None else None

def load_params():
    """Merge YAML (qpt/default_params.yml) with DEFAULTS."""
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, 'default_params.yml')
    with open(cfg_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        cfg = yaml.safe_load(f) or {}
    user = (cfg.get('qpt') or cfg or {})
    p = {**DEFAULTS, **user}

    # Resolve gamma_fwd
    if p['gamma_fwd'] is None:
        p['gamma_fwd'] = _safe_float(p.get('gamma_f', 1.0), 1.0)

    # Resolve kappa0 (s^-1): explicit → tau_mem_s → tau_mem_ms → fallback 8.0
    k0 = p.get('kappa0', None)
    if k0 is not None:
        p['kappa0'] = _safe_float(k0, 8.0)
    else:
        tau_s = _safe_float(p.get('tau_mem_s', None), None)
        tau_ms = _safe_float(p.get('tau_mem_ms', None), None)
        if tau_s and tau_s > 0:
            p['kappa0'] = 1.0 / tau_s
        elif tau_ms and tau_ms > 0:
            p['kappa0'] = 1000.0 / tau_ms
        else:
            p['kappa0'] = 8.0  # ≈ 125 ms

    # Trials per λ
    if p['n_trials_per_lambda'] is None:
        p['n_trials_per_lambda'] = int(p.get('n_bootstrap', 1000))

    # Coerce numerics
    p['gamma_fwd'] = _safe_float(p['gamma_fwd'], 1.0)
    p['gamma_b_vals'] = [float(g) for g in p['gamma_b_vals']]
    p['t_max'] = _safe_float(p['t_max'], 0.2)
    p['n_t'] = int(p['n_t'])
    p['lambda_env_min'] = _safe_float(p['lambda_env_min'], 0.0)
    p['lambda_env_max'] = _safe_float(p['lambda_env_max'], 1.0)
    p['n_lambda_env'] = int(p['n_lambda_env'])
    p['noise_R'] = _safe_float(p['noise_R'], 0.05)
    p['ci_percent'] = _safe_float(p['ci_percent'], 95.0)
    p['seed'] = int(p['seed'])
    return p

def simulate_populations(t_s, gamma_fwd, gamma_b_vals, kappa0):
    """
    Return a 2D array (n_gb × n_t) with
        P(t) = exp( - (kappa0/2) * (gamma_fwd + gamma_b) * t ).
    """
    curves = []
    for gb in gamma_b_vals:
        rate = 0.5 * kappa0 * (gamma_fwd + gb)  # s^-1
        curves.append(np.exp(-rate * t_s))
    return np.vstack(curves).astype(float)

def main():
    load_state()  # no-op if utilities not present
    p = load_params()
    rng = np.random.default_rng(p['seed'])
    save_state()

    # --- Left panel data (time in seconds) ---
    t = np.linspace(0.0, p['t_max'], p['n_t'])
    pops = simulate_populations(t, p['gamma_fwd'], p['gamma_b_vals'], p['kappa0'])

    # --- Right panel data: R = λ_env / kappa0 + noise ---
    lambda_env = np.linspace(p['lambda_env_min'], p['lambda_env_max'], p['n_lambda_env'])
    R_true = lambda_env / p['kappa0']  # dimensionless
    n_trials = int(p['n_trials_per_lambda'])
    sigma = p['noise_R']
    alpha = (100.0 - p['ci_percent']) / 100.0

    R_mean, R_low, R_high = [], [], []
    for r0 in R_true:
        samples = r0 + rng.normal(0.0, sigma, size=n_trials)
        samples = np.clip(samples, 0.0, None)
        R_mean.append(samples.mean())
        R_low.append(np.percentile(samples, 100*alpha/2))
        R_high.append(np.percentile(samples, 100*(1 - alpha/2)))

    # --- Save ---
    out = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, 'qpt_sim_data.npz'),
             t=t.astype(float),
             gamma_b_vals=np.array(p['gamma_b_vals'], dtype=float),
             pops=pops,                              # 2D (n_gb × n_t)
             lambda_env=lambda_env.astype(float),    # s^-1
             R_mean=np.array(R_mean, dtype=float),   # dimensionless
             R_ci_low=np.array(R_low, dtype=float),
             R_ci_high=np.array(R_high, dtype=float),
             kappa0=float(p['kappa0']))              # for reference
    print(f"Saved qpt_sim_data.npz   (pops shape: {pops.shape},  κ0={p['kappa0']:.3f} s^-1)")

if __name__ == '__main__':
    main()

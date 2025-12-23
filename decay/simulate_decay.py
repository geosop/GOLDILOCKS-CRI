# -*- coding: utf-8 -*-
"""
decay/simulate_decay.py  •  CRI v0.3-SIM

Generates:
  - decay/output/decay_data.csv        (delta, lnA_pre, se_lnA)  <-- used for WLS/OLS/Tobit
  - decay/output/decay_curve.csv       (delta_cont, lnA_pre_cont) <-- used to draw the line
  - decay/output/decay_data_raw.csv    (delta, lnA_pre_raw)      <-- replicates (SI)
"""
import os, yaml, math
import numpy as np
import pandas as pd

def _load_params():
    here = os.path.dirname(__file__)
    path = os.path.join(here, 'default_params.yml')
    with open(path, 'r', encoding='utf-8') as f:
        y = yaml.safe_load(f)
    p = y['decay'] if isinstance(y, dict) and 'decay' in y else y
    return {
        'seed':        int(p.get('seed', 52)),
        'A0':          float(p.get('A0', 1.0)),
        # accept tau_fut as an alias to match manuscript notation
        'tau_f':       float(p.get('tau_fut', p.get('tau_f', 0.02))),
        'noise_log':   float(p.get('noise_log', 0.10)),
        'delta_start': float(p.get('delta_start', 0.0)),
        'delta_end':   float(p.get('delta_end', 0.02)),
        'delta_step':  float(p.get('delta_step', 0.005)),
        'n_cont':      int(p.get('n_cont', 300)),
        # NEW: number of replicate trials per discrete delay (for SEs)
        'n_rep':       int(p.get('n_rep', 40)),
        # Optional: apply detection floor in log-domain (default off to preserve Box-2a numbers)
        'apply_censoring': bool(p.get('apply_censoring', False)),
        'A_min':       float(p.get('A_min', p.get('epsilon_detection', 0.01))),
    }

env_seed = os.getenv("CRI_SEED", None)
if env_seed is not None:
    try:
        p["seed"] = int(env_seed)
    except Exception:
        pass

def main():
    p = _load_params() 
    print("simulate_decay.py VERSION: CRI v0.3-SIM",
      f"(n_rep={p['n_rep']}, noise_log={p['noise_log']}, "
      f"delta=[{p['delta_start']},{p['delta_end']}] step={p['delta_step']})")
    
    rng = np.random.default_rng(p['seed'])

    here = os.path.dirname(__file__)
    outd = os.path.join(here, 'output')
    os.makedirs(outd, exist_ok=True)

    # Discrete delays for orange points (e.g., 0, 5, 10, 15, 20 ms)
    deltas = np.arange(p['delta_start'], p['delta_end'] + 1e-12, p['delta_step'])
    # True mean in log domain
    lnA_true = np.log(p['A0']) - deltas / p['tau_f']

    # Simulate replicates in log-domain (homoscedastic log-noise)
    raw_rows = []
    c = float(np.log(p["A_min"])) if p.get("apply_censoring", False) else None  
    for d, mu in zip(deltas, lnA_true):
        y = mu + rng.normal(0.0, p['noise_log'], size=p['n_rep'])
        if c is not None:
            y = np.maximum(y, c)      
        for v in y:
            raw_rows.append({'delta': float(d), 'lnA_pre_raw': float(v)})
    pd.DataFrame(raw_rows).to_csv(os.path.join(outd, 'decay_data_raw.csv'), index=False)

    # Aggregate to mean ± SE per delay (this is what WLS will use)
    df_raw = pd.DataFrame(raw_rows)
    agg = df_raw.groupby('delta', as_index=False).agg(
        lnA_pre=('lnA_pre_raw', 'mean'),
        sd=('lnA_pre_raw', 'std'),
        n=('lnA_pre_raw', 'size')
    )
    # Standard error of the mean in log-domain
    agg['se_lnA'] = agg['sd'] / np.sqrt(agg['n'])
    agg[['delta', 'lnA_pre', 'se_lnA']].to_csv(os.path.join(outd, 'decay_data.csv'), index=False)

    # Dense noiseless curve for drawing the central line/band background
    x_cont = np.linspace(p['delta_start'], p['delta_end'], p['n_cont'])
    lnA_cont = np.log(p['A0']) - x_cont / p['tau_f']
    pd.DataFrame({'delta_cont': x_cont, 'lnA_pre_cont': lnA_cont}).to_csv(
        os.path.join(outd, 'decay_curve.csv'), index=False
    )

    print(f"Saved discrete data with SEs → {os.path.join(outd, 'decay_data.csv')}")
    print(f"Saved dense curve → {os.path.join(outd, 'decay_curve.csv')}")

if __name__ == '__main__':
    main()

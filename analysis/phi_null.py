"""Null model for the forward weight fraction: shuffle off-diagonal entries of each trained S, rerun the same FAS."""
import os, re, sys, numpy as np, torch, pandas as pd
# usage: python phi_null.py <lattice_pruning.py> <results folder with {task}_seed{seed}_base.pt>
src = open(sys.argv[1] if len(sys.argv) > 1 else 'lattice_pruning.py').read()
OUT = sys.argv[2] if len(sys.argv) > 2 else 'lattice_pruning_results_v2'
ns = {}
exec(src[src.index('def greedy_fas_order'):src.index('def blocktri_allowed')], {'np': np}, ns)
fas, fwd = ns['greedy_fas_order'], ns['forward_weight_fraction']
files = {(t, sd): f'{t}_seed{sd}_base.pt' for t in ['mnist_rows', 'mnist_seq'] for sd in [0, 1, 2]}
H = 128; rows = []
for (task, seed), f in files.items():
    W = torch.load(os.path.join(OUT, f), map_location='cpu')['lstm.weight_hh_l0']
    S = W.abs().reshape(4, H, H).sum(0).numpy()
    phi = fwd(S, fas(S))
    rng = np.random.default_rng(1000 + seed)
    off = ~np.eye(H, dtype=bool)
    nulls = []
    for r in range(20):
        Sn = S.copy(); v = Sn[off].copy(); rng.shuffle(v); Sn[off] = v
        nulls.append(fwd(Sn, fas(Sn)))
    nulls = np.array(nulls)
    rows.append(dict(task=task, seed=seed, phi=phi, null_mean=nulls.mean(), null_sd=nulls.std(ddof=1),
                     null_max=nulls.max(), z=(phi - nulls.mean()) / nulls.std(ddof=1)))
df = pd.DataFrame(rows); print(df.round(4).to_string()); df.to_csv('phi_null.csv', index=False)

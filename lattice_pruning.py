# %% [markdown]
# # Order-theoretic pruning of recurrent networks: audit and revised experiments
#
# Companion code for *Lattice-Based Pruning in Recurrent Neural Networks via Poset Modeling* (revision).
#
# **Version 2.** Changes from version 1:
#
# 1. **Equal-budget control `baseline_ft`.** The unpruned baseline gets the same fine-tuning epochs as every pruned model, so "pruned beats baseline" can no longer come from extra training.
# 2. **Longer fine-tuning.** `FT_EPOCHS = 3` in the full run.
# 3. **Train-from-start comparison (Part D2).** Each mask is held fixed for a full training run from the seed's initial weights: random acyclic vs random unconstrained recurrence, plus lottery-ticket-style rewinding of masks found on the trained network. This asks whether acyclic recurrence is a cheap structural prior.
# 4. **Reference deltas in Part E.** Every row is reported as a difference from its equal-budget reference.
#
# Results go to a new folder (`..._v2`), so version-1 results are never mixed in.
#
# **Version 3** (speed and scope; results stay comparable with version 2 and the run resumes in the same folder):
#
# 1. **Denormal flush.** On CPU, networks with zeroed recurrent columns ran 3–5× slower on `mnist_seq`. Gradients flowing back through 112 forget-gate products become subnormal floats, which CPUs handle very slowly. `torch.set_flush_denormal(True)` removes this. It changes only values below about 1e-38, so results are unaffected in practice.
# 2. **Duplicate masks are skipped.** At $H \ge 8$ the three `paper_continuous_tau*` masks are identical, so the mask is trained once and the row copied.
# 3. **Per-task settings (`TASK_OVERRIDES`).** `mnist_seq` runs fewer sparsities and drops the lottery-ticket reference by default; edit the overrides to change this.
# 4. **Colab support.** Set `DRIVE_DIR` to keep results on Google Drive, so a disconnect or runtime change loses nothing. The device is checked, with a warning when running on CPU.
#
# The notebook has five parts. Parts A–C run in seconds and need no data; Part D is the main experiment.
#
# | Part | What it does | Runtime |
# |---|---|---|
# | **A** | Checks the order-theory claims on random finite posets: Birkhoff (one meet-irreducible ideal per element of $P$), and what the manuscript's own criterion $\{t : {\downarrow}t\setminus\{t\}\ \text{meet-irreducible}\}$ actually selects | seconds |
# | **B** | Audits the original pruning rules (random/binary adjacency, centrality $1/\sum_i A_{ij}$) **without training**: how many columns they keep for each hidden size and $\tau$ | seconds |
# | **C** | Verifies the block-triangular stability proposition on a Cohen–Grossberg network: after order-consistent pruning, the Jacobian spectrum is the union of the diagonal-block spectra | seconds |
# | **D1** | Post-hoc pruning: one trained LSTM per seed; every pruning method applied **to that model's weights** at matched sparsity; one-shot and fine-tuned scores; equal-budget and no-recurrence controls; the original rules reproduced; SCC/condensation structure of every mask | ~25 min per seed for row-MNIST on a laptop CPU |
# | **D2** | Train-from-start: masks held fixed for a full training run from the initial weights (random acyclic vs random, and rewound magnitude/FAS masks) | ~20 min per seed for row-MNIST on a laptop CPU |
# | **E** | Tables and figures from the saved results | seconds |
#
# **Use a GPU for the full run.** On Colab: Runtime → Change runtime type → T4 GPU. On Colab's CPU, `mnist_seq` alone needs more than a day.
#
# QUICK mode runs everything above in roughly 10 minutes. The sequential task (`mnist_seq`, 112 steps) takes about 4× as long as row-wise MNIST; a GPU (Colab) cuts all times substantially.
#
# **Install** (once, in a terminal or a notebook cell):
# ```
# pip install torch torchvision networkx numpy scipy pandas matplotlib
# ```
# MNIST is downloaded automatically by `torchvision` into `DATA_DIR` on first run.
#
# **Resuming.** Every result row is appended to a CSV as soon as it is computed, and baseline models are checkpointed. Re-running Part D skips anything already done, so an interrupted run can simply be restarted. Delete `OUT_DIR` to start fresh.

# %%
# ===================== CONFIGURATION =====================
QUICK = True   # True: short smoke test (1 seed, 2 sparsities, 10k training images). Set False for the paper run.
RUN_FROM_START = True   # Part D2: train masked networks from the initial weights (adds ~20 min per seed per task on CPU)

CFG = dict(
    SEEDS       = [0, 1, 2],
    TASKS       = ['mnist_rows', 'mnist_seq'],   # rows: 28 steps x 28 px; seq: 112 steps x 7 px (needs recurrence more)
    HIDDEN      = 128,
    BATCH       = 128,
    LR          = 1e-3,
    BASE_EPOCHS = 8,          # training of the unpruned baseline (and of the no-recurrence control)
    FT_EPOCHS   = 3,          # fine-tuning after pruning, with the mask held fixed (baseline_ft gets the same)
    SPARSITIES  = [0.25, 0.5, 0.75, 0.9],   # fraction of hidden-to-hidden weights removed
    START_SPARSITIES = [0.5, 0.75, 0.9],     # sparsities for Part D2 (train from the initial weights, BASE_EPOCHS)
    EDGE_KEEP   = 'auto',     # fraction of off-diagonal neuron->neuron edges kept when building the thresholded
                              # dependency graph used by ALL graph/order scores (degree, PageRank, kappa).
                              # 'auto' = the value (per trained model) at which kappa is least degenerate,
                              # i.e. the fewest neurons tie; a number (e.g. 0.05) fixes it.
    COLLAPSE_GRID = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],  # edge-keep fractions for the SCC-collapse diagnostic
    PAPER_TAUS  = [0.2, 0.5, 0.7],
    N_VAL       = 5000,       # validation split carved from the 60k training images
    TRAIN_SUBSET= None,       # None = all remaining training images
    DATA_DIR    = './data',
    OUT_DIR     = './lattice_pruning_results_v2',
    DRIVE_DIR   = None,       # Colab: e.g. '/content/drive/MyDrive/lattice_pruning_results_v2'. When set, Google Drive
                              # is mounted and this folder is used as OUT_DIR (survives disconnects and runtime changes).
    NUM_THREADS = None,       # e.g. 4 to cap CPU threads; None = PyTorch default
    # Per-task settings that replace the defaults above for that task only.
    TASK_OVERRIDES = {
        'mnist_seq': dict(SPARSITIES=[0.5, 0.9], START_SPARSITIES=[0.5, 0.9],
                          START_METHODS=['start_pair_random', 'start_acyclic_random',
                                         'rewind_pair_magnitude', 'rewind_acyclic_fas']),
    },
    METHODS = ['col_random', 'col_magnitude', 'col_degree', 'col_pagerank', 'col_order_kappa',
               'pair_random', 'pair_magnitude', 'pair_blocktri_fas', 'pair_blocktri_randorder', 'elem_magnitude'],
    START_METHODS = ['start_pair_random', 'start_acyclic_random',
                     'rewind_pair_magnitude', 'rewind_acyclic_fas', 'rewind_elem_magnitude'],
)
if QUICK:
    CFG.update(SEEDS=[0], TASKS=['mnist_rows'], BASE_EPOCHS=3, FT_EPOCHS=1, SPARSITIES=[0.5, 0.9],
               START_SPARSITIES=[0.5, 0.9], TRAIN_SUBSET=10000, OUT_DIR='./lattice_pruning_results_v2_quick',
               DRIVE_DIR=None)

def task_cfg(task):
    """CFG with this task's overrides applied."""
    c = dict(CFG)
    c.update(CFG['TASK_OVERRIDES'].get(task, {}))
    return c
print({k: v for k, v in CFG.items()})

# %%
import os, time, math, json, random, warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from IPython.display import display
except ImportError:
    display = print

if CFG['DRIVE_DIR']:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        print('DRIVE_DIR is set but this is not Colab; using it as a normal folder.')
    CFG['OUT_DIR'] = CFG['DRIVE_DIR']
os.makedirs(CFG['OUT_DIR'], exist_ok=True)
print('Results folder:', os.path.abspath(CFG['OUT_DIR']))

torch.set_flush_denormal(True)   # see version-3 note: avoids a 3-5x CPU slowdown from subnormal gradients
if CFG['NUM_THREADS']:
    torch.set_num_threads(CFG['NUM_THREADS'])

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print('torch', torch.__version__, '| networkx', nx.__version__, '| device:', DEVICE)
if DEVICE.type == 'cpu' and not QUICK:
    print('WARNING: running the full experiment on CPU. On Colab, switch to a GPU runtime '
          '(Runtime -> Change runtime type -> T4 GPU); mnist_seq takes more than a day on Colab CPU.')

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# %% [markdown]
# ## Utilities: order theory, dependency graphs, scores and masks
#
# **Convention.** For an LSTM, `weight_hh_l0` has shape $(4H, H)$, gate blocks $(i, f, g, o)$. Column $j$ holds neuron $j$'s outgoing recurrent weights. The **neuron dependency graph** has an edge $j \to i$ with strength
# $$S_{ij} = \sum_{\text{gates } g} |W^{(g)}_{hh}[i, j]|.$$
# Sparsity is always the fraction of zero entries in $W_{hh}$ (input and output weights are never pruned), so it is comparable across methods.
#
# **Order score.** Thresholding $S$ (keep the top `EDGE_KEEP` fraction of off-diagonal edges) gives a directed graph. Degree, PageRank and $\kappa$ are all computed on this same graph. Quotienting by strongly connected components (SCCs) gives a genuine poset: the condensation. For neuron $t$ in SCC $C(t)$,
# $$\kappa(t) = \frac{|{\downarrow}C(t)| + |{\uparrow}C(t)| - |C(t)| - 1}{H - 1},$$
# counted in neurons. This is the fraction of other neurons that are upstream, downstream, or in the same module. $\kappa(t) = 1$ exactly when $t$ is comparable to everything, which is what the manuscript's ${\downarrow}t\setminus\{t\}$ criterion actually selects (Part A).

# %%
# ---------- finite posets and ideal lattices (Part A) ----------
def reflexive_transitive_closure(adj):
    """adj[u, v] = True for an edge u -> v. Returns leq[u, v] = (u <= v)."""
    n = adj.shape[0]
    leq = adj.copy() | np.eye(n, dtype=bool)
    for k in range(n):
        leq |= leq[:, [k]] & leq[[k], :]
    return leq

def enumerate_ideals(leq):
    """All order ideals (down-sets) of a finite poset, as bitmasks. Fine for n <= ~16."""
    n = leq.shape[0]
    below = [sum(1 << u for u in range(n) if leq[u, v]) for v in range(n)]   # bitmask of the down-set of v
    ideals = []
    for I in range(1 << n):
        if all(((I >> v) & 1) == 0 or (below[v] & ~I) == 0 for v in range(n)):
            ideals.append(I)
    return ideals

def meet_irreducible_ideals(ideals, n):
    """In a finite lattice, x is meet-irreducible iff it has exactly one upper cover.
    In O(P) the upper covers of I are exactly I + {v} for v minimal in P \\ I."""
    S = set(ideals); full = (1 << n) - 1
    mi = []
    for I in ideals:
        if I == full:
            continue
        covers = [I | (1 << v) for v in range(n) if not (I >> v) & 1 and (I | (1 << v)) in S]
        if len(covers) == 1:
            mi.append(I)
    return mi

def bits(I, n):
    return frozenset(v for v in range(n) if (I >> v) & 1)

def random_dag(n, p, rng):
    order = rng.permutation(n)
    adj = np.zeros((n, n), bool)
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < p:
                adj[order[a], order[b]] = True
    return adj

# ---------- neuron dependency graph from an LSTM ----------
def neuron_strength(W_hh, H):
    """S[i, j] = sum over the 4 gates of |W_hh[g*H + i, j]|: strength of the dependency j -> i."""
    W = W_hh.detach().abs().reshape(4, H, H).sum(0)
    return W.cpu().numpy()

def threshold_graph(S, keep_frac):
    """Keep the top keep_frac of off-diagonal entries of S. A[i, j] = True means edge j -> i."""
    H = S.shape[0]
    off = ~np.eye(H, dtype=bool)
    vals = S[off]
    k = max(1, int(round(keep_frac * vals.size)))
    thr = np.partition(vals, vals.size - k)[vals.size - k]
    return (S >= thr) & off

def graph_from_A(A):
    H = A.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(H))
    ii, jj = np.nonzero(A)
    G.add_edges_from(zip(jj.tolist(), ii.tolist()))    # j -> i
    return G

def order_kappa(A):
    """Comparability fraction kappa(t) in the condensation poset of the graph A (see markdown above)."""
    H = A.shape[0]
    G = graph_from_A(A)
    C = nx.condensation(G)
    size = {c: len(C.nodes[c]['members']) for c in C.nodes}
    kappa = np.zeros(H)
    for c in C.nodes:
        up = sum(size[a] for a in nx.ancestors(C, c))
        down = sum(size[d] for d in nx.descendants(C, c))
        val = (up + down + size[c] - 1) / (H - 1)
        for t in C.nodes[c]['members']:
            kappa[t] = val
    return kappa

def kappa_tie_frac(kappa):
    return pd.Series(np.round(kappa, 9)).value_counts().iloc[0] / len(kappa)

def auto_edge_keep(S, grid=None):
    """Pick the edge-keep fraction at which the order score kappa is most informative (smallest largest-tie group).
    This gives the order score its best chance; degree and PageRank are computed on the same graph."""
    grid = np.geomspace(0.001, 0.2, 25) if grid is None else grid
    best = None
    for kf in grid:
        tie = kappa_tie_frac(order_kappa(threshold_graph(S, kf)))
        if best is None or tie < best[1] - 1e-12 or (abs(tie - best[1]) < 1e-12 and kf > best[0]):
            best = (float(kf), tie)
    return best

def scc_stats(A):
    G = graph_from_A(A)
    sccs = list(nx.strongly_connected_components(G))
    C = nx.condensation(G, scc=sccs)
    H = A.shape[0]
    return dict(n_scc=len(sccs),
                largest_scc_frac=max(len(s) for s in sccs) / H,
                height=nx.dag_longest_path_length(C) + 1)

def condensation_height_width(A):
    """Height (Mirsky: longest chain) and width (Dilworth: largest antichain = minimum chain cover)
    of the condensation poset, in units of SCC modules."""
    G = graph_from_A(A)
    C = nx.condensation(G)
    n = C.number_of_nodes()
    height = nx.dag_longest_path_length(C) + 1 if n else 0
    if n <= 1:
        return height, n
    TC = nx.transitive_closure_dag(C)
    B = nx.Graph()
    left = [('L', u) for u in C.nodes]
    B.add_nodes_from(left, bipartite=0)
    B.add_nodes_from([('R', v) for v in C.nodes], bipartite=1)
    B.add_edges_from((('L', u), ('R', v)) for u, v in TC.edges)
    M = nx.bipartite.hopcroft_karp_matching(B, top_nodes=left)
    return height, n - len(M) // 2

# ---------- feedback-arc-set ordering (order-consistent pruning) ----------
def greedy_fas_order(S, polish=True):
    """Eades-Lin-Smyth style greedy ordering for a weighted digraph, with S[i, j] = weight of j -> i.
    Returns a neuron order (upstream first) that keeps as much weight as possible on forward edges."""
    W = S.astype(float).copy()
    np.fill_diagonal(W, 0.0)
    H = W.shape[0]
    rem = np.ones(H, bool)
    out_w = W.sum(0)          # out_w[u] = total weight of edges u -> (remaining)
    in_w = W.sum(1)           # in_w[u]  = total weight of edges (remaining) -> u
    order = []
    for _ in range(H):
        cand = np.nonzero(rem)[0]
        v = cand[np.argmax(out_w[cand] - in_w[cand])]
        order.append(v); rem[v] = False
        out_w -= W[v, :]      # edges u -> v disappear
        in_w -= W[:, v]       # edges v -> u disappear
    order = np.array(order)
    if polish:                # adjacent-swap local search
        changed = True
        while changed:
            changed = False
            for p in range(H - 1):
                a, b = order[p], order[p + 1]
                if W[a, b] > W[b, a]:          # edge b -> a is backward and heavier than a -> b
                    order[p], order[p + 1] = b, a
                    changed = True
    return order

def forward_weight_fraction(S, order):
    H = S.shape[0]
    pos = np.empty(H, int); pos[order] = np.arange(H)
    W = S.copy(); np.fill_diagonal(W, 0)
    fwd = pos[None, :] < pos[:, None]      # j before i  => edge j -> i is forward
    return W[fwd].sum() / W.sum()

def blocktri_allowed(order, k, H):
    """Allowed[i, j] = edge j -> i permitted under a block-lower-triangular structure with k contiguous
    blocks of the given order. k = H gives a strict order plus self-loops (no recurrent cycles between neurons)."""
    pos = np.empty(H, int); pos[order] = np.arange(H)
    blk = (pos * k) // H
    return blk[None, :] <= blk[:, None]

def blocks_for_sparsity(s, H):
    """Number of equal blocks k whose block-triangular pattern has sparsity (k-1)/(2k) <= s.
    For s >= ~0.5 the full order (k = H) is used and the remainder is filled by magnitude."""
    if s >= (H - 1) / (2 * H):
        return H
    return max(1, int(math.floor(1.0 / (1.0 - 2.0 * s) + 1e-9)))

# ---------- masks (all returned as float tensors of shape (4H, H)) ----------
def mask_columns(scores, s, H, rng):
    """Prune the round(s*H) columns with the lowest score. Ties broken at random."""
    n_prune = int(round(s * H))
    jitter = rng.random(H) * 1e-9 * (np.abs(scores).max() + 1e-12)
    idx = np.argsort(scores + jitter)[:n_prune]
    m = torch.ones(4 * H, H)
    m[:, torch.as_tensor(idx, dtype=torch.long)] = 0.0
    return m

def mask_pairs(keep_HH):
    return torch.as_tensor(keep_HH, dtype=torch.float32).repeat(4, 1)

def topk_pairs(S, s, allowed=None):
    """Keep the highest-S neuron pairs (all 4 gate entries together) so that a fraction s of pairs is removed.
    If 'allowed' is given, only allowed pairs may be kept."""
    H = S.shape[0]
    n = H * H
    n_keep = n - int(round(s * n))
    score = S.astype(float).copy()
    if allowed is not None:
        score[~allowed] = -np.inf
        n_keep = min(n_keep, int(allowed.sum()))
    flat = score.ravel()
    keep = np.zeros(n, bool)
    if n_keep > 0:
        keep[np.argpartition(-flat, n_keep - 1)[:n_keep]] = True
    return keep.reshape(H, H)

def mask_elements_magnitude(W_hh, s):
    W = W_hh.detach().abs().cpu().flatten()
    n_keep = W.numel() - int(round(s * W.numel()))
    m = torch.zeros_like(W)
    if n_keep > 0:
        m[torch.topk(W, n_keep).indices] = 1.0
    return m.reshape(W_hh.shape)

def mask_to_neuron_graph(mask, H):
    return (mask.reshape(4, H, H).amax(0) > 0).cpu().numpy() & ~np.eye(H, dtype=bool)

def sparsity_of(mask):
    return 1.0 - mask.float().mean().item()

# ---------- the manuscript's original rules, reproduced as described in Algorithms 3-4 ----------
def paper_continuous_keep(H, tau, seed, eps=1e-8):
    """A ~ U(0.5, 2.0), centrality c_j = 1/(sum_i A_ij + eps), keep column j iff c_j >= tau.
    Note: A is drawn independently of the network's weights."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.5, 2.0, size=(H, H))
    c = 1.0 / (A.sum(0) + eps)
    return c >= tau

def paper_binary_keep(H):
    """'First half of the neurons designated meet-irreducible': keep columns 0..H/2-1 (weight-independent)."""
    keep = np.zeros(H, bool); keep[: H // 2] = True
    return keep

def mask_from_column_keep(keep, H):
    m = torch.ones(4 * H, H)
    m[:, torch.as_tensor(~keep)] = 0.0
    return m

# %% [markdown]
# ## Part A: what the lattice criteria actually select (no training)
#
# Three claims are checked by brute force on random posets. Ideal lattices are enumerated by bitmask, so $|P| \le 12$.
#
# 1. **Birkhoff.** The meet-irreducible elements of $\mathcal O(P)$ are exactly the ideals $P\setminus{\uparrow}p$, one per $p \in P$. "Retain the meet-irreducible elements" therefore retains everything.
# 2. **The manuscript's criterion.** $\{t : {\downarrow}t\setminus\{t\}$ is meet-irreducible$\}$ equals $\{t : t$ is comparable to every element of $P\}$.
# 3. **Degeneracy.** In a fully connected recurrent layer unrolled in time, no element is comparable to everything, so $M = \varnothing$. On a chain, every element is critical.

# %%
rng = np.random.default_rng(0)
n_checks, ok_birkhoff, ok_criterion = 300, 0, 0
for _ in range(n_checks):
    n = int(rng.integers(3, 12))
    adj = random_dag(n, float(rng.uniform(0.1, 0.6)), rng)
    leq = reflexive_transitive_closure(adj)
    ideals = enumerate_ideals(leq)
    full = frozenset(range(n))
    mi = {bits(I, n) for I in meet_irreducible_ideals(ideals, n)}
    up = [frozenset(v for v in range(n) if leq[p, v]) for p in range(n)]
    birkhoff = {full - up[p] for p in range(n)}
    ok_birkhoff += (mi == birkhoff and len(mi) == n)
    paper_M = {t for t in range(n) if frozenset(u for u in range(n) if leq[u, t] and u != t) in mi}
    comparable_to_all = {t for t in range(n) if all(leq[t, u] or leq[u, t] for u in range(n))}
    ok_criterion += (paper_M == comparable_to_all)
print(f'Birkhoff (meet-irreducibles = {{P minus up(p)}}, exactly |P| of them): {ok_birkhoff}/{n_checks} random posets')
print(f"Manuscript criterion == 'comparable to every element':        {ok_criterion}/{n_checks} random posets")

def paper_critical(adj):
    n = adj.shape[0]
    leq = reflexive_transitive_closure(adj)
    mi = {bits(I, n) for I in meet_irreducible_ideals(enumerate_ideals(leq), n)}
    return sorted(t for t in range(n) if frozenset(u for u in range(n) if leq[u, t] and u != t) in mi)

# Figure 1 of the manuscript: a < c, b < c  (nodes 0=a, 1=b, 2=c)
adj = np.zeros((3, 3), bool); adj[0, 2] = adj[1, 2] = True
print("\nFigure 1 toy poset {a<c, b<c}: manuscript criterion selects", [ 'abc'[t] for t in paper_critical(adj)],
      "(the caption marks a and b as critical)")

# Figure 2 of the manuscript: 1 -> 2 -> 3 and 1 -> 3 (a chain)
adj = np.zeros((3, 3), bool); adj[0, 1] = adj[1, 2] = adj[0, 2] = True
print("Figure 2 chain 1->2->3:            manuscript criterion selects", [t + 1 for t in paper_critical(adj)], "(everything)")

# Dense recurrent layer, n neurons unrolled over T steps: (i,t) -> (j,t+1) for all i, j
n_neu, T = 3, 3
idx = lambda i, t: t * n_neu + i
adj = np.zeros((n_neu * T, n_neu * T), bool)
for t in range(T - 1):
    for i in range(n_neu):
        for j in range(n_neu):
            adj[idx(i, t), idx(j, t + 1)] = True
print(f"Dense recurrent layer ({n_neu} neurons x {T} steps): manuscript criterion selects", paper_critical(adj), "(nothing)")

# %% [markdown]
# ## Part B: audit of the original pruning rules (no training)
#
# The original continuous rule draws $A \sim \mathcal U(0.5, 2)$ **independently of the trained weights** and keeps column $j$ iff $1/\sum_i A_{ij} \ge \tau$. Since $\sum_i A_{ij} \approx 1.25H$, a column survives only if $H$ is tiny. The table shows the fraction of recurrent columns kept for each hidden size and $\tau$.
#
# Compare this with your original code's hidden size and reported sparsity (0.9847–0.9943). If the table shows 0% kept at your $H$ but the paper reports less than 100% sparsity, the sparsity in the original code was computed over a different parameter set.

# %%
rows = []
for H_ in [4, 8, 16, 32, 64, 128, 256]:
    for tau in CFG['PAPER_TAUS']:
        keep = paper_continuous_keep(H_, tau, seed=0)
        rows.append(dict(H=H_, tau=tau, columns_kept_pct=100 * keep.mean(),
                         W_hh_sparsity_pct=100 * (1 - keep.mean())))
audit = pd.DataFrame(rows).pivot(index='H', columns='tau', values='columns_kept_pct')
audit.columns = [f'tau={t}: % columns kept' for t in audit.columns]
display(audit)
print('Binary rule keeps exactly the first half of the columns at every tau -> W_hh sparsity 50%, independent of the weights.')

# %% [markdown]
# ## Part C: block-triangular stability in a Cohen–Grossberg network (no training)
#
# Take $\dot x_i = a_i(x_i)\big[-b_i(x_i) + \sum_j w_{ij} f_j(x_j) + I_i\big]$ with $a_i(x) = 1 + \tfrac12\tanh x > 0$, $b_i(x) = x$ and $f = \tanh$. At an equilibrium the bracket vanishes, so
# $$J = \mathrm{diag}\big(a_i(x^*)\big)\big(-I + W\,\mathrm{diag}(f'(x^*))\big).$$
# If pruning makes $W$ block-lower-triangular in some neuron order, then $J$ is block-lower-triangular in that order. Hence $\mathrm{spec}(J) = \bigcup_k \mathrm{spec}(J_{kk})$, and local stability holds iff every diagonal block is stable. The cell checks this numerically after FAS-ordered pruning into $k$ blocks.

# %%
from scipy.optimize import fsolve, linear_sum_assignment

def cg_check(n=48, k=6, gain=1.5, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, gain / np.sqrt(n), (n, n))
    I_ext = rng.normal(0, 0.5, n)
    order = greedy_fas_order(np.abs(W))
    allowed = blocktri_allowed(order, k, n)
    Wp = W * allowed
    a = lambda x: 1 + 0.5 * np.tanh(x)
    rhs = lambda x: a(x) * (-x + Wp @ np.tanh(x) + I_ext)
    x_star = fsolve(rhs, np.zeros(n), xtol=1e-13)
    J = np.diag(a(x_star)) @ (-np.eye(n) + Wp @ np.diag(1 - np.tanh(x_star) ** 2))
    Jp = J[np.ix_(order, order)]
    pos = np.empty(n, int); pos[order] = np.arange(n)
    blk = (np.arange(n) * k) // n
    upper = np.abs(Jp[blk[:, None] < blk[None, :]]).max()
    eig_full = np.linalg.eigvals(Jp)
    eig_blocks = np.concatenate([np.linalg.eigvals(Jp[np.ix_(blk == b, blk == b)]) for b in range(k)])
    r, c = linear_sum_assignment(np.abs(eig_full[:, None] - eig_blocks[None, :]))
    print(f'n={n}, k={k} blocks | residual at equilibrium {np.abs(rhs(x_star)).max():.1e}')
    print(f'  max |entry| above the block diagonal (should be 0):        {upper:.1e}')
    print(f'  max eigenvalue mismatch, J vs union of diagonal blocks:     {np.abs(eig_full[r] - eig_blocks[c]).max():.1e}')
    print(f'  kept fraction of |W| mass after order-consistent pruning: {np.abs(Wp).sum() / np.abs(W).sum():.3f}')
    for b in range(k):
        ev = np.linalg.eigvals(Jp[np.ix_(blk == b, blk == b)])
        print(f'  block {b}: size {np.sum(blk == b):2d}, max Re(eig) = {ev.real.max():+.3f}')
    print(f'  whole network: max Re(eig) = {eig_full.real.max():+.3f}  (stable iff every block is)')

cg_check()

# %% [markdown]
# ## Part D: main experiment
#
# ### D1: post-hoc pruning of a trained network
#
# For each task and seed:
#
# 1. Train an unpruned LSTM baseline (`BASE_EPOCHS`). Then fine-tune a copy of it, unpruned, for `FT_EPOCHS`: this is **`baseline_ft`**, the equal-budget reference for every fine-tuned pruned model. Train a **no-recurrence control** from scratch with $W_{hh} \equiv 0$; it tells you what accuracy needs no recurrent weights at all.
# 2. On the **trained** baseline weights, build every mask at each target sparsity. Evaluate one-shot (no retraining), then fine-tune with the mask held fixed. Each method starts from the same baseline.
# 3. Record structure for every mask: number of SCCs, largest SCC, and the condensation's height and width.
#
# | Method | Granularity | Rule |
# |---|---|---|
# | `col_random` | column (neuron's outgoing recurrence) | random |
# | `col_magnitude` | column | L2 norm of the column |
# | `col_degree` | column | in+out degree in the thresholded dependency graph |
# | `col_pagerank` | column | PageRank of the reversed thresholded graph (influence) |
# | `col_order_kappa` | column | comparability fraction $\kappa$ in the condensation poset |
# | `pair_random` | neuron pair (4 gate entries) | random |
# | `pair_magnitude` | neuron pair | $S_{ij}$ |
# | `pair_blocktri_fas` | neuron pair | block-lower-triangular in the FAS order, then magnitude inside the allowed pattern |
# | `pair_blocktri_randorder` | neuron pair | same pattern in a **random** order (control: does the order matter?) |
# | `elem_magnitude` | single weight | $|w|$ (standard unstructured magnitude pruning) |
# | `paper_continuous_tau*`, `paper_binary` | column | the original rules (native sparsity, weight-independent) |
#
# The key comparisons:
#
# - `col_order_kappa` vs `col_degree` / `col_pagerank`: does the order score add anything over graph centrality?
# - `pair_blocktri_fas` vs `pair_blocktri_randorder`: does the dependency order matter?
# - `pair_blocktri_fas` vs `pair_magnitude`: what does acyclicity cost at matched sparsity?
# - Everything vs `baseline_ft` (equal training budget) and vs `no_recurrence`.
#
# ### D2: training from the initial weights with a fixed mask
#
# Each mask is applied to the seed's **initial** weights and held fixed for a full `BASE_EPOCHS` training run. The reference is the dense `baseline`, which has the same budget.
#
# | Method | Mask |
# |---|---|
# | `start_pair_random` | random neuron-pair mask |
# | `start_acyclic_random` | block-lower-triangular in a random neuron order, random fill. At initialisation all neuron orders are equivalent, so this is *the* acyclic architecture |
# | `rewind_pair_magnitude` | `pair_magnitude` mask from the trained baseline, weights rewound to initialisation (lottery-ticket style) |
# | `rewind_acyclic_fas` | `pair_blocktri_fas` mask from the trained baseline, rewound |
# | `rewind_elem_magnitude` | `elem_magnitude` mask, rewound (standard lottery-ticket reference) |
#
# The key comparisons:
#
# - `start_acyclic_random` vs `start_pair_random`: the cost of acyclicity as an architectural prior. The two differ only in the acyclic constraint.
# - `rewind_acyclic_fas` vs `start_acyclic_random`: does the order learned by the dense network help a fresh acyclic network?
# - `rewind_acyclic_fas` vs `rewind_pair_magnitude`: the cost of acyclicity when both masks use the trained weights.

# %%
# ---------- data ----------
_DATA_CACHE = {}

def load_mnist(cfg):
    if 'mnist' in _DATA_CACHE:
        return _DATA_CACHE['mnist']
    from torchvision import datasets
    tr = datasets.MNIST(cfg['DATA_DIR'], train=True, download=True)
    te = datasets.MNIST(cfg['DATA_DIR'], train=False, download=True)
    norm = lambda x: ((x.float() / 255.0) - 0.1307) / 0.3081
    Xtr, ytr = norm(tr.data), tr.targets.clone()
    Xte, yte = norm(te.data), te.targets.clone()
    g = torch.Generator().manual_seed(12345)            # fixed split, identical across seeds
    perm = torch.randperm(len(Xtr), generator=g)
    val_idx, tr_idx = perm[: cfg['N_VAL']], perm[cfg['N_VAL']:]
    if cfg['TRAIN_SUBSET']:
        tr_idx = tr_idx[: cfg['TRAIN_SUBSET']]
    out = dict(train=(Xtr[tr_idx], ytr[tr_idx]), val=(Xtr[val_idx], ytr[val_idx]), test=(Xte, yte))
    _DATA_CACHE['mnist'] = out
    return out

def as_sequence(X, task):
    """X: (N, 28, 28). mnist_rows -> (N, 28, 28); mnist_seq -> (N, 112, 7) (row-major chunks of 7 pixels)."""
    if task == 'mnist_rows':
        return X
    if task == 'mnist_seq':
        return X.reshape(X.shape[0], 112, 7)
    raise ValueError(task)

def get_task_data(task, cfg):
    d = load_mnist(cfg)
    return {k: (as_sequence(X, task), y) for k, (X, y) in d.items()}

# ---------- model, training, evaluation ----------
class SeqLSTM(nn.Module):
    def __init__(self, d_in, H, n_out=10):
        super().__init__()
        self.H = H
        self.lstm = nn.LSTM(d_in, H, batch_first=True)
        self.head = nn.Linear(H, n_out)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])
    @property
    def W_hh(self):
        return self.lstm.weight_hh_l0

@torch.no_grad()
def evaluate(model, X, y, bs=1000):
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(X), bs):
        xb, yb = X[i:i + bs].to(DEVICE), y[i:i + bs].to(DEVICE)
        out = model(xb)
        loss += F.cross_entropy(out, yb, reduction='sum').item()
        correct += (out.argmax(1) == yb).sum().item()
    return correct / len(X), loss / len(X)

def apply_mask(model, mask):
    with torch.no_grad():
        model.W_hh.mul_(mask.to(model.W_hh.device))

def train(model, data, epochs, cfg, mask=None, seed=0):
    """Adam training. With a mask, gradients and weights of pruned entries are zeroed at every step,
    so pruned connections stay pruned (no regrowth)."""
    if epochs <= 0:
        return
    Xtr, ytr = data['train']
    opt = torch.optim.Adam(model.parameters(), lr=cfg['LR'])
    m = mask.to(DEVICE) if mask is not None else None
    if m is not None:
        apply_mask(model, m)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), cfg['BATCH']):
            idx = perm[i:i + cfg['BATCH']]
            xb, yb = Xtr[idx].to(DEVICE), ytr[idx].to(DEVICE)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if m is not None:
                model.W_hh.grad.mul_(m)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if m is not None:
                apply_mask(model, m)

def new_model(task, cfg):
    d_in = 28 if task == 'mnist_rows' else 7
    return SeqLSTM(d_in, cfg['HIDDEN']).to(DEVICE)

# %%
# ---------- result bookkeeping (resumable) ----------
RESULTS_CSV = os.path.join(CFG['OUT_DIR'], 'results.csv')
COLLAPSE_CSV = os.path.join(CFG['OUT_DIR'], 'collapse_diagnostic.csv')

def load_done(path, keys):
    if not os.path.exists(path):
        return set(), []
    df = pd.read_csv(path)
    return set(map(tuple, df[keys].astype(str).values.tolist())), df.to_dict('records')

RESULT_COLUMNS = ['task', 'seed', 'method', 'target_sparsity', 'actual_sparsity', 'mask_sparsity',
                  'val_oneshot', 'test_oneshot', 'val_ft', 'test_ft', 'ft_epochs', 'ft_seconds',
                  'n_scc', 'largest_scc_frac', 'poset_height', 'poset_width',
                  'edge_keep', 'blocks_k', 'fas_forward_weight', 'kappa_modal_tie_frac', 'kappa_eq1_frac']
COLLAPSE_COLUMNS = ['task', 'seed', 'edge_keep', 'n_scc', 'largest_scc_frac', 'height',
                    'kappa_eq1_frac', 'kappa_modal_tie_frac']

def append_row(path, row, columns):
    pd.DataFrame([row]).reindex(columns=columns).to_csv(path, mode='a', header=not os.path.exists(path), index=False)

KEYS = ['task', 'seed', 'method', 'target_sparsity']
DONE, _ = load_done(RESULTS_CSV, KEYS)
print(f'{len(DONE)} result rows already on disk in {RESULTS_CSV}')

def build_masks(model, s, cfg, rng, edge_keep):
    """All matched-sparsity masks for one target sparsity s, computed from the trained weights."""
    H = cfg['HIDDEN']
    W = model.W_hh.detach().cpu()
    S = neuron_strength(W, H)
    A = threshold_graph(S, edge_keep)
    Gw = graph_from_A(A)
    for (u, v) in Gw.edges:
        Gw[u][v]['weight'] = float(S[v, u])
    col_l2 = W.reshape(4, H, H).pow(2).sum((0, 1)).sqrt().numpy()
    degree = A.sum(0) + A.sum(1)                         # out-degree (column) + in-degree (row)
    pr = nx.pagerank(Gw.reverse(copy=True), weight='weight')
    pagerank = np.array([pr[i] for i in range(H)])
    kappa = order_kappa(A)
    fas = greedy_fas_order(S)
    k = blocks_for_sparsity(s, H)
    allowed_fas = blocktri_allowed(fas, k, H)
    allowed_rnd = blocktri_allowed(rng.permutation(H), k, H)
    masks = {
        'col_random':      mask_columns(rng.random(H), s, H, rng),
        'col_magnitude':   mask_columns(col_l2, s, H, rng),
        'col_degree':      mask_columns(degree.astype(float), s, H, rng),
        'col_pagerank':    mask_columns(pagerank, s, H, rng),
        'col_order_kappa': mask_columns(kappa, s, H, rng),
        'pair_random':     mask_pairs(topk_pairs(rng.random((H, H)), s)),
        'pair_magnitude':  mask_pairs(topk_pairs(S, s)),
        'pair_blocktri_fas':       mask_pairs(topk_pairs(S, s, allowed_fas)),
        'pair_blocktri_randorder': mask_pairs(topk_pairs(S, s, allowed_rnd)),
        'elem_magnitude':  mask_elements_magnitude(W, s),
    }
    info = dict(edge_keep=edge_keep, blocks_k=k,
                fas_forward_weight=forward_weight_fraction(S, fas),
                kappa_modal_tie_frac=kappa_tie_frac(kappa),
                kappa_eq1_frac=float(np.mean(kappa > 1 - 1e-9)))
    return masks, info

def structure_of(mask, H):
    A = mask_to_neuron_graph(mask, H)
    st = scc_stats(A)
    h, w = condensation_height_width(A)
    return dict(n_scc=st['n_scc'], largest_scc_frac=st['largest_scc_frac'], poset_height=h, poset_width=w)

def train_from_start(init_state, task, data, mask, cfg, seed):
    """Apply a fixed mask to the initial weights and train for BASE_EPOCHS (same budget as the dense baseline)."""
    model = new_model(task, cfg)
    model.load_state_dict(init_state)
    t0 = time.time()
    train(model, data, cfg['BASE_EPOCHS'], cfg, mask=mask, seed=seed)
    v, _ = evaluate(model, *data['val']); t, _ = evaluate(model, *data['test'])
    actual = 1.0 - (model.W_hh.detach() != 0).float().mean().item()
    return dict(val_ft=v, test_ft=t, actual_sparsity=actual, mask_sparsity=sparsity_of(mask),
                ft_seconds=time.time() - t0, ft_epochs=cfg['BASE_EPOCHS'])

def eval_and_finetune(base_state, task, data, mask, cfg, seed):
    model = new_model(task, cfg)
    model.load_state_dict(base_state)
    apply_mask(model, mask)
    val1, _ = evaluate(model, *data['val']); test1, _ = evaluate(model, *data['test'])
    t0 = time.time()
    train(model, data, cfg['FT_EPOCHS'], cfg, mask=mask, seed=seed + 1000)
    val2, _ = evaluate(model, *data['val']); test2, _ = evaluate(model, *data['test'])
    actual = 1.0 - (model.W_hh.detach() != 0).float().mean().item()
    return dict(val_oneshot=val1, test_oneshot=test1, val_ft=val2, test_ft=test2,
                actual_sparsity=actual, mask_sparsity=sparsity_of(mask), ft_seconds=time.time() - t0,
                ft_epochs=cfg['FT_EPOCHS'])

# %%
H = CFG['HIDDEN']
for task in CFG['TASKS']:
    TC = task_cfg(task)
    data = get_task_data(task, CFG)
    print(f"\n=== task {task}: train {len(data['train'][0])}, val {len(data['val'][0])}, test {len(data['test'][0])}, "
          f"input {tuple(data['train'][0].shape[1:])} ===")
    for seed in CFG['SEEDS']:
        set_seed(seed)
        rng = np.random.default_rng(seed)
        ckpt = os.path.join(CFG['OUT_DIR'], f'{task}_seed{seed}_base.pt')

        # 1. baseline (initial weights are kept for Part D2)
        model = new_model(task, CFG)
        init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        else:
            t0 = time.time()
            train(model, data, CFG['BASE_EPOCHS'], CFG, seed=seed)
            torch.save(model.state_dict(), ckpt)
            print(f'[{task} s{seed}] baseline trained in {time.time() - t0:.0f}s')
        base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        key = (task, str(seed), 'baseline', '0.0')
        if key not in DONE:
            v, _ = evaluate(model, *data['val']); t, _ = evaluate(model, *data['test'])
            row = dict(task=task, seed=seed, method='baseline', target_sparsity=0.0, val_oneshot=v, test_oneshot=t,
                       val_ft=v, test_ft=t, actual_sparsity=0.0, mask_sparsity=0.0)
            row.update(structure_of(torch.ones(4 * H, H), H))
            append_row(RESULTS_CSV, row, RESULT_COLUMNS); DONE.add(key)
            print(f'[{task} s{seed}] baseline  val {v:.4f}  test {t:.4f}')

        # 1b. equal-budget control: the unpruned baseline fine-tuned for FT_EPOCHS, like every pruned model
        key = (task, str(seed), 'baseline_ft', '0.0')
        if key not in DONE:
            row = dict(task=task, seed=seed, method='baseline_ft', target_sparsity=0.0)
            row.update(eval_and_finetune(base_state, task, data, torch.ones(4 * H, H), CFG, seed))
            row.update(structure_of(torch.ones(4 * H, H), H))
            append_row(RESULTS_CSV, row, RESULT_COLUMNS); DONE.add(key)
            print(f"[{task} s{seed}] baseline_ft ({CFG['FT_EPOCHS']} extra epochs)  test {row['test_ft']:.4f}")

        # 2. no-recurrence control (W_hh = 0, trained from scratch with the same budget)
        key = (task, str(seed), 'no_recurrence', '1.0')
        if key not in DONE:
            set_seed(seed)
            m0 = new_model(task, CFG)
            zero = torch.zeros(4 * H, H)
            train(m0, data, CFG['BASE_EPOCHS'], CFG, mask=zero, seed=seed)
            v, _ = evaluate(m0, *data['val']); t, _ = evaluate(m0, *data['test'])
            append_row(RESULTS_CSV, dict(task=task, seed=seed, method='no_recurrence', target_sparsity=1.0,
                                         val_oneshot=v, test_oneshot=t, val_ft=v, test_ft=t,
                                         actual_sparsity=1.0, mask_sparsity=1.0, ft_epochs=CFG['BASE_EPOCHS']),
                       RESULT_COLUMNS)
            DONE.add(key)
            print(f'[{task} s{seed}] no-recurrence control  val {v:.4f}  test {t:.4f}')

        # 3. SCC-collapse diagnostic on the trained weights
        S = neuron_strength(model.W_hh, H)
        collapse_done = (os.path.exists(COLLAPSE_CSV) and
                         ((pd.read_csv(COLLAPSE_CSV).task == task) & (pd.read_csv(COLLAPSE_CSV).seed == seed)).any())
        if not collapse_done:
            for kf in CFG['COLLAPSE_GRID']:
                A = threshold_graph(S, kf)
                st = scc_stats(A)
                kap = order_kappa(A)
                append_row(COLLAPSE_CSV, dict(task=task, seed=seed, edge_keep=kf, **st,
                                              kappa_eq1_frac=float(np.mean(kap > 1 - 1e-9)),
                                              kappa_modal_tie_frac=kappa_tie_frac(kap)), COLLAPSE_COLUMNS)

        if CFG['EDGE_KEEP'] == 'auto':
            edge_keep, tie = auto_edge_keep(S)
            print(f'[{task} s{seed}] auto EDGE_KEEP = {edge_keep:.4f} (largest kappa tie group = {tie:.2f} of neurons)')
        else:
            edge_keep = float(CFG['EDGE_KEEP'])

        # 4. the manuscript's original rules (native sparsity, weight-independent)
        paper_masks = {'paper_binary': mask_from_column_keep(paper_binary_keep(H), H)}
        for tau in CFG['PAPER_TAUS']:
            paper_masks[f'paper_continuous_tau{tau}'] = mask_from_column_keep(paper_continuous_keep(H, tau, seed), H)
        computed = []   # (mask, result) pairs, so identical masks are trained only once
        if os.path.exists(RESULTS_CSV):
            prev = pd.read_csv(RESULTS_CSV)
            prev = prev[(prev.task == task) & (prev.seed == seed)].set_index('method')
            for name, mask in paper_masks.items():
                if name in prev.index:
                    computed.append((mask, prev.loc[name].to_dict()))
        for name, mask in paper_masks.items():
            key = (task, str(seed), name, f'{round(sparsity_of(mask), 4)}')
            if key in DONE:
                continue
            row = dict(task=task, seed=seed, method=name, target_sparsity=round(sparsity_of(mask), 4))
            same = [res for (m_, res) in computed if torch.equal(m_, mask)]
            if same:
                row.update({k: same[0][k] for k in ['val_oneshot', 'test_oneshot', 'val_ft', 'test_ft', 'actual_sparsity',
                                                    'mask_sparsity', 'ft_seconds', 'ft_epochs'] if k in same[0]})
                print(f'[{task} s{seed}] {name}: mask identical to an earlier one, result copied')
            else:
                row.update(eval_and_finetune(base_state, task, data, mask, CFG, seed))
                computed.append((mask, dict(row)))
            row.update(structure_of(mask, H))
            append_row(RESULTS_CSV, row, RESULT_COLUMNS); DONE.add(key)
            print(f"[{task} s{seed}] {name:28s} sparsity {row['actual_sparsity']:.3f}  "
                  f"one-shot {row['test_oneshot']:.4f}  fine-tuned {row['test_ft']:.4f}")

        # 5. matched-sparsity comparison
        for s in TC['SPARSITIES']:
            pending = [m for m in TC['METHODS'] if (task, str(seed), m, str(s)) not in DONE]
            if not pending:
                continue
            masks, info = build_masks(model, s, CFG, np.random.default_rng(seed * 100 + int(s * 100)), edge_keep)
            for name in pending:
                row = dict(task=task, seed=seed, method=name, target_sparsity=s, **info)
                row.update(eval_and_finetune(base_state, task, data, masks[name], CFG, seed))
                row.update(structure_of(masks[name], H))
                append_row(RESULTS_CSV, row, RESULT_COLUMNS); DONE.add((task, str(seed), name, str(s)))
                print(f"[{task} s{seed}] s={s:<4} {name:24s} one-shot {row['test_oneshot']:.4f}  "
                      f"fine-tuned {row['test_ft']:.4f}  SCCs {row['n_scc']:3d}  largest {row['largest_scc_frac']:.2f}  "
                      f"height {row['poset_height']:3d}  width {row['poset_width']:3d}")

        # 6. Part D2: train from the initial weights with the mask held fixed
        if not RUN_FROM_START:
            continue
        for s in TC['START_SPARSITIES']:
            pending = [m for m in TC['START_METHODS'] if (task, str(seed), m, str(s)) not in DONE]
            if not pending:
                continue
            # same rng seed as D1, so the rewound masks are exactly the D1 masks at this sparsity
            masks, info = build_masks(model, s, CFG, np.random.default_rng(seed * 100 + int(s * 100)), edge_keep)
            rs = np.random.default_rng(seed * 100 + int(s * 100) + 7)
            k = blocks_for_sparsity(s, H)
            if 'elem_magnitude' not in masks:
                masks['elem_magnitude'] = mask_elements_magnitude(model.W_hh.detach().cpu(), s)
            start_masks = {
                'start_pair_random':     mask_pairs(topk_pairs(rs.random((H, H)), s)),
                'start_acyclic_random':  mask_pairs(topk_pairs(rs.random((H, H)), s,
                                                               blocktri_allowed(rs.permutation(H), k, H))),
                'rewind_pair_magnitude': masks['pair_magnitude'],
                'rewind_acyclic_fas':    masks['pair_blocktri_fas'],
                'rewind_elem_magnitude': masks['elem_magnitude'],
            }
            for name in pending:
                row = dict(task=task, seed=seed, method=name, target_sparsity=s, **info)
                row.update(train_from_start(init_state, task, data, start_masks[name], CFG, seed))
                row.update(structure_of(start_masks[name], H))
                append_row(RESULTS_CSV, row, RESULT_COLUMNS); DONE.add((task, str(seed), name, str(s)))
                print(f"[{task} s{seed}] s={s:<4} {name:24s} trained from start: test {row['test_ft']:.4f}  "
                      f"SCCs {row['n_scc']:3d}  height {row['poset_height']:3d}  width {row['poset_width']:3d}")
print('\nDone. Results in', RESULTS_CSV)

# %% [markdown]
# ## Part E: results
#
# Means ± SD across seeds of **test** accuracy, both one-shot (pruned, no retraining) and after fine-tuning. Select any operating point on the **validation** columns; the test columns are for reporting only.
#
# `delta_vs_ref` is each row's fine-tuned test accuracy minus its equal-budget reference from the same seed:
#
# - post-hoc pruned models (D1) are compared with `baseline_ft`;
# - models trained from the initial weights (D2) and `no_recurrence` are compared with `baseline`.

# %%
df = pd.read_csv(RESULTS_CSV)
# keep only the sparsities and methods configured for each task (rows from earlier settings stay on disk)
def _configured(r):
    tc = task_cfg(r.task)
    if r.method.startswith(('start_', 'rewind_')):
        return r.method in tc['START_METHODS'] and r.target_sparsity in tc['START_SPARSITIES']
    if r.method in tc['METHODS']:
        return r.target_sparsity in tc['SPARSITIES']
    return True
df = df[df.apply(_configured, axis=1) & df.task.isin(CFG['TASKS'])].copy()
FROM_START = ('start_', 'rewind_')
df['phase'] = np.where(df.method.str.startswith(FROM_START) | (df.method == 'no_recurrence'), 'D2/from-start',
                       np.where(df.method == 'baseline', 'reference', 'D1/post-hoc'))
ref_ft = df[df.method == 'baseline_ft'].set_index(['task', 'seed']).test_ft
ref_base = df[df.method == 'baseline'].set_index(['task', 'seed']).test_ft
idx = list(zip(df.task, df.seed))
df['ref'] = [ (ref_base if ph == 'D2/from-start' else ref_ft).get(i, np.nan) for ph, i in zip(df.phase, idx)]
df.loc[df.method.isin(['baseline', 'baseline_ft']), 'ref'] = np.nan
df['delta_vs_ref'] = df.test_ft - df.ref
agg = (df.groupby(['task', 'phase', 'method', 'target_sparsity'])
         .agg(n_seeds=('seed', 'nunique'), sparsity=('actual_sparsity', 'mean'),
              val_ft=('val_ft', 'mean'),
              test_oneshot=('test_oneshot', 'mean'), test_oneshot_sd=('test_oneshot', 'std'),
              test_ft=('test_ft', 'mean'), test_ft_sd=('test_ft', 'std'),
              delta_vs_ref=('delta_vs_ref', 'mean'), delta_vs_ref_sd=('delta_vs_ref', 'std'),
              n_scc=('n_scc', 'mean'), largest_scc=('largest_scc_frac', 'mean'),
              height=('poset_height', 'mean'), width=('poset_width', 'mean'))
         .reset_index()
         .sort_values(['task', 'phase', 'target_sparsity', 'test_ft'], ascending=[True, True, True, False]))
pd.set_option('display.width', 200, 'display.max_rows', 200)
display(agg.round(4))
agg.to_csv(os.path.join(CFG['OUT_DIR'], 'summary.csv'), index=False)

# %%
STYLE = {
    'col_random': ('#999999', '--', 'o'), 'col_magnitude': ('#1f77b4', '-', 'o'), 'col_degree': ('#2ca02c', '-', 's'),
    'col_pagerank': ('#17becf', '-', '^'), 'col_order_kappa': ('#d62728', '-', 'D'),
    'pair_random': ('#999999', ':', 'x'), 'pair_magnitude': ('#1f77b4', ':', 'x'),
    'pair_blocktri_fas': ('#d62728', ':', 'P'), 'pair_blocktri_randorder': ('#ff9896', ':', 'P'),
    'elem_magnitude': ('#9467bd', '-.', 'v'),
}
for task in df.task.unique():
    sub = agg[agg.task == task]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, col, title in [(axes[0], 'test_oneshot', 'one-shot (no retraining)'),
                           (axes[1], 'test_ft', f"after {CFG['FT_EPOCHS']} epoch(s) masked fine-tuning")]:
        for m, (c, ls, mk) in STYLE.items():
            r = sub[sub.method == m].sort_values('target_sparsity')
            if len(r):
                ax.errorbar(r.sparsity, r[col], yerr=r[col + '_sd'].fillna(0), color=c, ls=ls, marker=mk,
                            ms=5, capsize=2, label=m)
        refs = [('baseline', 'black'), ('no_recurrence', '#e377c2')]
        if col == 'test_ft':
            refs = [('baseline_ft', 'black'), ('no_recurrence', '#e377c2')]
        for m, c in refs:
            r = sub[sub.method == m]
            if len(r):
                ax.axhline(r.test_ft.mean(), color=c, lw=1, ls='--', label=m)
        pr = sub[sub.method.str.startswith('paper_')]
        ax.scatter(pr.sparsity, pr[col], marker='*', s=120, color='orange', zorder=5, label='paper rules')
        ax.set_title(f'{task}: {title}'); ax.set_xlabel('W_hh sparsity'); ax.grid(alpha=0.3)
    axes[0].set_ylabel('test accuracy')
    axes[1].legend(fontsize=7, loc='lower left', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['OUT_DIR'], f'accuracy_vs_sparsity_{task}.png'), dpi=200)
    plt.show()

# Part D2 figure: training from the initial weights
START_STYLE = {
    'start_pair_random': ('#999999', ':', 'x'), 'start_acyclic_random': ('#ff9896', '-', 'P'),
    'rewind_pair_magnitude': ('#1f77b4', ':', 'x'), 'rewind_acyclic_fas': ('#d62728', '-', 'P'),
    'rewind_elem_magnitude': ('#9467bd', '-.', 'v'),
}
for task in df.task.unique():
    sub = agg[agg.task == task]
    if not sub.method.str.startswith(FROM_START).any():
        continue
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m, (c, ls, mk) in START_STYLE.items():
        r = sub[sub.method == m].sort_values('target_sparsity')
        if len(r):
            ax.errorbar(r.sparsity, r.test_ft, yerr=r.test_ft_sd.fillna(0), color=c, ls=ls, marker=mk,
                        ms=5, capsize=2, label=m)
    for m, c in [('baseline', 'black'), ('no_recurrence', '#e377c2')]:
        r = sub[sub.method == m]
        if len(r):
            ax.axhline(r.test_ft.mean(), color=c, lw=1, ls='--', label=m + ' (same budget)')
    ax.set(title=f"{task}: trained from initial weights ({CFG['BASE_EPOCHS']} epochs, fixed mask)",
           xlabel='W_hh sparsity', ylabel='test accuracy')
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['OUT_DIR'], f'from_start_{task}.png'), dpi=200)
    plt.show()

# %% [markdown]
# **Paired contrasts.** These are per-seed differences in test accuracy at matched sparsity, the numbers to report for each key question. `wins` counts the seeds where the first method was better. For D2 methods the `test_ft` stage means "after training from the initial weights".

# %%
contrasts = [('col_order_kappa', 'col_degree'), ('col_order_kappa', 'col_pagerank'), ('col_order_kappa', 'col_magnitude'),
             ('col_order_kappa', 'col_random'), ('pair_blocktri_fas', 'pair_blocktri_randorder'),
             ('pair_blocktri_fas', 'pair_magnitude'), ('pair_magnitude', 'elem_magnitude'),
             ('paper_binary', 'col_random'),
             ('start_acyclic_random', 'start_pair_random'), ('rewind_acyclic_fas', 'start_acyclic_random'),
             ('rewind_acyclic_fas', 'rewind_pair_magnitude'), ('rewind_pair_magnitude', 'start_pair_random'),
             ('rewind_pair_magnitude', 'rewind_elem_magnitude')]
rows = []
for task in df.task.unique():
    for s in sorted(df[(df.task == task) & (df.target_sparsity > 0) & (df.target_sparsity < 1)].target_sparsity.unique()):
        for a, b in contrasts:
            A_ = df[(df.task == task) & (df.method == a) & (df.target_sparsity == s)].set_index('seed')
            B_ = df[(df.task == task) & (df.method == b) & (df.target_sparsity == s)].set_index('seed')
            common = A_.index.intersection(B_.index)
            if len(common) == 0:
                continue
            for col in ['test_oneshot', 'test_ft']:
                d = (A_.loc[common, col] - B_.loc[common, col]).values
                if np.all(np.isnan(d)):
                    continue
                rows.append(dict(task=task, sparsity=s, contrast=f'{a} - {b}', stage=col, n=len(d),
                                 mean_diff=d.mean(), sd=d.std(ddof=1) if len(d) > 1 else np.nan,
                                 wins=int((d > 0).sum())))
con = pd.DataFrame(rows)
display(con.round(4))

# %% [markdown]
# **SCC-collapse diagnostic.** For the trained baseline's dependency graph, the plot shows how large the biggest strongly connected component is as more edges are kept, and how often $\kappa$ ties. When the largest SCC covers almost all neurons, every neuron gets the same order score: the poset carries no information at that threshold.

# %%
if os.path.exists(COLLAPSE_CSV):
    col = pd.read_csv(COLLAPSE_CSV)
    cagg = col.groupby(['task', 'edge_keep']).mean(numeric_only=True).reset_index()
    display(cagg.round(3))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for task in cagg.task.unique():
        r = cagg[cagg.task == task]
        ax[0].semilogx(r.edge_keep, r.largest_scc_frac, 'o-', label=task)
        ax[1].semilogx(r.edge_keep, r.kappa_modal_tie_frac, 's-', label=f'{task}: largest kappa tie')
        ax[1].semilogx(r.edge_keep, r.kappa_eq1_frac, 'd--', label=f'{task}: kappa = 1 (manuscript criterion)')
    ax[0].set(xlabel='fraction of off-diagonal edges kept', ylabel='largest SCC / H', title='SCC collapse')
    ax[1].set(xlabel='fraction of off-diagonal edges kept', ylabel='fraction of neurons', title='Order-score degeneracy')
    for a in ax:
        if 'edge_keep' in df and df.edge_keep.notna().any():
            for v in df.edge_keep.dropna().unique():
                a.axvline(v, color='gray', ls=':', lw=0.8)
        a.grid(alpha=0.3); a.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['OUT_DIR'], 'scc_collapse.png'), dpi=200)
    plt.show()

# %% [markdown]
# ## How to read the output
#
# - **`baseline_ft`.** If a pruned model beats `baseline_ft`, the pruning helped. If it beats only `baseline`, the gain came from the extra training epochs.
# - **Paper rules vs `no_recurrence`.** If `paper_continuous_tau*` lands at about 100% $W_{hh}$ sparsity with accuracy close to `no_recurrence`, the original ~87% figure measures an LSTM without recurrent weights. The cell state still integrates through input-driven gates.
# - **`paper_binary` vs `col_random` at 0.5.** Both masks ignore the weights. If they match within seed noise, the binary rule is random structured pruning.
# - **`col_order_kappa` vs `col_degree` / `col_pagerank`.** This is the test of whether the order structure adds anything over graph centrality. With `EDGE_KEEP='auto'` the graph is thresholded where $\kappa$ is least degenerate, which gives the order score its best chance. Check `kappa_modal_tie_frac`: if many neurons still tie, the score is partly random, and the collapse plot shows why. A denser threshold puts every neuron in one SCC.
# - **`pair_blocktri_fas` vs `pair_blocktri_randorder`.** If the FAS order wins, the dependency order carries real information. If it also stays close to `pair_magnitude`, the cost of making the recurrence acyclic is small. That cost is what buys the block-triangular stability decomposition verified in Part C.
# - **D2: `start_acyclic_random` vs `start_pair_random`.** A small gap means acyclic (cascade) recurrence is a cheap architectural prior, and the stability decomposition comes almost for free. A large gap that shrinks from post-hoc (D1) to from-start (D2) means the network can reorganise around the constraint when trained with it.
# - **D2: `rewind_acyclic_fas` vs `start_acyclic_random`.** A win means the dependency order learned by the dense network transfers to a fresh network.
# - **Height and width.** For block-triangular masks, height is the length of the cascade (sequential depth) and width is the largest set of independent modules (available parallelism). Dense triangular masks give a chain (width 1). Width grows only when sparsity also removes edges inside the order.
# - **Compute.** Masked dense matrices do not run faster. Report nonzero counts (FLOPs per step) and treat wall-clock speed-up as future work, unless you implement reduced or sparse kernels.

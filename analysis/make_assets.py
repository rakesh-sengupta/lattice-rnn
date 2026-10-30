"""Builds every table, figure and in-text number of the manuscript from the saved results.

Inputs : data/results.csv, data/collapse_diagnostic.csv (written by lattice_pruning.ipynb, v3)
Outputs: tables/*.tex, figs/*.pdf, numbers.tex (LaTeX macros for every number quoted in the text)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
os.makedirs(os.path.join(HERE, 'figs'), exist_ok=True)

df = pd.read_csv(os.path.join(HERE, 'data', 'results.csv'))
col = pd.read_csv(os.path.join(HERE, 'data', 'collapse_diagnostic.csv'))
assert not df.duplicated(['task', 'seed', 'method', 'target_sparsity']).any()

TASKS = ['mnist_rows', 'mnist_seq']
TASK_TEX = {'mnist_rows': 'Row-MNIST', 'mnist_seq': 'Chunked sequential MNIST'}
SP = {'mnist_rows': [0.25, 0.5, 0.75, 0.9], 'mnist_seq': [0.5, 0.9]}
SP_START = {'mnist_rows': [0.5, 0.75, 0.9], 'mnist_seq': [0.5, 0.9]}
SEEDS = sorted(df.seed.unique())
N_SEEDS = len(SEEDS)

macros = {}


def mac(name, value):
    macros[name] = value


def sel(task, method, s=None):
    x = df[(df.task == task) & (df.method == method)]
    if s is not None:
        x = x[np.isclose(x.target_sparsity, s)]
    return x.set_index('seed')


def ms(task, method, s=None, colname='test_ft'):
    v = sel(task, method, s)[colname].values * 100
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan), len(v)


def z(txt):
    """Avoid printing -0.0 / +0.0."""
    if txt in ('-0.0', '+0.0'):
        return '0.0'
    if txt.startswith('-'):
        return '\\ensuremath{-}' + txt[1:]
    if txt.startswith('+'):
        return '\\ensuremath{+}' + txt[1:]
    return txt


def fmt_ms(m, s):
    return f'{z(f"{m:.1f}")} $\\pm$ {s:.1f}'


def paired(task, a, b, s, colname='test_ft'):
    A, B = sel(task, a, s)[colname], sel(task, b, s)[colname]
    c = A.index.intersection(B.index)
    d = (A[c] - B[c]).values * 100
    return d.mean(), d.std(ddof=1), int((d > 0).sum()), len(d)


TKEY = {'mnist_rows': 'Rows', 'mnist_seq': 'Seq'}


def spkey(s):
    return {0.25: 'TwentyFive', 0.5: 'Fifty', 0.75: 'SeventyFive', 0.9: 'Ninety'}[s]


# ---------------------------------------------------------------- Table: original rules
rows = []
for t in TASKS:
    entries = [('Unpruned, $E_0$ epochs (reference)', 'baseline', None),
               ('Unpruned, $E_0+E_1$ epochs (equal-budget reference)', 'baseline_ft', None),
               ('No recurrence ($W_{hh}\\equiv 0$), $E_0$ epochs', 'no_recurrence', None),
               ('Continuous rule, $\\tau\\in\\{0.2,0.5,0.7\\}$', 'paper_continuous_tau0.2', None),
               ('Binary rule (first half of neurons kept)', 'paper_binary', None),
               ('Random columns, matched sparsity', 'col_random', 0.5)]
    for label, m, s in entries:
        x = sel(t, m, s)
        sp = x.actual_sparsity.mean() * 100
        one = x.test_oneshot.values * 100
        ft = x.test_ft.values * 100
        rows.append((t, label, sp, one.mean(), one.std(ddof=1), ft.mean(), ft.std(ddof=1), m))
    # identical masks for all tau
    a = sel(t, 'paper_continuous_tau0.2').test_ft
    for tau in ['0.5', '0.7']:
        assert np.allclose(a.values, sel(t, f'paper_continuous_tau{tau}').test_ft.values)

with open(os.path.join(HERE, 'tables', 'tab_original.tex'), 'w') as f:
    f.write('\\begin{tabular}{@{}lrcc@{}}\n\\toprule\n')
    f.write('Model & $W_{hh}$ sparsity (\\%) & One-shot acc.\\ (\\%) & Final acc.\\ (\\%) \\\\\n\\midrule\n')
    for t in TASKS:
        f.write(f'\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{TASK_TEX[t]}}}}} \\\\\n')
        for (tt, label, sp, om, osd, fm, fsd, m) in [r for r in rows if r[0] == t]:
            one = '---' if m in ('baseline', 'baseline_ft', 'no_recurrence') else fmt_ms(om, osd)
            f.write(f'\\quad {label} & {sp:.1f} & {one} & {fmt_ms(fm, fsd)} \\\\\n')
        if t != TASKS[-1]:
            f.write('\\addlinespace\n')
    f.write('\\bottomrule\n\\end{tabular}\n')

for t in TASKS:
    k = TKEY[t]
    for m, name in [('baseline', 'Base'), ('baseline_ft', 'BaseFt'), ('no_recurrence', 'NoRec'),
                    ('paper_continuous_tau0.2', 'Cont'), ('paper_binary', 'Bin')]:
        mm, sd, _ = ms(t, m)
        mac(f'{k}{name}', f'{mm:.1f}')
        mac(f'{k}{name}SD', f'{sd:.1f}')
    mm, sd, _ = ms(t, 'paper_continuous_tau0.2', colname='test_oneshot')
    mac(f'{k}ContOneShot', f'{mm:.1f}')
    mm, sd, _ = ms(t, 'col_random', 0.5)
    mac(f'{k}ColRandFifty', f'{mm:.1f}')
    d = paired(t, 'paper_binary', 'col_random', 0.5)
    mac(f'{k}BinMinusRand', f'{d[0]:+.1f}')
    mac(f'{k}BinMinusRandSD', f'{d[1]:.1f}')
    d = paired(t, 'paper_continuous_tau0.2', 'no_recurrence', None) if False else None

# ---------------------------------------------------------------- Table: post-hoc matched sparsity
METHODS = [('Unstructured', [('elem_magnitude', 'Magnitude $|w|$')]),
           ('Neuron pair', [('pair_magnitude', 'Magnitude $S_{ij}$'),
                            ('pair_random', 'Random')]),
           ('Acyclic (cascade)', [('pair_blocktri_fas', 'FAS order'),
                                  ('pair_blocktri_randorder', 'Random order')]),
           ('Column (neuron)', [('col_order_kappa', 'Order score $\\kappa$'),
                                ('col_degree', 'Degree'),
                                ('col_pagerank', 'PageRank'),
                                ('col_magnitude', 'Norm $\\|W_{hh}^{(:,j)}\\|_2$'),
                                ('col_random', 'Random')])]


def posthoc_table(colname, fname):
    with open(os.path.join(HERE, 'tables', fname), 'w') as f:
        ncol = len(SP['mnist_rows']) + len(SP['mnist_seq'])
        f.write('\\begin{tabular}{@{}ll' + 'c' * ncol + '@{}}\n\\toprule\n')
        f.write(' & & \\multicolumn{4}{c}{Row-MNIST} & \\multicolumn{2}{c}{Chunked seq.\\ MNIST} \\\\\n')
        f.write('\\cmidrule(lr){3-6}\\cmidrule(l){7-8}\n')
        f.write('Granularity & Score & ' + ' & '.join(f'{int(s*100)}\\%' for t in TASKS for s in SP[t]) + ' \\\\\n\\midrule\n')
        for gname, ms_ in METHODS:
            for i, (m, label) in enumerate(ms_):
                cells = []
                for t in TASKS:
                    for s in SP[t]:
                        mm, sd, n = ms(t, m, s, colname)
                        assert n == N_SEEDS
                        cells.append(fmt_ms(mm, sd))
                f.write((gname if i == 0 else '') + f' & {label} & ' + ' & '.join(cells) + ' \\\\\n')
            f.write('\\addlinespace\n')
        ref = 'baseline_ft' if colname == 'test_ft' else 'baseline'
        reflabel = 'Unpruned, equal budget' if colname == 'test_ft' else 'Unpruned'
        cells = []
        for t in TASKS:
            mm, sd, _ = ms(t, ref, None, 'test_ft')
            cells.append(f'\\multicolumn{{{len(SP[t])}}}{{c}}{{{fmt_ms(mm, sd)}}}')
        f.write(f'Reference & {reflabel} & ' + ' & '.join(cells) + ' \\\\\n')
        cells = []
        for t in TASKS:
            mm, sd, _ = ms(t, 'no_recurrence', None, 'test_ft')
            cells.append(f'\\multicolumn{{{len(SP[t])}}}{{c}}{{{fmt_ms(mm, sd)}}}')
        f.write('Control & No recurrence & ' + ' & '.join(cells) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')


posthoc_table('test_ft', 'tab_posthoc_ft.tex')
posthoc_table('test_oneshot', 'tab_posthoc_oneshot.tex')

for t in TASKS:
    for s in SP[t]:
        for m, nm in [('elem_magnitude', 'Elem'), ('pair_magnitude', 'PairMag'), ('pair_random', 'PairRand'),
                      ('pair_blocktri_fas', 'Fas'), ('pair_blocktri_randorder', 'FasRand'),
                      ('col_order_kappa', 'Kappa'), ('col_degree', 'Degree'), ('col_pagerank', 'PageRank'),
                      ('col_magnitude', 'ColMag'), ('col_random', 'ColRand')]:
            mm, sd, _ = ms(t, m, s)
            mac(f'{TKEY[t]}{nm}{spkey(s)}', f'{mm:.1f}')
            mm, sd, _ = ms(t, m, s, 'test_oneshot')
            mac(f'{TKEY[t]}{nm}{spkey(s)}OneShot', f'{mm:.1f}')
    col_ms = [ms(t, m, 0.9)[0] for m in ['col_order_kappa', 'col_degree', 'col_pagerank', 'col_magnitude', 'col_random']]
    mac(f'{TKEY[t]}ColMinNinety', f'{min(col_ms):.1f}')
    mac(f'{TKEY[t]}ColMaxNinety', f'{max(col_ms):.1f}')

# ---------------------------------------------------------------- Table: from-start
START = [('start_pair_random', 'Random pair mask'),
         ('start_acyclic_random', 'Random acyclic mask'),
         ('rewind_pair_magnitude', 'Rewound pair-magnitude mask'),
         ('rewind_acyclic_fas', 'Rewound acyclic FAS mask'),
         ('rewind_elem_magnitude', 'Rewound unstructured magnitude mask')]
with open(os.path.join(HERE, 'tables', 'tab_fromstart.tex'), 'w') as f:
    f.write('\\begin{tabular}{@{}l' + 'c' * 5 + '@{}}\n\\toprule\n')
    f.write(' & \\multicolumn{3}{c}{Row-MNIST} & \\multicolumn{2}{c}{Chunked seq.\\ MNIST} \\\\\n')
    f.write('\\cmidrule(lr){2-4}\\cmidrule(l){5-6}\n')
    f.write('Mask held fixed from initialisation & ' + ' & '.join(f'{int(s*100)}\\%' for t in TASKS for s in SP_START[t]) + ' \\\\\n\\midrule\n')
    for m, label in START:
        cells = []
        for t in TASKS:
            for s in SP_START[t]:
                x = sel(t, m, s)
                if len(x) == 0:
                    cells.append('---')
                else:
                    mm, sd, n = ms(t, m, s)
                    assert n == N_SEEDS
                    cells.append(fmt_ms(mm, sd))
        f.write(f'{label} & ' + ' & '.join(cells) + ' \\\\\n')
    f.write('\\addlinespace\n')
    for m, label in [('baseline', 'Dense (same budget)'), ('no_recurrence', 'No recurrence (same budget)')]:
        cells = []
        for t in TASKS:
            mm, sd, _ = ms(t, m)
            cells.append(f'\\multicolumn{{{len(SP_START[t])}}}{{c}}{{{fmt_ms(mm, sd)}}}')
        f.write(f'{label} & ' + ' & '.join(cells) + ' \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')

for t in TASKS:
    for s in SP_START[t]:
        for m, nm in [('start_pair_random', 'StartRand'), ('start_acyclic_random', 'StartAcyc'),
                      ('rewind_pair_magnitude', 'RewPairMag'), ('rewind_acyclic_fas', 'RewFas'),
                      ('rewind_elem_magnitude', 'RewElem')]:
            if len(sel(t, m, s)):
                mac(f'{TKEY[t]}{nm}{spkey(s)}', f'{ms(t, m, s)[0]:.1f}')

# ---------------------------------------------------------------- Table: paired contrasts
CONTRASTS = [
    ('Order score vs.\\ centrality', [
        ('col_order_kappa', 'col_degree', '$\\kappa$ $-$ degree'),
        ('col_order_kappa', 'col_pagerank', '$\\kappa$ $-$ PageRank'),
        ('col_order_kappa', 'col_magnitude', '$\\kappa$ $-$ column norm'),
        ('col_order_kappa', 'col_random', '$\\kappa$ $-$ random column')]),
    ('Does the order matter?', [
        ('pair_blocktri_fas', 'pair_blocktri_randorder', 'Acyclic: FAS $-$ random order'),
        ('rewind_acyclic_fas', 'start_acyclic_random', 'From start: rewound FAS $-$ random acyclic')]),
    ('Cost of acyclicity', [
        ('pair_blocktri_fas', 'pair_magnitude', 'Post hoc: acyclic FAS $-$ pair magnitude'),
        ('pair_blocktri_fas', 'pair_random', 'Post hoc: acyclic FAS $-$ random pair'),
        ('start_acyclic_random', 'start_pair_random', 'From start: random acyclic $-$ random pair'),
        ('rewind_acyclic_fas', 'rewind_pair_magnitude', 'From start: rewound FAS $-$ rewound pair magnitude')]),
]


def contrast_cell(t, a, b, s, colname='test_ft'):
    if len(sel(t, a, s)) == 0 or len(sel(t, b, s)) == 0:
        return '---'
    m, sd, w, n = paired(t, a, b, s, colname)
    return f'{z(f"{m:+.1f}")} $\\pm$ {sd:.1f} ({w}/{n})'


with open(os.path.join(HERE, 'tables', 'tab_contrasts.tex'), 'w') as f:
    f.write('\\begin{tabular}{@{}l' + 'c' * 6 + '@{}}\n\\toprule\n')
    f.write(' & \\multicolumn{4}{c}{Row-MNIST} & \\multicolumn{2}{c}{Chunked seq.\\ MNIST} \\\\\n')
    f.write('\\cmidrule(lr){2-5}\\cmidrule(l){6-7}\n')
    f.write('Contrast (percentage points) & ' + ' & '.join(f'{int(s*100)}\\%' for t in TASKS for s in SP[t]) + ' \\\\\n\\midrule\n')
    for gname, cs in CONTRASTS:
        f.write(f'\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{gname}}}}} \\\\\n')
        for a, b, label in cs:
            cells = [contrast_cell(t, a, b, s) for t in TASKS for s in SP[t]]
            f.write(f'\\quad {label} & ' + ' & '.join(cells) + ' \\\\\n')
        f.write('\\addlinespace\n')
    f.write('\\multicolumn{7}{@{}l}{\\textit{One shot (no retraining)}} \\\\\n')
    for a, b, label in [('col_order_kappa', 'col_degree', '$\\kappa$ $-$ degree'),
                        ('col_order_kappa', 'col_random', '$\\kappa$ $-$ random column'),
                        ('pair_blocktri_fas', 'pair_blocktri_randorder', 'Acyclic: FAS $-$ random order')]:
        cells = [contrast_cell(t, a, b, s, 'test_oneshot') for t in TASKS for s in SP[t]]
        f.write(f'\\quad {label} & ' + ' & '.join(cells) + ' \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')

for t in TASKS:
    for s in SP[t]:
        for a, b, nm in [('col_order_kappa', 'col_degree', 'KappaDeg'), ('col_order_kappa', 'col_random', 'KappaRand'),
                         ('pair_blocktri_fas', 'pair_blocktri_randorder', 'FasRand'),
                         ('pair_blocktri_fas', 'pair_magnitude', 'FasPairMag'),
                         ('col_order_kappa', 'col_pagerank', 'KappaPR'), ('col_order_kappa', 'col_magnitude', 'KappaColMag')]:
            m, sd, w, n = paired(t, a, b, s)
            mac(f'{TKEY[t]}D{nm}{spkey(s)}', f'{m:+.1f}')
            mac(f'{TKEY[t]}D{nm}{spkey(s)}Wins', f'{w}')
            m, sd, w, n = paired(t, a, b, s, 'test_oneshot')
            mac(f'{TKEY[t]}D{nm}{spkey(s)}OneShot', f'{m:+.1f}')
            mac(f'{TKEY[t]}D{nm}{spkey(s)}OneShotWins', f'{w}')
    for s in SP_START[t]:
        for a, b, nm in [('start_acyclic_random', 'start_pair_random', 'AcycStart'),
                         ('rewind_acyclic_fas', 'start_acyclic_random', 'RewFasAcyc'),
                         ('rewind_acyclic_fas', 'rewind_pair_magnitude', 'RewFasPair')]:
            m, sd, w, n = paired(t, a, b, s)
            mac(f'{TKEY[t]}D{nm}{spkey(s)}', f'{m:+.1f}')
            mac(f'{TKEY[t]}D{nm}{spkey(s)}SD', f'{sd:.1f}')
            mac(f'{TKEY[t]}D{nm}{spkey(s)}Wins', f'{w}')

# ---------------------------------------------------------------- structure statistics
fw = df.dropna(subset=['fas_forward_weight']).groupby(['task', 'seed']).fas_forward_weight.first()
mac('FwdMin', f'{fw.min():.3f}')
mac('FwdMax', f'{fw.max():.3f}')
ek = df.dropna(subset=['edge_keep']).groupby(['task', 'seed']).edge_keep.first()
mac('EdgeKeepMin', f'{ek.min()*100:.2f}')
mac('EdgeKeepMax', f'{ek.max()*100:.2f}')
tie = df.dropna(subset=['kappa_modal_tie_frac']).groupby(['task', 'seed']).kappa_modal_tie_frac.first()
mac('TieMin', f'{tie.min()*100:.0f}')
mac('TieMax', f'{tie.max()*100:.0f}')
cg = col.groupby('edge_keep').largest_scc_frac.agg(['min', 'max'])
mac('SccFiveMin', f'{cg.loc[0.05, "min"]*100:.0f}')
mac('SccFiveMax', f'{cg.loc[0.05, "max"]*100:.0f}')
mac('SccTenMin', f'{cg.loc[0.1, "min"]*100:.0f}')
for t in TASKS:
    for m, nm in [('pair_blocktri_fas', 'Fas'), ('pair_blocktri_randorder', 'FasRand'), ('start_acyclic_random', 'StartAcyc')]:
        x = sel(t, m, 0.9)
        mac(f'{TKEY[t]}{nm}NinetyHeight', f'{x.poset_height.mean():.0f}')
        mac(f'{TKEY[t]}{nm}NinetyWidth', f'{x.poset_width.mean():.0f}')
mac('NSeeds', str(N_SEEDS))
pn = pd.read_csv(os.path.join(HERE, 'data', 'phi_null.csv'))
mac('PhiNullMin', f'{pn.null_mean.min():.3f}')
mac('PhiNullMax', f'{pn.null_mean.max():.3f}')
exc = (pn.phi - pn.null_mean) * 100
mac('PhiExcessMin', f'{exc.min():.1f}')
mac('PhiExcessMax', f'{exc.max():.1f}')
mac('PhiZMin', f'{pn.z.min():.0f}')
mac('PhiNullShuffles', '20')
# one-shot pair magnitude at 50% on seq (per-seed range)
v = sel('mnist_seq', 'pair_magnitude', 0.5).test_oneshot.values * 100
mac('SeqPairMagFiftyOneShotMin', f'{v.min():.1f}')
mac('SeqPairMagFiftyOneShotMax', f'{v.max():.1f}')
col90 = sel('mnist_rows', 'col_random', 0.9).actual_sparsity.mean() * 100
mac('ColNinetySparsity', f'{col90:.1f}')
gaps = [ms('mnist_rows', 'baseline_ft')[0] - ms('mnist_rows', m, sp)[0]
        for _, mlist in METHODS for m, _ in mlist for sp in [0.25, 0.5, 0.75]]
mac('RowsMaxGapSeventyFive', f'{max(gaps):.1f}')
ft_med = df.groupby('task').ft_seconds.median()
mac('RowsFtSeconds', f'{ft_med["mnist_rows"]:.1f}')
mac('SeqFtSeconds', f'{ft_med["mnist_seq"]:.1f}')

with open(os.path.join(HERE, 'numbers.tex'), 'w') as f:
    f.write('% Auto-generated by make_assets.py from data/results.csv -- do not edit by hand.\n')
    for k, v in sorted(macros.items()):
        assert k.isalpha(), k
        f.write(f'\\newcommand{{\\n{k}}}{{{z(v)}}}\n')
        try:
            fv = float(v)
            f.write(f'\\newcommand{{\\n{k}Abs}}{{{abs(fv):.1f}}}\n')
        except ValueError:
            pass
print(f'{len(macros)} macros written')

# ---------------------------------------------------------------- figures
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.edgecolor': '#52514e',
    'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b',
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
    'grid.color': '#e4e3df', 'grid.linewidth': 0.5,
})
BLUE, AQUA, ORANGE, VIOLET, GRAY, INK = '#2a78d6', '#1baf7a', '#eb6834', '#6250d6', '#8a8984', '#0b0b0b'

# Figure: SCC collapse
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
for t, c, mk in [('mnist_rows', BLUE, 'o'), ('mnist_seq', ORANGE, 's')]:
    g = col[col.task == t].groupby('edge_keep')
    x = np.array(sorted(col.edge_keep.unique())) * 100
    for ax, var in [(axes[0], 'largest_scc_frac'), (axes[1], 'kappa_modal_tie_frac')]:
        m, lo, hi = g[var].mean().values, g[var].min().values, g[var].max().values
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(x, m, color=c, marker=mk, ms=4, lw=1.4, label=TASK_TEX[t])
    m = g['kappa_eq1_frac'].mean().values
    axes[1].plot(x, m, color=c, ls='--', lw=1.0, marker=mk, ms=3, mfc='white',
                 label=f'{TASK_TEX[t]}: $\\kappa=1$')
for ax in axes:
    ax.set_xscale('log')
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.minorticks_off()
    ax.axvspan(ek.min() * 100, ek.max() * 100, color='#d8d7d2', alpha=0.6, lw=0)
    ax.grid(True, which='major')
    ax.set_xlabel(r'Off-diagonal edges kept, $\rho$ (%)')
    ax.set_ylim(-0.03, 1.03)
axes[0].set_ylabel('Largest SCC / $H$')
axes[0].set_title('(a) Strongly connected components', loc='left')
axes[1].set_ylabel('Fraction of neurons')
axes[1].set_title(r'(b) Largest tie in $\kappa$ (solid); $\kappa=1$ (dashed)', loc='left')
axes[0].legend(frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'figs', 'fig_scc_collapse.pdf'))
plt.close(fig)

# Figure: post-hoc accuracy vs sparsity
SHOW = [('elem_magnitude', 'Unstructured magnitude', BLUE, '-', 'o', True),
        ('pair_magnitude', 'Pair magnitude', AQUA, '-', 's', True),
        ('pair_random', 'Pair random', GRAY, '--', 's', False),
        ('pair_blocktri_fas', 'Acyclic, FAS order', ORANGE, '-', '^', True),
        ('pair_blocktri_randorder', 'Acyclic, random order', ORANGE, '--', '^', False),
        ('col_order_kappa', r'Column, $\kappa$', VIOLET, '-', 'D', True),
        ('col_random', 'Column, random', GRAY, ':', 'D', False)]


def panel(ax, t, methods, colname, sps, ref, refname):
    for m, label, c, ls, mk, filled in methods:
        xs, ys, es = [], [], []
        for s in sps:
            x = sel(t, m, s)
            if len(x) == 0:
                continue
            v = x[colname].values * 100
            xs.append(s * 100); ys.append(v.mean()); es.append(v.std(ddof=1))
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=c, ls=ls, marker=mk, ms=4.5, lw=1.4, capsize=2,
                        mfc=c if filled else 'white', mec=c, elinewidth=0.8, label=label)
    for rm, rlabel, rls in [(ref, refname, '--'), ('no_recurrence', 'No recurrence', '-.')]:
        v = sel(t, rm).test_ft.mean() * 100
        ax.axhline(v, color=INK if rm == ref else GRAY, ls=rls, lw=0.8)
        lo, hi = ax.get_ylim()
        right = rm == ref
        ax.annotate(f'{rlabel} ({v:.1f})', xy=(1.0 if right else 0.0, v), xycoords=('axes fraction', 'data'),
                    xytext=(-3 if right else 3, 2), textcoords='offset points', ha='right' if right else 'left',
                    va='bottom', fontsize=6.5,
                    color='#52514e')
    ax.grid(True, axis='y')
    ax.set_xlabel(r'$W_{hh}$ sparsity (%)')
    ax.set_xticks([s * 100 for s in sps])


fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, t in zip(axes, TASKS):
    panel(ax, t, SHOW, 'test_ft', SP[t], 'baseline_ft', 'Unpruned, equal budget')
    ax.set_title(f'({"ab"[TASKS.index(t)]}) {TASK_TEX[t]}', loc='left')
axes[0].set_ylabel('Test accuracy after fine-tuning (%)')
axes[0].set_ylim(89, 100)
axes[1].set_ylim(45, 100)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(os.path.join(HERE, 'figs', 'fig_posthoc.pdf'))
plt.close(fig)

# Figure: from start
SHOW_START = [('rewind_elem_magnitude', 'Rewound unstructured magnitude', BLUE, '-', 'o', True),
              ('rewind_pair_magnitude', 'Rewound pair magnitude', AQUA, '-', 's', True),
              ('start_pair_random', 'Random pair', GRAY, '--', 's', False),
              ('rewind_acyclic_fas', 'Rewound acyclic FAS', ORANGE, '-', '^', True),
              ('start_acyclic_random', 'Random acyclic', ORANGE, '--', '^', False)]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75))
for ax, t in zip(axes, TASKS):
    panel(ax, t, SHOW_START, 'test_ft', SP_START[t], 'baseline', 'Dense, same budget')
    ax.set_title(f'({"ab"[TASKS.index(t)]}) {TASK_TEX[t]}', loc='left')
axes[0].set_ylabel('Test accuracy (%)')
axes[0].set_ylim(91, 99.5)
axes[1].set_ylim(45, 100)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(os.path.join(HERE, 'figs', 'fig_fromstart.pdf'))
plt.close(fig)
print('figures written')

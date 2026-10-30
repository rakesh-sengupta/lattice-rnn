# Lattice-based pruning of recurrent neural networks: corrected theory and matched-sparsity evaluation

Code and data for

> R. Sengupta, *Lattice-Based Pruning of Recurrent Neural Networks via Poset Modeling: A Corrected Theory and a Matched-Sparsity Evaluation*. Revised version of [arXiv:2502.16525](https://arxiv.org/abs/2502.16525).

The revised paper corrects the order-theoretic analysis of the first arXiv version and replaces its experiments with a matched-sparsity evaluation of a 128-unit LSTM. It covers:

- **Order theory.** Meet-irreducible ideals of a finite poset are in bijection with its elements, and the earlier critical-set criterion selects exactly the elements comparable to all others.
- **Dependency order.** An order is built from the condensation of a thresholded dependency graph into strongly connected components. It yields an order score κ and acyclic (cascade) pruning in a feedback-arc-set ordering.
- **Stability.** For Cohen–Grossberg networks and LSTMs, acyclic pruning makes local stability decompose over modules.
- **Experiments.** Row-wise and chunked sequential MNIST, 3 seeds, equal training budgets, post-hoc pruning and training from initialisation.

## Contents

| Path | What it is |
|---|---|
| `lattice_pruning.ipynb` | The experiment notebook (also runs on Google Colab). |
| `lattice_pruning.py` | The same code as a plain script, in `# %%` cell format. |
| `analysis/make_assets.py` | Builds every table, figure and in-text number of the paper from `analysis/data/`. |
| `analysis/phi_null.py` | Shuffled-weight null model for the forward weight fraction φ. |
| `analysis/data/results.csv` | Raw per-seed results of the full run used in the paper. |
| `analysis/data/collapse_diagnostic.csv` | SCC-collapse diagnostic of the trained baselines. |
| `analysis/data/phi_null.csv` | Output of `phi_null.py` (20 shuffles per network). |
| `checkpoints/*_base.pt` | The six trained unpruned baselines (2 tasks × 3 seeds). |

## The notebook

The notebook has five parts:

| Part | Content | Runtime |
|---|---|---|
| **A** | Order-theory checks on random finite posets: the Birkhoff characterisation and what the meet-irreducibility criterion selects. | seconds |
| **B** | Audit of the pruning rules of the first version, without training. | seconds |
| **C** | Numerical check of the block-triangular stability theorem on a Cohen–Grossberg network. | seconds |
| **D1** | Post-hoc pruning at matched sparsity, with equal-budget and no-recurrence controls. Methods: random, magnitude, degree, PageRank and κ columns; random, magnitude and acyclic neuron pairs; unstructured magnitude; and the original rules. | see below |
| **D2** | Masks held fixed while training from initialisation (random acyclic vs random unconstrained, plus rewound masks). | see below |
| **E** | Tables, figures and paired contrasts. | seconds |

Settings live in the first code cell:
- `QUICK = True` runs a smoke test in about 10 minutes on a CPU.
- `QUICK = False` runs the full experiment of the paper. Use a GPU; on Colab choose Runtime → Change runtime type → T4 GPU. On CPU, `mnist_seq` alone takes more than a day.
- `DRIVE_DIR` saves results to Google Drive.

Every result row is written to disk as soon as it is computed, so an interrupted run resumes where it stopped.

## Reproducing the paper's tables and figures

```bash
pip install -r requirements.txt
cd analysis
python make_assets.py              # writes tables/*.tex, figs/*.pdf and numbers.tex
python phi_null.py ../lattice_pruning.py ../checkpoints   # recomputes phi_null.csv
```

To regenerate `results.csv` from scratch, run the notebook with `QUICK = False`. Then copy `results.csv` and `collapse_diagnostic.csv` from its output folder (`lattice_pruning_results_v2/`) into `analysis/data/`.

## Experimental setup (summary)

- **Model:** single-layer LSTM, H = 128, with a linear readout.
- **Training:** Adam (lr 1e-3), batch 128, gradient clipping 1.0. Baselines train for 8 epochs; pruned models fine-tune for 3 epochs with the mask fixed (weights and gradients zeroed).
- **Tasks:** row-MNIST (28 steps × 28 pixels) and chunked sequential MNIST (112 steps × 7 pixels). The data split is 55k / 5k / 10k.
- **Sparsity** is the fraction of zero entries in `weight_hh_l0` (4H × H); input and output weights are never pruned.
- **Seeds:** 0, 1, 2.

## Citation

See `CITATION.cff`. Until the journal version appears, please cite arXiv:2502.16525.

## License

MIT (see `LICENSE`).

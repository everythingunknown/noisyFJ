# Noisy FJ

Julia 1.11.5 and Python 3.11+.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run.sh
```

The pipeline runs Exact, FP, AA, PFS and FRPFS on the 12 networks at 1,000, 2,000 and 4,000 forests. On the four small networks it also runs 500 paired opinion-and-noise trials with two rounds of measurement allocation, and compares exact and sampled centralities for every node. AA, left-Jacobi GMRES(30) and SOR are compared on eight networks using one Julia/BLAS thread, the same 16 sources, and three runs per source.

Measurement allocation uses the decreasing score `w_i/[k_i(k_i+1)]`, with one initial observation per node and exactly `n` additional observations each round. Degree uses in-degree, Structural uses exact column sums, and Centrality uses 4,000-forest FRPFS scores. Middle updates its variance proxy between rounds. Each trial redraws opinions, and all policies share each node's observation sequence. Errors are evaluated by sparse linear solves.

Data sources and preprocessing are specified in `datasets.toml` and `prepare.py`; original edge directions are retained. Random generators initialize automatically, so fresh sampled results vary between runs. This code reruns the experimental procedures; it does not embed the particular random realizations used for the reported tables. Screening and estimation use separate generators in fresh runs.

`solver_parameters.toml` gives the retained SOR relaxation parameters and their previously measured selection costs on the paper's machine. Solver totals add graph/workspace setup, the sum of per-source median query times, and this retained selection cost for SOR. Screening, reference construction and compilation warmup are excluded. The retained costs are fixed measurements, not new measurements on the machine executing the code. At omega=1, SOR reduces to the base local iteration (BLI).

Results include `results/comparison.csv`, `results/forest_budgets.csv`, `results/resampling.csv`, `results/all_nodes_summary.csv`, `results/solver_baselines.csv`, `results/figure1.pdf`, and `results/figure2.pdf`. Full estimates, aligned all-node curves, per-query timings and per-trial measurements are saved in each dataset directory. Large-network runs require sufficient RAM for the sparse graphs and the GMRES workspace.

One dataset:

```sh
bash run.sh college-msg
```

Completed stages can be reused after interruption. Use a fresh results directory for an independent run.

# Particle Competition and Cooperation (PCC)

MATLAB and C/MEX implementation of **Particle Competition and Cooperation (PCC)** for semi-supervised learning on networks.

This repository accompanies the paper:

> Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves; Pedrycz, Witold; and Liu, Jiming. "Particle Competition and Cooperation in Networks for Semi-Supervised Learning." *IEEE Transactions on Knowledge and Data Engineering*, **24**(9), 1686–1698 (2012). https://doi.org/10.1109/TKDE.2011.119

- [Published article (DOI)](https://doi.org/10.1109/TKDE.2011.119)
- [Author-hosted accepted manuscript (PDF)](https://www.fabriciobreve.com/artigos/ieee-tkde-2009.pdf)

## Overview

PCC is a graph-based semi-supervised learning method. Labeled vertices initialize particles belonging to their respective classes. As particles traverse the network, they compete for vertices of different classes and cooperate with particles of their own class. The algorithm combines random and preferential (greedy) movement and updates class dominance values over time.

The input can be a feature matrix, from which a graph is built, or a previously constructed graph. The implementation provides a pure MATLAB execution path and a C/MEX accelerated path.

**Note on versions:** This repository has evolved since the 2012 publication, including graph-construction changes, early-stopping options, performance improvements, and a March 2026 correction to the greedy-walk fallback. The current `master` branch should not be assumed to reproduce every detail of the original 2012 implementation. See [`changelog.txt`](changelog.txt) and the commit history.

## Requirements

- **MATLAB R2021a or newer** for the current `name=value` function-call syntax used in `example.m` and `benchmark.m`. MATLAB `arguments` blocks were introduced earlier, in R2019b, but the scripts in this repository use the newer call syntax.
- **Statistics and Machine Learning Toolbox** (for example, `knnsearch`, `unifrnd`, and `confusionmat`).
- A compatible C compiler configured with MATLAB's `mex` command if you need to rebuild the accelerated routine.

The repository includes precompiled `pccloop` MEX binaries for **64-bit Windows** (`.mexw64`) and **64-bit Linux** (`.mexa64`). Binary compatibility with a given MATLAB release or system is not guaranteed.

### Compiling the MEX routine

From the repository folder in MATLAB:

```matlab
mex -setup C
mex pccloop.c
```

The generated extension depends on your platform. If compilation is unavailable, use `mex=false` to run the pure MATLAB implementation.

## Quick start

The included `example.m` demonstrates PCC on the **Wine** dataset from the UCI Machine Learning Repository. It randomly labels approximately 10% of the samples (with at least one labeled sample per class), trains with `k=10` and standardized Euclidean distance, and reports accuracy and timing for both MATLAB and MEX implementations.

Set the MATLAB current folder to the repository root, then run:

```matlab
example
```

To call PCC directly:

```matlab
% X: one sample per row; slabel: class IDs 1, 2, ... and 0 for unlabeled
owner = pcc(X, uint16(slabel), k=10, disttype="seuclidean", mex=false);
```

Change `mex=false` to `mex=true` to use the compiled routine.

For a larger timing comparison using **Wine** and **Digits**, run:

```matlab
benchmark
```

The benchmark can take substantially longer with early stopping disabled. Review its `N_RUNS` and `use_earlystop` settings before running it.

## Interface and options

```matlab
[owner, pot, owndeg, distnode] = pcc(X, slabel, ...
    k=10, disttype="seuclidean", mex=true);
```

- `X`: feature matrix (rows are samples), or a graph structure when `Xgraph=true`.
- `slabel`: labels for the samples; **0** means unlabeled, positive consecutive integers denote classes.
- `k`: number of nearest neighbors used in graph construction (default approximately 5% of the number of samples).
- `disttype`: distance metric passed to `knnsearch` (default `"euclidean"`).
- `mex`: choose accelerated MEX (`true`, default) or pure MATLAB (`false`).
- `earlystop`: enable convergence-based stopping (default `true`).
- `valpha`, `maxiter`: early-stopping control and maximum number of iterations.
- `pgrd`, `deltav`, `deltap`, `dexp`: model dynamics parameters.
- `nclass`: number of classes; inferred from `slabel` by default.
- `useseed` and `seed`: optional random seed controls.
- `Xgraph`: accept a graph previously built with `pcc_buildgraph`.

`owner` contains the predicted class of each vertex. `pot` contains class dominance values; `owndeg` provides the accumulated, normalized fuzzy ownership output. `distnode` contains particle-to-vertex distance records. Consult the comments in [`pcc.m`](pcc.m) for the full interface.

### Stochastic behavior and reproducibility

PCC is stochastic, and the MATLAB and MEX paths use **different random-number generators**: MATLAB `rng` versus C `rand`/`srand`. Equal seed values do **not** imply identical random streams, trajectories, or classification results between implementations. When comparing methods, document seeds, platform, options, and MATLAB version. Measured speedups depend on the dataset, configuration, machine, and stopping policy.

## Repository contents

| File | Description |
| --- | --- |
| `pcc.m` | Main PCC implementation and configuration |
| `pcc_buildgraph.m` | Builds and returns a graph that can be reused |
| `pccloop.c` | C source for the accelerated MEX routine |
| `pccloop.mexw64`, `pccloop.mexa64` | Precompiled MEX binaries for Windows and Linux |
| `example.m` | Wine dataset example (MATLAB versus MEX) |
| `benchmark.m` | Wine/Digits timing and accuracy comparisons |
| `slabelgen.m`, `stmwevalk.m`, `kappa.m`, `sumcol.m` | Supporting utilities |
| `wine.data`, `wine.names`, `digits.mat` | Included example datasets |
| `changelog.txt` | Historical implementation changes |

Historical editor and binary backup files are retained in the repository; they are not required for the quick-start example.

## Citation

If you use this implementation in research, please cite the original article:

```bibtex
@article{breve2012particle,
  author  = {Breve, Fabricio Aparecido and Zhao, Liang and Quiles, Marcos Gon{\c{c}}alves and Pedrycz, Witold and Liu, Jiming},
  title   = {Particle Competition and Cooperation in Networks for Semi-Supervised Learning},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume  = {24},
  number  = {9},
  pages   = {1686--1698},
  year    = {2012},
  doi     = {10.1109/TKDE.2011.119}
}
```

## License

The original project source code is licensed under the [Apache License 2.0](LICENSE). Copyright © 2012–2026 Fabricio Aparecido Breve.

The included Wine and Digits datasets may have separate licenses or attribution requirements. Their presence in the repository does not mean that the Apache License for the code applies to third-party datasets.

# graph-fraud-drift

[![CI](https://github.com/rkesseku/graph-fraud-drift/actions/workflows/ci.yml/badge.svg)](https://github.com/rkesseku/graph-fraud-drift/actions/workflows/ci.yml)

**Do graph neural networks actually beat gradient boosting for illicit-transaction detection, once you evaluate them honestly over time?**
This repo answers that on the public Elliptic Bitcoin transaction graph with a strict temporal split, five models, several seeds, cost-sensitive alert thresholds and a block bootstrap over time steps, and it shows what happens to every model after a real-world regime change (the dark-market shutdown in the test period).

## Why this exists

Fraud and anti-money-laundering models are usually reported with a random train/test split and a single F1 score. Both hide the two things that matter in production:

1. **Time.** Models are trained on the past and scored on the future, and criminal behaviour changes. A random split leaks future patterns into training.
2. **Decisions.** An alert threshold is a business choice: a missed illicit transaction costs far more than a false alert an analyst has to clear. F1 at 0.5 is not that choice.

The project is also a direct test of a popular claim: that graph neural networks (GNNs) are the natural model for transaction graphs. Here they are compared like-for-like against XGBoost, with and without hand-built graph features.

## Data

[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) (Weber et al., 2019): 203,769 Bitcoin transactions, 234,355 directed payment flows and 49 time steps about two weeks apart. Each transaction has 93 local features and 72 features aggregated over its one-hop neighbourhood; feature meanings are anonymised. 46,564 transactions are labelled (4,545 illicit, 42,019 licit, an illicit share of 9.8%); the remaining 157,205 are unlabelled but still take part in the graph. Edges only connect transactions in the same time step.

The data is not redistributed here. Download the three CSV files (`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`) from Kaggle or the PyTorch Geometric mirror into `data/raw/`.

## Method

| Step | What happens |
|---|---|
| Split | Fit on steps 1–29, validate on 30–34, test on 35–49 (16,670 labelled test transactions, 1,083 illicit). After model selection, every model is **refit on 1–34** with the chosen tree or epoch count. The time step itself is never a feature. |
| XGBoost ×3 | Local features only (93); local + aggregated (165); 165 + engineered graph features. Early stopping on validation AUPRC. |
| Graph features | Label-free, so no target leakage: log in/out/total degree, PageRank, mean neighbour degree, and one- and two-hop neighbourhood means of the local features (`graph_features.py`). |
| GNNs | GraphSAGE (mean aggregator) and GAT (4 heads), two layers, written in plain PyTorch (`gnn.py`, no torch_geometric needed), trained full-batch on the whole graph with a class-weighted loss; unlabelled nodes pass messages but never enter the loss. Epochs chosen by early stopping on validation AUPRC. |
| Seeds | 3 seeds per model; tables report mean ± standard deviation. |
| Thresholds | F1 at the conventional 0.5, plus a **cost-optimal threshold** chosen on validation steps 30–34 only, assuming a missed illicit transaction costs 10× a false alert. The threshold is chosen on **rolling-origin** scores from the deployed model's recipe: each validation step is scored by a model refit with the same tree count and seed on all earlier steps, so the threshold lives on the refit model's score scale. The original procedure (threshold from the 1–29 model applied to the refit model) is reported alongside. |
| Uncertainty | 95% intervals from 2000 bootstrap resamples of the 15 test *time steps* (a time step is the natural unit of dependence), pooled over seeds. |
| Drift | Per-step F1 across the test period, and F1 before vs. after the dark-market shutdown at step 43. |

## Results

XGBoost numbers come from `graphfraud run-all --models xgb_local xgb_all xgb_all_graph --n-jobs 2` on the full dataset (2026-11-19, with the threshold fix below); the GNN numbers come from the previous full run and were **not re-run** (the GAT alone took most of the earlier 115-minute run). Everything is in `docs/results/metrics.json` (each entry records its provenance; log in `docs/results/run_log.txt`). Illicit is the positive class.

| Model | F1 at 0.5 | Precision | Recall | AUPRC | F1 steps 35–42 | F1 steps 43–49 |
|---|---|---|---|---|---|---|
| XGBoost, local features (93) | 0.775 ± 0.004 | 0.860 | 0.706 | 0.787 ± 0.001 | 0.876 | 0.029 |
| XGBoost, local + aggregated (165) | 0.817 ± 0.002 | 0.931 | 0.728 | 0.800 ± 0.001 | 0.911 | 0.029 |
| XGBoost, 165 + engineered graph features | 0.810 ± 0.017 | 0.932 | 0.716 | 0.803 ± 0.014 | 0.904 | 0.023 |
| GraphSAGE (165 features), previous run | 0.687 ± 0.016 | 0.856 | 0.575 | 0.671 ± 0.010 | 0.769 | 0.017 |
| GAT (165 features), previous run | 0.533 ± 0.015 | 0.509 | 0.563 | 0.539 ± 0.022 | 0.602 | 0.029 |

Difference in F1 against XGBoost with local + aggregated features, with 95% time-step bootstrap interval (GNN rows: against the previous run's XGBoost (165), F1 0.816):

| Model | ΔF1 vs. XGBoost (165) | 95% interval |
|---|---|---|
| XGBoost, local features (93) | -0.041 | [-0.055, -0.030] |
| XGBoost, 165 + engineered graph features | -0.007 | [-0.017, 0.001] |
| GraphSAGE (165 features), previous run | -0.130 | [-0.187, -0.083] |
| GAT (165 features), previous run | -0.285 | [-0.389, -0.189] |

Cost-sensitive alerting (cost per test transaction with missed illicit = 10, false alert = 1; mean over 3 seeds). Both thresholds are chosen on validation steps 30–34 only. *Original*: chosen on the 1–29 model's scores but applied to the model refit on 1–34 (a score-scale mismatch). *Rolling-origin*: chosen on one-step-ahead scores of models refit with the deployed recipe, so it matches the evaluated model. Recall and precision are at the respective threshold.

| Model | Cost at 0.5 | Original: cost | recall | precision | Rolling-origin: cost | recall | precision |
|---|---|---|---|---|---|---|---|
| XGBoost, local features (93) | 0.1986 | 0.2148 | 0.766 | 0.444 | 0.2101 | 0.765 | 0.465 |
| XGBoost, local + aggregated (165) | 0.1805 | 0.1966 | 0.755 | 0.569 | 0.1878 | 0.748 | 0.675 |
| XGBoost, 165 + engineered graph features | 0.1878 | 0.1962 | 0.753 | 0.585 | 0.1864 | 0.739 | 0.751 |
| GraphSAGE (165 features), previous run | 0.2826 | 0.2422 | 0.649 | 0.756 | not run | | |
| GAT (165 features), previous run | 0.3199 | 0.3108 | 0.588 | 0.470 | not run | | |

A third variant keeps the 1–29 model itself as the deployed model with its own (self-consistent) threshold. For XGBoost (165) it costs 0.1887 at 0.5 and 0.2165 at its threshold: worse than refitting, and its own validation threshold still does not transfer.

![F1 by model](docs/results/f1_comparison.png)
![Per-step F1](docs/results/per_step_f1.png)

### What the numbers say

1. **Gradient boosting wins, clearly.** XGBoost on the 165 provided features is the best model by F1 (0.817, AUPRC 0.800). GraphSAGE scores 0.130 F1 lower (interval excludes zero) and GAT 0.285 lower. In this setup, message passing over the transaction graph does not beat a tree model that already sees the dataset's one-hop aggregate features.
2. **Neighbourhood information still matters, in tabular form.** Adding the 72 aggregated (one-hop) features to the 93 local ones raises XGBoost F1 by 0.041 (interval [0.030, 0.055]). Graph structure is useful; the gain comes from summarising it as features, not from a GNN.
3. **More engineered graph features did not help.** Adding degree, PageRank and two-hop means changed F1 by -0.007 (interval [-0.017, 0.001], which includes zero) and AUPRC by +0.003: no measurable gain for 191 extra columns. Their effect is also unstable: one seed early-stopped at 53 trees (F1 0.791) while the other two used 365-376 trees (F1 0.819-0.820). An earlier 8-thread run had all three seeds stop at 45-127 trees and a significant loss of 0.021, so that earlier, stronger "it hurts" reading did not survive a re-run.
4. **Every model collapses after the dark-market shutdown.** F1 on test steps 35-42 is 0.60-0.91 depending on the model; on steps 43-49 it is 0.02-0.03 for all five. None of them catches the new illicit behaviour. That is a monitoring and retraining problem, not a modelling one, and a random train/test split would never reveal it.
5. **A cost threshold chosen on the past still does not pay off on the future, but the original setup overstated the loss.** The original procedure picked the threshold on the 1–29 model and applied it to the refit model; for XGBoost (165) that raised test cost from 0.1805 (at 0.5) to 0.1966. Choosing the threshold on the deployed recipe's own out-of-time scores (rolling origin) cut that excess by more than half, to 0.1878, with similar recall (0.748) and much higher precision (0.675 vs. 0.569). It was cheaper than the original threshold in 8 of 9 XGBoost model-seed runs. But it beat 0.5 in only 2 of 9 runs (both for the graph-feature model, mean 0.1864 vs. 0.1878) and never for XGBoost (165): once the score scale is fixed, the expected cost saving from a 10:1 threshold is still not realised, because the post-shutdown steps break the conditions it was tuned on. For GraphSAGE (previous run, original procedure) the threshold did lower cost (0.283 to 0.242). Alert thresholds need recalibrating on recent data, and tuning them on a model other than the deployed one adds avoidable error.

## Limitations

- One dataset, anonymised features, and labels only for about 23% of transactions; illicit labels come from heuristics described in the Elliptic paper, not from a full audit.
- The GNNs use one reasonable configuration (two layers, hidden size 64 for GraphSAGE, 4×16 for GAT) without a hyperparameter search; a tuned or temporal GNN might narrow the gap. XGBoost was not tuned either beyond early stopping.
- The cost ratio of 10:1 is an assumption chosen to show the method; real ratios depend on the institution.
- The rolling-origin threshold is computed on validation steps whose tree count was also selected on those steps, so it is mildly optimistic; it never touches the test period.
- XGBoost results depend on the number of threads (`hist` reduces histograms in a thread-dependent order, which changes the early-stopped tree count). The XGBoost numbers here come from a 2-thread run; an earlier 8-thread run gave F1 0.816 for the 165-feature model.
- The bootstrap treats time steps as exchangeable, which understates uncertainty if drift is systematic; with 15 test steps the intervals are wide.

## How to run

```bash
pip install -e ".[dev]"          # CPU-only PyTorch is enough
graphfraud describe --raw-dir data/raw
graphfraud run-all --raw-dir data/raw --out-dir docs/results   # writes metrics.json and figures
python docs/brief/make_figures.py                                 # figures for the brief
pytest && ruff check src tests                                    # synthetic-graph tests, no download needed
```

The earlier full run took about 115 minutes on a shared 14-core laptop CPU, mostly the GAT. The rolling-origin threshold refits the final recipe once per validation step (5 extra fits per model and seed); the XGBoost-only re-run with 2 threads took about 70 minutes. `--no-rolling-threshold` skips it, `--n-jobs` caps threads.

## Repository layout

```
src/graphfraud/
  data.py            Elliptic CSV loader
  split.py           temporal split (fit / val / train / test, pre/post shutdown)
  graph_features.py  label-free degree, PageRank, multi-hop means
  boosting.py        XGBoost with validation early stopping and refit
  gnn.py             GraphSAGE and GAT in plain PyTorch
  metrics.py         illicit metrics, cost-optimal threshold, time-step bootstrap
  pipeline.py        end-to-end experiment
  plots.py, cli.py, testing.py (synthetic Elliptic-format data)
docs/results/        metrics.json, figures, run log
docs/brief/          project brief (brief.pdf)
```

## Reference

Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* KDD Workshop on Anomaly Detection in Finance. arXiv:1908.02591.

## License

MIT, see [LICENSE](LICENSE). The Elliptic data is subject to its own terms.

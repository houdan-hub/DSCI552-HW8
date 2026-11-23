# DSCI-552 Homework 8 — Supervised / Semi-Supervised / Unsupervised Learning and Active Learning

Tianbo Hou · GitHub: houdan-hub · USC ID: 8943381210

## What this is

Machine-learning homework for DSCI 552 (USC). Part 1 uses the
[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
data set (569 cases, 30 features, malignant = positive) to compare an L1-penalised
SVM, self-training with the farthest-boundary rule, k-means and spectral clustering
over 30 paired Monte-Carlo splits with 20% of each class held out. Part 2 uses the
[Banknote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
data set (900 train / 472 test) to compare passive and active SVM sampling with
50 repetitions and 90 successively larger pools each.

## Contents

- `Hou_Tianbo_HW8.ipynb` — full report: code, executed outputs, figures and written conclusions.
- `data/wdbc.data`, `data/wdbc.names` — the UCI breast-cancer data and its description.
- `data/data_banknote_authentication.txt` — the UCI banknote data (unmodified).
- `requirements.txt` — Python packages needed to re-run the notebook.

## How to run

```bash
pip install -r requirements.txt
jupyter execute Hou_Tianbo_HW8.ipynb
```

`jupyter execute` runs the notebook headlessly through `nbclient` and writes the
outputs back in place, which is how the committed outputs were produced. A fresh
clone runs as is: all paths are relative to the repository root and every seed is
fixed. A full run takes about 30 minutes; the 50 x 90 SVM fits in part 2 are the
slowest cells.

## Main results — part 1 (mean over 30 test splits, 113 cases each)

| method | accuracy | precision | recall | F1 | AUC |
|---|---|---|---|---|---|
| L1-SVM, supervised | 0.9696 | 0.9798 | 0.9381 | 0.9580 | 0.9914 |
| L1-SVM, self-training | 0.9693 | 0.9768 | 0.9405 | 0.9578 | 0.9910 |
| k-means, cluster majority | 0.9088 | 0.9269 | 0.8198 | 0.8688 | 0.9730 |
| Spectral clustering | 0.9100 | 0.9262 | 0.8246 | 0.8712 | 0.9524 |

- The penalty parameter $C$ is re-tuned by stratified 5-fold cross-validation in
  every run and at every pool size; the supervised median is $C = 0.3162$.
- Self-training follows the homework's literal farthest-from-boundary rule. It
  adds 228 pseudo-labels per run but is effectively tied with full supervision:
  the least certain point carries little new information.
- k-means uses 20 random restarts per run and keeps the lowest inertia. Spectral
  clustering uses an RBF affinity; $\gamma = 1$ collapses to a 567/2 split, so a
  grid search selects $\gamma = 10^{-5}$, the closest available balance to 357/212.
- The unsupervised methods lose accuracy mainly through malignant recall, not
  through ranking: their AUCs are still 0.9730 and 0.9524.

## Main results — part 2 (mean test error over 50 runs)

| pool size | passive | active |
|---|---|---|
| 10 | 0.1269 | 0.1269 |
| 50 | 0.0404 | 0.0175 |
| 100 | 0.0282 | 0.0126 |
| 500 | 0.0167 | 0.0125 |
| 900 | 0.0126 | 0.0127 |

The active sampler is better at 88 of the 90 pool sizes, with a minimum of 0.0123
at 870 points. Its advantage is largest for small pools and vanishes at the end,
because when almost all 900 training points are already in the pool the choice of
the next ten points can no longer matter.

## References

- "Breast Cancer Wisconsin (Diagnostic) Data Set." *UCI Machine Learning Repository*.
- "Banknote Authentication Data Set." *UCI Machine Learning Repository*.
- Tibshirani, Robert. "Regression Shrinkage and Selection via the Lasso." *JRSS-B*, 58(1), 1996, 267-288.
- Ng, Andrew Y., Michael I. Jordan, and Yair Weiss. "On Spectral Clustering: Analysis and an Algorithm." *NIPS* 14, 2002, 849-856.
- von Luxburg, Ulrike. "A Tutorial on Spectral Clustering." *Statistics and Computing*, 17(4), 2007, 395-416.
- Tong, Simon, and Daphne Koller. "Support Vector Machine Active Learning with Applications to Text Classification." *JMLR* 2, 2001, 45-66.
- Pedregosa, Fabian, et al. "Scikit-learn: Machine Learning in Python." *JMLR* 12, 2011, 2825-2830.

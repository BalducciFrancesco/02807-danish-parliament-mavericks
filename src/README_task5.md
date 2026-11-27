# Task 5 — Evaluation & validation

Hand-implemented ARI and NMI, a permutation null model, stability under
resampling, and a one-table method comparison. Method-agnostic: every
function takes label vectors, so any clustering can be plugged in.

## Usage

```python
import pandas as pd
from src.evaluation import compare_methods

gt = pd.read_csv("outputs/ground_truth_modal.csv").sort_values("actor")

labels = {
    "louvain": louvain_labels,
    "dbscan":  dbscan_labels,
}
compare_methods(labels, gt["party_label"].values)
```

**Label vectors must be sorted by `actor`** so they align with the ground
truth. Mismatched order produces wrong scores that still look plausible.

Returns one row per method: `n_clusters`, `n_noise`, `ari`, `nmi`,
`nmi_adjusted`, `ari_p`, `nmi_p`, `ari_stability_mean`, `ari_ci_low`,
`ari_ci_high`.

## Functions

| function | what it does |
|---|---|
| `adjusted_rand_index(a, b)` | ARI from the contingency table |
| `normalized_mutual_info(a, b, average=)` | NMI, `arithmetic`/`geometric`/`max` |
| `permutation_test(labels, gt, metric=, n=)` | label-shuffle null, empirical p-value |
| `stability(labels, gt, frac=, n=)` | subsample MPs, mean/sd/95% interval |
| `compare_methods(dict, gt)` | tidy comparison table |

Both metrics validated against `sklearn.metrics` to <1e-9 on identical
labels, random labels, and a hand-worked example. sklearn is used only as a
correctness check, never inside the implementations.

## Ground truth

`party` in `votes.csv` is time-varying (party on the date of the vote), but
ARI and NMI need exactly one label per MP. Three variants are provided:

| file | MPs | classes | singletons |
|---|---|---|---|
| `ground_truth_first.csv` | 135 | 13 | 1 |
| `ground_truth_modal.csv` | 135 | 14 | 3 (all UFG) |
| `ground_truth_noswitch.csv` | 124 | 11 | 0 |

**Use `modal`** (party held for the most votes). `first` assigns 4 MPs to
Nye Borgerlige, which dissolved in January 2024, so it labels them with a
party that did not exist for most of the period.

Singletons cannot be recovered by any clustering method, so they cap the
achievable ARI. `noswitch` has none, which is why it scores highest — that
is structural, not evidence of better clustering.

Each file has a `ufg_flag` column for the independents, so they can be kept
as singletons, collapsed to one "independent" label, or excluded.

Built by `src/build_labels.py`, which applies the group's two <50% filters
(tenure and attendance).

## Reading the results

**NMI is not chance-corrected.** Randomly shuffled labels score well above
zero on this ground truth, and the inflation grows with the number of
classes. ARI is chance-corrected and reads directly. So use `ari` and
`nmi_adjusted`; raw `nmi` is not interpretable on its own. `permutation_test`
reports the actual null mean for a given label set.

DBSCAN noise (`-1`) is excluded from `n_clusters` and reported separately as
`n_noise`.

`stability` subsamples **without** replacement by default. Bootstrapping with
replacement inflates ARI by ~0.03, because a duplicated MP appears in both
label vectors and agrees with itself by construction.

## Scope

The two <50% filters retain **135 of 235 actors**. The removal rate falls
with party size, and all Greenlandic and Faroese parties (Naleraq, Inuit
Ataqatigiit, Siumut, Sambandsflokkurin, Javnaðarflokkurin) are fully
removed — their MPs cast very few votes. Results therefore describe
long-serving mainland MPs, not the full Folketing.

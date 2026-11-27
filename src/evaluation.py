"""Adjusted Rand Index and Normalised Mutual Information.

Both computed from the contingency table. numpy only, no sklearn.
"""

import numpy as np


def adjusted_rand_index(labels_a, labels_b):
    """Adjusted Rand Index between two integer label vectors.

    ARI measures agreement between two partitionsings, corrected for
    chance: ARI = 0 means agreement no better than random, 1 means
    perfect agreement, and values can be negative.

    Formula
    -------
    Let the contingency table be n_ij = number of items sharing label i
    in ``labels_a`` and label j in ``labels_b``, with row sums a_i and
    column sums b_j, and n = sum(n_ij). Then::

        ARI = ( sum_ij C(n_ij, 2) - [sum_i C(a_i, 2) * sum_j C(b_j, 2)] / C(n, 2) )
            / ( [sum_i C(a_i, 2) + sum_j C(b_j, 2)] / 2
                - [sum_i C(a_i, 2) * sum_j C(b_j, 2)] / C(n, 2) )

    where C(k, 2) = k * (k - 1) / 2.

    Parameters
    ----------
    labels_a, labels_b : array-like of int
        Two label vectors of equal length. Labels are remapped to dense
        integers internally, so non-contiguous or arbitrary integer
        encodings are handled correctly.

    Returns
    -------
    float
        The adjusted Rand index.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError("labels_a and labels_b must have the same shape")
    if a.size == 0:
        return 1.0

    # Remap labels to dense integers so the contingency table is compact.
    a_keys = np.unique(a)
    b_keys = np.unique(b)
    a_map = {k: i for i, k in enumerate(a_keys)}
    b_map = {k: i for i, k in enumerate(b_keys)}
    a_idx = np.array([a_map[v] for v in a])
    b_idx = np.array([b_map[v] for v in b])

    n = a.size
    nij = np.zeros((len(a_keys), len(b_keys)), dtype=np.int64)
    np.add.at(nij, (a_idx, b_idx), 1)

    ai = nij.sum(axis=1)          # row sums
    bj = nij.sum(axis=0)          # column sums

    comb2 = lambda k: k * (k - 1) // 2

    sum_nij = comb2(nij).sum()
    sum_ai = comb2(ai).sum()
    sum_bj = comb2(bj).sum()
    c_n = comb2(n)

    if c_n == 0:
        return 1.0

    expected = sum_ai * sum_bj / c_n
    max_index = (sum_ai + sum_bj) / 2.0

    denom = max_index - expected
    if denom == 0:
        return 0.0

    return float((sum_nij - expected) / denom)


def normalized_mutual_info(labels_a, labels_b, average='arithmetic'):
    """Normalised Mutual Information between two integer label vectors.

    NMI measures the shared information between two partitionings,
    normalised so 1 means perfect agreement and 0 means none.

    Formulas (natural log)
    ----------------------
    Let p_ij = n_ij / n be the joint probability of label pair (i, j),
    p_i = sum_j p_ij and p_j = sum_i p_ij the marginals.

    Mutual information::

        MI = sum_ij p_ij * log(p_ij / (p_i * p_j))      (p_ij > 0)

    Entropies::

        H_a = -sum_i p_i * log(p_i)     (p_i > 0)
        H_b = -sum_j p_j * log(p_j)     (p_j > 0)

    Normalisation, selected by ``average``:

    - 'arithmetic':  NMI = 2 * MI / (H_a + H_b)
    - 'geometric':   NMI = MI / sqrt(H_a * H_b)
    - 'max':         NMI = MI / max(H_a, H_b)

    Degenerate case: if H_a == 0 or H_b == 0, one partition puts every
    point in a single cluster, so there is no shared information to
    measure. Return 0.0 rather than dividing by zero.

    Parameters
    ----------
    labels_a, labels_b : array-like of int
        Two label vectors of equal length.
    average : str, default 'arithmetic'
        One of 'arithmetic', 'geometric', 'max'.

    Returns
    -------
    float
        The normalised mutual information.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError("labels_a and labels_b must have the same shape")
    if a.size == 0:
        return 0.0

    a_keys = np.unique(a)
    b_keys = np.unique(b)
    a_map = {k: i for i, k in enumerate(a_keys)}
    b_map = {k: i for i, k in enumerate(b_keys)}
    a_idx = np.array([a_map[v] for v in a])
    b_idx = np.array([b_map[v] for v in b])

    n = a.size
    nij = np.zeros((len(a_keys), len(b_keys)), dtype=np.int64)
    np.add.at(nij, (a_idx, b_idx), 1)
    p = nij / n

    pi = p.sum(axis=1)
    pj = p.sum(axis=0)

    # Mutual information: only over cells with p_ij > 0.
    mask = p > 0
    mi = (p[mask] * np.log(p[mask] / (pi[:, None] * pj)[mask])).sum()

    # Entropies, only over non-empty clusters.
    ha = -(pi[pi > 0] * np.log(pi[pi > 0])).sum()
    hb = -(pj[pj > 0] * np.log(pj[pj > 0])).sum()

    if ha == 0 or hb == 0:
        # One partition is a single cluster: no information shared.
        return 0.0

    if average == 'arithmetic':
        denom = ha + hb
        return float(2.0 * mi / denom) if denom else 0.0
    elif average == 'geometric':
        denom = np.sqrt(ha * hb)
        return float(mi / denom) if denom else 0.0
    elif average == 'max':
        denom = max(ha, hb)
        return float(mi / denom) if denom else 0.0
    else:
        raise ValueError("average must be one of 'arithmetic', 'geometric', 'max'")


def permutation_test(labels, ground_truth, metric='ari', n=1000, seed=0):
    """Permutation null model for ARI or NMI.

    The observed score is compared against a null distribution built by
    shuffling the assignment of ``ground_truth`` labels to points. Because
    the labels are permuted *in place* rather than resampled, the class-size
    distribution is preserved exactly: the null keeps both marginals of the
    contingency table and destroys only the correspondence between the two
    partitions. This is the standard "fixed marginals" null for clustering
    agreement.

    Empirical p-value::

        p = (1 + count(null >= observed)) / (1 + n)

    The +1 in the numerator avoids reporting p = 0 when the observed score
    is the maximum of the null distribution. An empirical p-value of exactly
    0 would be impossible to distinguish from "no null draw ever reached
    it", which is an artefact of finite n; the +1 in the denominator makes
    the smallest possible p-value 1/(n+1) (roughly 0.001 for n=1000), a
    conservative floor that keeps the test honest.

    Parameters
    ----------
    labels, ground_truth : array-like of int
        Two label vectors of equal length; ``ground_truth`` is the one
        permuted.
    metric : str, default 'ari'
        'ari' -> adjusted_rand_index, 'nmi' -> normalized_mutual_info.
    n : int, default 1000
        Number of permutations.
    seed : int, default 0
        Seed for the RNG.

    Returns
    -------
    dict
        Keys: 'observed', 'null_mean', 'null_std', 'p_value', 'null_dist'
        (the full null distribution as a numpy array).
    """
    labels = np.asarray(labels)
    ground_truth = np.asarray(ground_truth)
    if labels.shape != ground_truth.shape:
        raise ValueError("labels and ground_truth must have the same shape")

    score = adjusted_rand_index if metric == 'ari' else (
        normalized_mutual_info if metric == 'nmi' else None)
    if score is None:
        raise ValueError("metric must be 'ari' or 'nmi'")

    observed = score(labels, ground_truth)

    rng = np.random.default_rng(seed)
    null_dist = np.empty(n, dtype=float)
    for i in range(n):
        perm = rng.permutation(ground_truth)
        null_dist[i] = score(labels, perm)

    null_mean = float(null_dist.mean())
    null_std = float(null_dist.std(ddof=1))
    p_value = float((1.0 + (null_dist >= observed).sum()) / (1.0 + n))

    return {
        'observed': float(observed),
        'null_mean': null_mean,
        'null_std': null_std,
        'p_value': p_value,
        'null_dist': null_dist,
    }


def stability(labels, ground_truth, metric='ari', frac=0.8, n=500,
              seed=0, replace=False):
    """Stability of a metric under resampling of points.

    This measures METRIC stability: it takes the label vectors as given,
    subsamples (or bootstraps) the set of points, and recomputes the metric
    on the subset. It does NOT re-run the clustering algorithm on each
    resample; therefore it measures how robust the *agreement score* is to
    the composition of the test set, not how stable the *clustering itself*
    is. To measure cluster stability you would need to re-cluster on each
    resample.

    Subsampling without replacement (replace=False, default) draws a random
    subset of size floor(frac * n) each iteration. This is the recommended
    mode because it keeps the label vectors clean.

    Subsampling with replacement (replace=True) is a bootstrap. WARNING:
    with replacement, duplicated points appear in BOTH label vectors
    simultaneously. Every duplicate pair agrees by construction, which
    artificially inflates agreement scores — ARI can be biased upward
    substantially. This mode is provided for comparison only; prefer
    replace=False.

    Parameters
    ----------
    labels, ground_truth : array-like of int
        Two label vectors of equal length.
    metric : str, default 'ari'
        'ari' -> adjusted_rand_index, 'nmi' -> normalized_mutual_info.
    frac : float, default 0.8
        Fraction of points to keep per draw.
    n : int, default 500
        Number of resamples.
    seed : int, default 0
        RNG seed.
    replace : bool, default False
        If True, bootstrap (with replacement); if False, subsample (without
        replacement).

    Returns
    -------
    dict
        Keys: 'mean', 'std', 'ci_low', 'ci_high', 'scores' (numpy array of
        length n with the metric on each resample).
    """
    labels = np.asarray(labels)
    ground_truth = np.asarray(ground_truth)
    if labels.shape != ground_truth.shape:
        raise ValueError("labels and ground_truth must have the same shape")

    score = adjusted_rand_index if metric == 'ari' else (
        normalized_mutual_info if metric == 'nmi' else None)
    if score is None:
        raise ValueError("metric must be 'ari' or 'nmi'")

    n_points = labels.shape[0]
    k = max(1, int(frac * n_points))
    if k < 2:
        raise ValueError("frac too small: need at least 2 points")

    rng = np.random.default_rng(seed)
    scores = np.empty(n, dtype=float)
    for i in range(n):
        if replace:
            idx = rng.integers(0, n_points, size=k)
        else:
            idx = rng.choice(n_points, size=k, replace=False)
        scores[i] = score(labels[idx], ground_truth[idx])

    return {
        'mean': float(scores.mean()),
        'std': float(scores.std(ddof=1)),
        'ci_low': float(np.percentile(scores, 2.5)),
        'ci_high': float(np.percentile(scores, 97.5)),
        'scores': scores,
    }


import pandas as pd


def compare_methods(label_dict, ground_truth, n_perm=1000, n_boot=500, seed=0):
    """Compare multiple clustering methods using metrics and stability.

    Computes ARI and NMI for each method, adjusts NMI for chance using a
    permutation test, and calculates ARI stability via subsampling.

    Adjusted NMI formula::

        NMI_adj = (NMI_obs - NMI_null) / (1 - NMI_null)

    This is necessary because raw NMI is not chance-corrected and tends
    to have a high baseline (e.g., ~0.50 on small datasets with many classes),
    whereas ARI is inherently chance-corrected.

    Parameters
    ----------
    label_dict : dict
        Map of method_name -> labels vector.
    ground_truth : array-like
        The true labels.
    n_perm, n_boot : int
        Permutations and bootstrap iterations.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for name, labels in label_dict.items():
        labels = np.asarray(labels)
        ari_obs = adjusted_rand_index(labels, ground_truth)
        nmi_obs = normalized_mutual_info(labels, ground_truth)

        # Null distributions
        perm_ari = permutation_test(labels, ground_truth, metric='ari', n=n_perm, seed=seed)
        perm_nmi = permutation_test(labels, ground_truth, metric='nmi', n=n_perm, seed=seed)

        # NMI adjustment
        nmi_null = perm_nmi['null_mean']
        nmi_adj = (nmi_obs - nmi_null) / (1.0 - nmi_null) if nmi_null < 1.0 else float ('nan')

        # Stability
        stab = stability(labels, ground_truth, metric='ari', n=n_boot, seed=seed)

        rows.append({
            'method': name,
            'n_clusters': np.unique(labels[labels != -1]).size,
            'n_noise': int((labels == -1).sum()),
            'ari': ari_obs,
            'nmi': nmi_obs,
            'nmi_adjusted': nmi_adj,
            'ari_p': perm_ari['p_value'],
            'nmi_p': perm_nmi['p_value'],
            'ari_stability_mean': stab['mean'],
            'ari_ci_low': stab['ci_low'],
            'ari_ci_high': stab['ci_high'],
        })

    df = pd.DataFrame(rows).sort_values('ari', ascending=False)
    return df.round(4)

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from evaluation import permutation_test

for name in ["first", "modal", "noswitch"]:
    df = pd.read_csv(f"outputs/ground_truth_{name}.csv").sort_values("actor")
    gt = df["party_label"].values

    rng = np.random.default_rng(0)
    random_labels = rng.permutation(gt)

    print(f"--- {name}: {len(gt)} MPs, {len(set(gt))} classes ---")
    for metric in ["ari", "nmi"]:
        r = permutation_test(random_labels, gt, metric=metric, n=1000)
        print(f"  {metric:>3}  observed {r['observed']:+.4f}   "
              f"null_mean {r['null_mean']:+.4f}   "
              f"null_sd {r['null_std']:.4f}   p {r['p_value']:.4f}")
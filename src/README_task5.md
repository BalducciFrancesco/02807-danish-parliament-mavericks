# Task 5 — Evaluation

## Usage
```python
from src.evaluation import compare_methods
import pandas as pd

gt = pd.read_csv("outputs/ground_truth_modal.csv").sort_values("actor")
labels = {"louvain": louvain_labels, "dbscan": dbscan_labels}
compare_methods(labels, gt["party_label"].values)

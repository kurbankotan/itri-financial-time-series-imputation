"""Check every prespecified 5-feature fold/mask shape before benchmarking."""

import json
from pathlib import Path

from masking import run_lengths, strict_block_mask


HERE = Path(__file__).resolve().parent
audit = json.loads((HERE.parent / "metadata" / "received_snapshot_audit.json").read_text())
folds = audit["assets"][0]["fold_counts_train_val_test"]
checks = 0
max_observed = 0
for fold, counts in folds.items():
    for partition, rows in zip(("train", "validation", "test"), counts):
        for block in (1, 2, 3, 5, 10, 20):
            for rate in (0.10, 0.20, 0.30):
                for seed in (11, 23, 37):
                    mask = strict_block_mask((rows, 5), rate, block, seed,
                                             edge_guard=20)
                    target = round(rate * (rows - 40) * 5)
                    lengths = run_lengths(mask)
                    assert int(mask.sum()) == target
                    assert int(lengths.max()) <= block
                    max_observed = max(max_observed, int(lengths.max()))
                    checks += 1
print({"folds": list(folds), "mask_configurations_checked": checks,
       "max_observed_run": max_observed, "status": "PASS"})

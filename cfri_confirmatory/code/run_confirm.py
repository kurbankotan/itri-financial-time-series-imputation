#!/usr/bin/env python3
"""Command-line driver of the CFRI confirmatory study (fold D). Steps, in order:

  python run_confirm.py --root DIR data                 acquire + QC, writes data_receipt.json (score-blind)
  python run_confirm.py --root DIR cpu [--workers N]    CPU methods, feature-wise and whole-row masks
  python run_confirm.py --root DIR saits --variant V2 --mask-type feature [--smoke 1]
  python run_confirm.py --root DIR analyze              prespecified analysis + results ZIP (no raw prices)

Every step records the SHA-256 of the code files it ran with (code_sha256.json); later steps refuse to run if
the code changed after an earlier step had started.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROTOCOL_ID = "CFRI-confirmatory-protocol-v1"


def code_hashes() -> dict:
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(CODE_DIR.glob("*.py"))}


def check_code(root: Path) -> None:
    path, now = root / "code_sha256.json", code_hashes()
    if path.exists():
        before = json.loads(path.read_text())
        if before["files"] != now:
            raise SystemExit("code differs from the code used by an earlier step of this run; stopping")
    else:
        path.write_text(json.dumps({"protocol": PROTOCOL_ID, "files": now}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    sub = ap.add_subparsers(dest="step", required=True)
    dat = sub.add_parser("data")
    dat.add_argument("--accept-provider-failures", action="store_true",
                     help="finalize the receipt although some downloads failed repeatedly (score-blind)")
    cpu = sub.add_parser("cpu")
    cpu.add_argument("--workers", type=int, default=0)
    cpu.add_argument("--mask-types", default="feature,row")
    sa = sub.add_parser("saits")
    sa.add_argument("--variant", required=True, choices=["V1", "V2"])
    sa.add_argument("--mask-type", required=True, choices=["feature", "row"])
    sa.add_argument("--smoke", type=int, default=0)
    sa.add_argument("--threads", type=int, default=2)
    sub.add_parser("analyze")
    a = ap.parse_args()
    if a.step in ("cpu", "saits"):   # bounded BLAS threads per process (several processes may run at once)
        n = "1" if a.step == "cpu" else str(a.threads)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = n
    sys.path.insert(0, str(CODE_DIR))
    root = a.root
    root.mkdir(parents=True, exist_ok=True)
    check_code(root)
    import confirm_data as D
    if a.step == "data":
        receipt = D.acquire(root)
        print(f"eligible assets: {receipt['eligible_count']} of 34; excluded: "
              f"{[e['ticker'] for e in receipt['excluded']] or 'none'} (reasons in data_receipt.json)")
        failed = D.provider_failures(receipt)
        if failed and not receipt.get("provider_failures_accepted"):
            if a.accept_provider_failures:
                D.accept_provider_failures(root)
                print("provider failures accepted as exclusions:", failed)
            else:
                print(f"Download failed for {failed} after three attempts. No method has run. Run the data step "
                      "again later; if the failure persists, rerun it with --accept-provider-failures.")
                raise SystemExit(2)
        return
    if not (root / D.RECEIPT).exists():
        raise SystemExit("run the data step first")
    receipt = json.loads((root / D.RECEIPT).read_text())
    if D.provider_failures(receipt) and not receipt.get("provider_failures_accepted"):
        raise SystemExit("some downloads failed; rerun the data step first (see data_receipt.json)")
    if a.step == "cpu":
        import confirm_cpu
        frames = D.load_frames(root)
        log = confirm_cpu.run_cpu(root, frames, tuple(a.mask_types.split(",")), a.workers or None)
        if not log["complete"]:
            raise SystemExit("CPU step incomplete; run it again to resume")
        print("CPU step complete:", log["shards"], "shards")
    elif a.step == "saits":
        import confirm_saits
        frames = D.load_frames(root)
        if a.smoke:
            confirm_saits.run_saits(root, dict(list(frames.items())[:1]), a.variant, a.mask_type,
                                    smoke_conditions=a.smoke, threads=a.threads)
        else:
            confirm_saits.run_saits(root, frames, a.variant, a.mask_type, threads=a.threads)
            print(f"SAITS {a.variant} {a.mask_type} complete")
    elif a.step == "analyze":
        import confirm_analyze
        confirm_analyze.analyze(root)    # needs only the receipt and the metric shards, not the raw data
        zpath = confirm_analyze.export(root, CODE_DIR)
        print((root / "fold_D_report.md").read_text())
        print("results ZIP:", zpath)


if __name__ == "__main__":
    main()

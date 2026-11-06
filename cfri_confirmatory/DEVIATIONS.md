# Deviations from CFRI confirmatory protocol v1

Protocol v1 was frozen on 27 October 2026. Every deviation is recorded here with its date, reason and effect,
before the affected results are reported. Analyses affected by a deviation made after data access are labeled
exploratory.

## 1. Public time stamp of the freeze (recorded 27 October 2026)

**What the protocol said.** Sections 1 and 12 state that the protocol and code would be committed to this public
repository before the 2026 data were downloaded, so that the commit time would time-stamp the freeze.

**What happened.** The confirmatory run was started before that commit. The data were acquired at 07:10:10 to
07:10:18 UTC on 27 October 2026. This folder, including the results, was committed to the repository after the run.

**Evidence that the analysis was fixed before data access.**

1. Google Drive server timestamps of the run folder show this order:
   - The Colab notebook was created at 07:07:52 UTC. It embeds the six run files and their SHA-256 values.
   - `code_sha256.json` was created at 07:10:07 UTC.
   - The first data request came after that. `data_receipt.json` was written at 07:10:18 UTC.
2. The hashes in `code_sha256.json` equal those listed in Section 11 of the protocol.
3. `confirm_analyze.py` is one of the hashed files. It implements the primary hypotheses, the decision rule and the
   secondary analyses.

**Effect.** Data, code, conditions and analyses are unchanged; all results were produced as prespecified. The
confirmatory evaluation is reported as prespecified with a time-stamped code freeze, not as a publicly preregistered
study.

No other deviations.

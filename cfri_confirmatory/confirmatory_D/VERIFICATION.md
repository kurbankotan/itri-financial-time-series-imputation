# Verification of the confirmatory run (fold D)

**Results file.** `CFRI_fold_D_results.zip`, SHA-256
`3a3e789b6352ded56d9adc0f15acb358e5df5b43321d88c19e7e496920afbc23`. It is the ZIP produced by the analysis step of
`notebooks/CFRI_confirmatory_fold_D_Colab.ipynb` on 27 October 2026. It contains metrics, receipts and code only;
raw prices are not included. The other files in this folder are copies extracted from it.

## Run environment

The run used Google Colab with Python 3.13.16, NumPy 2.1.3, pandas 2.2.3, SciPy 1.18.1, scikit-learn 1.6.1, PyPOTS 1.5
and PyTorch 2.11.0+cu130. The hardware was 12 vCPUs and an NVIDIA A100-SXM4-40GB. Each SAITS job ran in one session,
with the four jobs running at the same time.

## Checks

1. **Code.** The six run files in the ZIP (`code/`) and the hashes in `code_sha256.json` equal the SHA-256 values
   listed in Section 11 of the protocol. The run therefore used the frozen code.
2. **Data.** `data_receipt.json` records the following:
   - All 34 ETFs passed the quality rules, with no exclusions and no accepted provider failures.
   - Each asset has 2,953 sessions from 2015-01-02 to 2026-09-30, including 187 sessions in 2026.
   - XNYS coverage is complete in every year, with no absent sessions and no rows outside the calendar.
   - The 34 downloads ran from 07:10:10 to 07:10:18 UTC on 27 October 2026.
3. **Completeness.**
   - There are 68 CPU shards: 34 assets × 2 mask types, each with 51 feasible conditions, giving 3,468 conditions with
     0 errors.
   - There are 136 SAITS shards: 34 assets × V1 and V2 × 2 mask types, each covering all 51 conditions, with 0 errors.
   - Under both mask types, the block length 20 at rate 0.10 pair was infeasible for every asset, as stated in
     Appendix A of the protocol.
4. **Independent re-analysis.** `confirm_analyze.analyze` from the frozen code was run on the extracted shards.
   - It reproduced `fold_D_results.json` with zero differences in every field except the run-metadata echo.
   - It reproduced `fold_D_report.md` and both asset-level tables byte for byte.
5. **Time order.** Google Drive server timestamps (UTC, 27 October 2026) of the run folder show the following order:
   - 07:07:52: the Colab notebook was created. It embeds the six run files and their SHA-256 values.
   - 07:10:07: `code_sha256.json` was created. It holds the same hashes.
   - 07:10:18: `data_receipt.json` was written. The first data request came after 07:10:07.

   The public commit of this folder was made after the run (see `../DEVIATIONS.md`).

# CFRI simple held-out-asset evaluation

Study: `cfri-heldout-assets-v1`. Classification: **locally frozen asset holdout**. The target ETFs were not in the earlier 34-ETF development study, but the January–September 2026 market period is shared. This is not a future-period replication.

Local source freeze UTC: `2026-10-29T04:21:09.197388+00:00`; embedded bundle SHA-256: `8f7a5176ea48339980379ca1b802205d5a8b05e129ef96e9ef06c2a6ad776979`. GitHub publication after this run does not establish a public pre-run timestamp.

Eligible target ETFs: 16/16; eligible fixed peers: 34/34. Required ETF wins: 12/16.

The endpoint averages five feature NRMSEs per condition, conditions within ETF, then ETFs equally. Differences are proposed method minus comparator; lower is better.

## feature masks

Verified conditions per method: 816.

| Method | NRMSE | Price NRMSE | Volume NRMSE | OHLC violations |
|---|---:|---:|---:|---:|
| cfri_joint | 0.13887 | 0.05091 | 0.49070 | 0.0000 |
| cfri_old_proj | 0.13968 | 0.05192 | 0.49070 | 0.0000 |
| saits_joint | 0.13990 | 0.04684 | 0.51215 | 0.0000 |

## row masks

Verified conditions per method: 816.

| Method | NRMSE | Price NRMSE | Volume NRMSE | OHLC violations |
|---|---:|---:|---:|---:|
| cfri_joint | 0.17133 | 0.09020 | 0.49584 | 0.0000 |
| cfri_peer_marketwide_joint | 0.17133 | 0.09020 | 0.49584 | 0.0000 |
| cfri_peer_target_joint | 0.15714 | 0.07246 | 0.49584 | 0.0000 |
| saits_joint | 0.26168 | 0.18593 | 0.56465 | 0.0000 |
| saits_peer_joint | 0.25771 | 0.18098 | 0.56465 | 0.0000 |

## Prespecified hypotheses

Each succeeds only when the equal-ETF mean error is lower and at least 75% of eligible ETFs have a lower mean error.

| Hypothesis | Comparison | Difference | Descriptive 95% ETF bootstrap interval | ETF wins | Decision |
|---|---|---:|---:|---:|---|
| H1 | cfri_joint vs saits_joint (feature) | -0.00103 | [-0.00511, +0.00261] | 9/16 | not met |
| H2 | cfri_peer_target_joint vs cfri_joint (row) | -0.01419 | [-0.02447, -0.00606] | 16/16 | met |
| H3 | cfri_peer_target_joint vs saits_peer_joint (row) | -0.10058 | [-0.12040, -0.08335] | 16/16 | met |

H1 uses target-only information for both methods. H2 measures the added peer bridge on whole-row outages. H3 gives both methods the same peer pool and bridge, with each gain selected on 2025 validation labels.

## Interpretation limits

- The 34 peer ETFs and the 2026 calendar were used in the earlier study; only target assets are held out here. A future-period replication is still needed.
- Target-specific row results assume other ETFs' same-day closes are available. The market-wide stress case hides those closes and must fall back to no-peer CFRI.
- The 16 ETFs share market shocks; the asset bootstrap is descriptive, not an independent-sample confidence statement. Repeated masks are not independent samples.
- This is offline gap imputation using observed values on both sides of a gap, not forecasting.
- Raw Yahoo OHLCV tables are retained privately and excluded from this ZIP.

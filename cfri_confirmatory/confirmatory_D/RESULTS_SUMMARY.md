# CFRI confirmatory results, fold D (test period 2 January to 30 September 2026)

**Design.** 34 ETFs × 51 feasible conditions per mask type. CFRI is `cfri_fixed_proj`. The endpoint is the
asset-level mean of the five-feature macro NRMSE. Every number below comes from `fold_D_results.json` and the shards
in `CFRI_fold_D_results.zip`.

## Primary hypotheses (feature-wise masks)

| | Comparator | Comparator mean | CFRI mean | Difference (95% bootstrap) | CFRI lower in | k | Met |
|---|---|---:|---:|---|---:|---:|---|
| H1 | Log-linear interpolation | 0.3616 | 0.2822 | −0.0795 (−0.0934, −0.0669) | 34/34 | 25 | yes |
| H2 | Original ITRI, w = 1, p = 5 | 0.3864 | 0.2822 | −0.1042 (−0.1521, −0.0743) | 34/34 | 25 | yes |
| H3 | CFRI without residual (with projection) | 0.3049 | 0.2822 | −0.0228 (−0.0288, −0.0180) | 34/34 | 25 | yes |
| H4 | Normalized SAITS V2 | 0.2950 | 0.2822 | −0.0128 (−0.0217, −0.0061) | 29/34 | 25 | yes |

All four primary hypotheses are met. The prespecified interpretation (protocol Section 7) applies:

> CFRI is confirmed on 2026 data against log-linear interpolation, the original ITRI and SAITS V2, and the learned
> residual contributes beyond the projection and the volume rule.

H1 to H3 are also met in each leave-one-region-out subset.

## Projection-equalized comparisons (secondary)

| | Comparator | Comparator mean | Difference (95% bootstrap) | CFRI lower in |
|---|---|---:|---|---:|
| P1 | Log-linear + projection | 0.3230 | −0.0408 (−0.0491, −0.0335) | 34/34 |
| P2 | ITRI + projection | 0.3477 | −0.0656 (−0.1058, −0.0414) | 34/34 |
| P3 | SAITS V2 + projection | 0.2861 | −0.0040 (−0.0117, +0.0022) | 13/34 |

When SAITS V2 receives the same OHLC projection, it and CFRI do not differ on the macro endpoint. The median
asset-level difference is +0.0007.

The two methods trade places by feature:

- **Prices.** On the four prices, SAITS V2 + projection is lower:
  - mean price NRMSE 0.0714 against 0.0789 for CFRI;
  - CFRI is lower in only 4 of 34 ETFs.
- **Volume.** On volume, CFRI is lower:
  - 1.0951 against 1.1448;
  - CFRI is lower in 25 of 34 ETFs.

Against SAITS V2 without projection, CFRI is also lower on prices: 0.0789 against 0.0825, in 24 of 34 ETFs.

## Other secondary results

### Gap length

CFRI's advantage over log-linear interpolation grows with block length:

| Block length | CFRI | Log-linear |
|---:|---:|---:|
| 1 day | 0.2495 | 0.2984 |
| 20 days | 0.3465 | 0.4849 |

Against SAITS V2 + projection, the number of ETFs in which CFRI is lower changes with block length:

| Block length (days) | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| CFRI lower (of 34) | 16 | 23 | 22 | 18 | 14 | 11 |

### By feature, feature-wise masks

| Method | Open | High | Low | Close | Volume |
|---|---:|---:|---:|---:|---:|
| Log-linear | 0.1571 | 0.1449 | 0.1626 | 0.1582 | 1.1854 |
| CFRI | 0.0859 | 0.0686 | 0.0785 | 0.0828 | 1.0951 |
| SAITS V2 | 0.0908 | 0.0730 | 0.0788 | 0.0875 | 1.1448 |
| SAITS V2 + projection | 0.0803 | 0.0610 | 0.0669 | 0.0776 | 1.1448 |

### OHLC consistency (feature-wise masks)

Without projection, the following shares of rows with an imputed price violate the OHLC order:

| Method | Share of rows violating OHLC order |
|---|---:|
| Log-linear | 54.9% |
| ITRI | 54.9% |
| CFRI | 38.0% |
| SAITS V2 | 28.6% |
| Chained ridge | 15.3% |

After projection the share is 0%. The true values have no violations.

### Whole-row masks

With all prices of a day hidden, CFRI's prices equal log-linear interpolation, so CFRI differs from log-linear only
through its volume rule.

| Method | Mean NRMSE |
|---|---:|
| CFRI | 0.3399 |
| Log-linear | 0.3597 |
| SAITS V2 | 0.4942 |
| Chained ridge | 3.8794 |

CFRI is lower than log-linear in 33 of 34 ETFs and lower than SAITS V2 in 34 of 34.

### Asset concentration

Two ETFs, EWY and EFNL, have asset-level NRMSE above 1 for every method. The main cause is volume:

| ETF | Volume NRMSE |
|---|---:|
| EWY | 5.4 |
| EFNL | 5.0 |

Medians are therefore lower than means:

| Method | Median NRMSE |
|---|---:|
| CFRI | 0.2194 |
| SAITS V2 | 0.2269 |
| SAITS V2 + projection | 0.2170 |
| Log-linear | 0.2727 |

### Cost

| Method | Time per condition | Hardware and measurement |
|---|---:|---|
| All 16 CPU method variants together | 0.48 s | One thread per process, 12 vCPUs |
| CFRI alone | about 0.07 s | One CPU thread, separate timing on data of the same size |
| SAITS V2 | 13.5 s | A100 GPU, four jobs sharing the GPU |

## Agreement with the exploratory folds

CFRI's mean NRMSE was 0.2238, 0.1941 and 0.1699 in folds A to C and 0.2822 in fold D. In every fold it was below
log-linear interpolation, the original ITRI, the no-residual variant and SAITS V2.

In folds A to C it was lower in all 34 ETFs for each of these comparisons. In fold D it was lower in 34, 34, 34 and
29 ETFs respectively.

The projection-equalized SAITS comparison was not available for folds A to C.

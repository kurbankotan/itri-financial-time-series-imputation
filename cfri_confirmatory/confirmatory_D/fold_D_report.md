# CFRI confirmatory fold D results (cfri-core-v2)

Eligible assets: 34; excluded: none (reasons in data_receipt.json); CPU condition errors: 0.
SAITS assets completed (of 34): V2_feature 34, V2_row 34, V1_feature 34, V1_row 34; SAITS condition errors: 0.

## Primary hypotheses (feature-wise masks)

| | comparator | CFRI mean | comparator mean | difference (95% bootstrap) | CFRI lower in | k | met |
|---|---|---|---|---|---|---|---|
| H1 | log_linear | 0.2822 | 0.3616 | -0.0795 (-0.0934, -0.0669) | 34/34 | 25 | yes |
| H2 | itri_w1_p5 | 0.2822 | 0.3864 | -0.1042 (-0.1521, -0.0743) | 34/34 | 25 | yes |
| H3 | cfri_fixed_no_cross_proj | 0.2822 | 0.3049 | -0.0228 (-0.0288, -0.0180) | 34/34 | 25 | yes |
| H4 | saits_winnorm_V2 | 0.2822 | 0.2950 | -0.0128 (-0.0217, -0.0061) | 29/34 | 25 | yes |

## Projection-equalized comparisons (secondary, feature-wise masks)

| | comparator | CFRI mean | comparator mean | difference (95% bootstrap) | CFRI lower in | k | met |
|---|---|---|---|---|---|---|---|
| P1 | log_linear_proj | 0.2822 | 0.3230 | -0.0408 (-0.0491, -0.0335) | 34/34 | 25 | yes |
| P2 | itri_w1_p5_proj | 0.2822 | 0.3477 | -0.0656 (-0.1058, -0.0414) | 34/34 | 25 | yes |
| P3 | saits_winnorm_V2_proj | 0.2822 | 0.2861 | -0.0040 (-0.0117, +0.0022) | 13/34 | 25 | no |

## Mean asset-level NRMSE, feature masks (34 assets, 1734 conditions per method)

| method | mean | median | CFRI lower in |
|---|---|---|---|
| linear_raw | 0.3833 | 0.2834 | 34/34 |
| log_linear | 0.3616 | 0.2727 | 34/34 |
| itri_w1_p5 | 0.3864 | 0.2876 | 34/34 |
| chained_ridge | 0.5173 | 0.4398 | 34/34 |
| cfri_adaptive_no_cross | 0.3555 | 0.2710 | 34/34 |
| cfri_adaptive | 0.3093 | 0.2419 | 34/34 |
| cfri_fixed_no_cross | 0.3436 | 0.2661 | 34/34 |
| cfri_fixed | 0.2963 | 0.2304 | 34/34 |
| linear_raw_proj | 0.3447 | 0.2537 | 34/34 |
| log_linear_proj | 0.3230 | 0.2447 | 34/34 |
| itri_w1_p5_proj | 0.3477 | 0.2613 | 34/34 |
| chained_ridge_proj | 0.5147 | 0.4363 | 34/34 |
| cfri_adaptive_no_cross_proj | 0.3169 | 0.2484 | 34/34 |
| cfri_adaptive_proj | 0.2947 | 0.2303 | 34/34 |
| cfri_fixed_no_cross_proj | 0.3049 | 0.2398 | 34/34 |
| cfri_fixed_proj | 0.2822 | 0.2194 |  |
| saits_winnorm_V1 (paired on 1734 of 1734 conditions) | 0.3128 | 0.2407 | 34/34 |
| saits_winnorm_V2 (paired on 1734 of 1734 conditions) | 0.2950 | 0.2269 | 29/34 |
| saits_winnorm_V1_proj (paired on 1734 of 1734 conditions) | 0.2987 | 0.2278 | 29/34 |
| saits_winnorm_V2_proj (paired on 1734 of 1734 conditions) | 0.2861 | 0.2170 | 13/34 |

OHLC violation share of rows with an imputed price: cfri_adaptive 0.387, cfri_adaptive_no_cross 0.549, cfri_adaptive_no_cross_proj 0.000, cfri_adaptive_proj 0.000, cfri_fixed 0.380, cfri_fixed_no_cross 0.549, cfri_fixed_no_cross_proj 0.000, cfri_fixed_proj 0.000, chained_ridge 0.153, chained_ridge_proj 0.000, itri_w1_p5 0.549, itri_w1_p5_proj 0.000, linear_raw 0.549, linear_raw_proj 0.000, log_linear 0.549, log_linear_proj 0.000, saits_winnorm_V1 0.348, saits_winnorm_V1_proj 0.000, saits_winnorm_V2 0.286, saits_winnorm_V2_proj 0.000, truth 0.000

## Mean asset-level NRMSE, row masks (34 assets, 1734 conditions per method)

| method | mean | median | CFRI lower in |
|---|---|---|---|
| linear_raw | 0.3818 | 0.2893 | 34/34 |
| log_linear | 0.3597 | 0.2743 | 33/34 |
| itri_w1_p5 | 0.3784 | 0.2829 | 34/34 |
| chained_ridge | 3.8794 | 3.4875 | 34/34 |
| cfri_adaptive_no_cross | 0.3557 | 0.2706 | 33/34 |
| cfri_adaptive | 0.3557 | 0.2706 | 33/34 |
| cfri_fixed_no_cross | 0.3399 | 0.2666 | 2/34 |
| cfri_fixed | 0.3399 | 0.2666 | 0/34 |
| linear_raw_proj | 0.3818 | 0.2893 | 34/34 |
| log_linear_proj | 0.3597 | 0.2743 | 33/34 |
| itri_w1_p5_proj | 0.3784 | 0.2829 | 34/34 |
| chained_ridge_proj | 3.8794 | 3.4875 | 34/34 |
| cfri_adaptive_no_cross_proj | 0.3557 | 0.2706 | 33/34 |
| cfri_adaptive_proj | 0.3557 | 0.2706 | 33/34 |
| cfri_fixed_no_cross_proj | 0.3399 | 0.2666 | 2/34 |
| cfri_fixed_proj | 0.3399 | 0.2666 |  |
| saits_winnorm_V1 (paired on 1734 of 1734 conditions) | 0.5177 | 0.4089 | 34/34 |
| saits_winnorm_V2 (paired on 1734 of 1734 conditions) | 0.4942 | 0.3988 | 34/34 |
| saits_winnorm_V1_proj (paired on 1734 of 1734 conditions) | 0.5175 | 0.4088 | 34/34 |
| saits_winnorm_V2_proj (paired on 1734 of 1734 conditions) | 0.4941 | 0.3987 | 34/34 |

OHLC violation share of rows with an imputed price: cfri_adaptive 0.000, cfri_adaptive_no_cross 0.000, cfri_adaptive_no_cross_proj 0.000, cfri_adaptive_proj 0.000, cfri_fixed 0.000, cfri_fixed_no_cross 0.000, cfri_fixed_no_cross_proj 0.000, cfri_fixed_proj 0.000, chained_ridge 0.000, chained_ridge_proj 0.000, itri_w1_p5 0.025, itri_w1_p5_proj 0.000, linear_raw 0.000, linear_raw_proj 0.000, log_linear 0.000, log_linear_proj 0.000, saits_winnorm_V1 0.335, saits_winnorm_V1_proj 0.000, saits_winnorm_V2 0.133, saits_winnorm_V2_proj 0.000, truth 0.000

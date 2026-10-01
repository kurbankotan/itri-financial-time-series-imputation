# External-validation results (descriptive)

The primary endpoint is feature-macro NRMSE. Results below first average conditions and folds within each asset, then summarize the 34 assets. They do not treat masks or ETFs as independent markets.

## All block lengths

| method | mean_asset_nrmse | median_asset_nrmse | mean_asset_nmae |
| --- | --- | --- | --- |
| linear | 0.270358 | 0.267163 | 0.181566 |
| itri_w1_p5 | 0.272325 | 0.266233 | 0.197332 |
| pchip | 0.278530 | 0.275754 | 0.186855 |
| chained_ridge | 0.332995 | 0.341987 | 0.189858 |
| saits | 1.145963 | 1.005723 | 0.920136 |
| brits | 1.195044 | 1.087426 | 0.985992 |
| knn5 | 1.601394 | 1.348001 | 1.210183 |

## One- and two-day blocks

| method | mean_asset_nrmse | median_asset_nrmse | mean_asset_nmae |
| --- | --- | --- | --- |
| itri_w1_p5 | 0.224157 | 0.222118 | 0.153721 |
| linear | 0.228123 | 0.222395 | 0.146842 |
| pchip | 0.233440 | 0.228317 | 0.149866 |
| chained_ridge | 0.352671 | 0.360072 | 0.190715 |
| saits | 1.150518 | 0.992920 | 0.911278 |
| brits | 1.168127 | 1.058822 | 0.946863 |
| knn5 | 1.644047 | 1.381924 | 1.231314 |

## Paired asset win counts

| scope | comparator | itri_wins | itri_losses | mean_difference | median_difference |
| --- | --- | --- | --- | --- | --- |
| all_blocks | brits | 34 | 0 | -0.922720 | -0.808859 |
| all_blocks | chained_ridge | 31 | 3 | -0.060670 | -0.045777 |
| all_blocks | knn5 | 34 | 0 | -1.329070 | -1.085569 |
| all_blocks | linear | 16 | 18 | 0.001967 | 0.000874 |
| all_blocks | pchip | 23 | 11 | -0.006206 | -0.005764 |
| all_blocks | saits | 34 | 0 | -0.873638 | -0.741086 |
| blocks_1_2 | brits | 34 | 0 | -0.943969 | -0.827159 |
| blocks_1_2 | chained_ridge | 34 | 0 | -0.128513 | -0.111267 |
| blocks_1_2 | knn5 | 34 | 0 | -1.419890 | -1.189411 |
| blocks_1_2 | linear | 24 | 10 | -0.003965 | -0.003230 |
| blocks_1_2 | pchip | 28 | 6 | -0.009283 | -0.008943 |
| blocks_1_2 | saits | 34 | 0 | -0.926361 | -0.766385 |

## Validation–test diagnostic

Validation selected neural epochs and is not a second held-out test.

| method | mean_test_nrmse | mean_validation_nrmse | mean_test_minus_validation_nrmse |
| --- | --- | --- | --- |
| brits | 1.195044 | 0.642550 | 0.552494 |
| chained_ridge | 0.332995 | 0.244638 | 0.088357 |
| itri_w1_p5 | 0.272325 | 0.224415 | 0.047910 |
| knn5 | 1.601394 | 0.940080 | 0.661314 |
| linear | 0.270358 | 0.230670 | 0.039688 |
| pchip | 0.278530 | 0.237798 | 0.040732 |
| saits | 1.145963 | 0.568933 | 0.577029 |

No causal or population-level significance claim follows from these descriptive comparisons.

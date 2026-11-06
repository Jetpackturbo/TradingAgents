# Sanity report

91/91 checks passed.

Horizon 30 min (6 x 5-min bars); tolerance |z| <= 3.0 using the across-seed standard error of gold.

Check-specification revisions (each made after a failure; original numbers are still shown):

- Placebo worlds: proxy-vs-gold rank agreement is not tested (all gold ~0, so it compares noise with noise). The placebo claim is that F carries no information, so the test is that truth-derived factors have no significant gold; price-only signal (e.g. microstructure reversal) is reported.
- Bootstrap calibration is tested against within-seed time-sampling noise (half-split sd), which is what a within-seed bootstrap estimates. Seed-to-seed sd is also shown: in UWTC worlds it is larger because per-seed parameters (per-symbol vol/drift) vary between seeds.

| world | check | expected | measured | pass |
|---|---|---|---|---|
| ref_slow | oracle gold = analytic E[RankIC] | +0.6512 | +0.6542 +/- 0.0014 (z=+2.17) | PASS |
| ref_slow | momentum div(last(close(13)), first(close(13))) gold = analytic | +0.2500 | +0.2520 +/- 0.0031 (z=+0.62) | PASS |
| ref_slow | oracle mispricing_corr = 1 | 1.0000 | 1.000000 | PASS |
| ref_slow | neg_oracle = -oracle (gold and proxy) | -0.654188 / -0.645621 | -0.654188 / -0.645621 | PASS |
| ref_slow | neg(momentum) = -momentum (DSL path) | -0.251984 | -0.251984 | PASS |
| ref_slow | random factor gold = 0 | 0 within 3 se | +0.0006 +/- 0.0011 (z=+0.51) | PASS |
| ref_slow | constant factor rejected | valid=False | valid=False | PASS |
| ref_slow | invalid DSL op rejected | None | None | PASS |
| ref_slow | block-bootstrap se calibrated to within-seed time sampling (oracle) | ratio in [0.67, 1.5] | mean boot se 0.0112 vs half-split sd 0.0134 (ratio 0.84); seed-to-seed sd 0.0109 (ratio 1.03) | PASS |
| ref_slow | block-bootstrap se calibrated to within-seed time sampling (momentum) | ratio in [0.67, 1.5] | mean boot se 0.0197 vs half-split sd 0.0235 (ratio 0.84); seed-to-seed sd 0.0250 (ratio 0.79) | PASS |
| ref_slow | block-bootstrap se calibrated to within-seed time sampling (random) | ratio in [0.67, 1.5] | mean boot se 0.0087 vs half-split sd 0.0094 (ratio 0.93); seed-to-seed sd 0.0088 (ratio 0.99) | PASS |
| ref_slow | proxy sign = gold sign (factors with significant gold) | 4/4 | 4/4 | PASS |
| ref_slow | proxy vs gold rank agreement (Spearman over factor set) | > 0.8 | 1.000 | PASS |
| ref_slow | mean |proxy - gold| (single train split vs held-out seeds) | report only | 0.0158 | PASS |
| ref_fast | oracle gold = analytic E[RankIC] | +0.3247 | +0.3261 +/- 0.0010 (z=+1.38) | PASS |
| ref_fast | momentum div(last(close(13)), first(close(13))) gold = analytic | +0.0350 | +0.0379 +/- 0.0025 (z=+1.17) | PASS |
| ref_fast | oracle mispricing_corr = 1 | 1.0000 | 1.000000 | PASS |
| ref_fast | neg_oracle = -oracle (gold and proxy) | -0.326096 / -0.327196 | -0.326096 / -0.327196 | PASS |
| ref_fast | neg(momentum) = -momentum (DSL path) | -0.037925 | -0.037925 | PASS |
| ref_fast | random factor gold = 0 | 0 within 3 se | +0.0020 +/- 0.0010 (z=+2.03) | PASS |
| ref_fast | constant factor rejected | valid=False | valid=False | PASS |
| ref_fast | invalid DSL op rejected | None | None | PASS |
| ref_fast | block-bootstrap se calibrated to within-seed time sampling (oracle) | ratio in [0.67, 1.5] | mean boot se 0.0083 vs half-split sd 0.0082 (ratio 1.01); seed-to-seed sd 0.0083 (ratio 0.99) | PASS |
| ref_fast | block-bootstrap se calibrated to within-seed time sampling (momentum) | ratio in [0.67, 1.5] | mean boot se 0.0181 vs half-split sd 0.0206 (ratio 0.88); seed-to-seed sd 0.0200 (ratio 0.91) | PASS |
| ref_fast | block-bootstrap se calibrated to within-seed time sampling (random) | ratio in [0.67, 1.5] | mean boot se 0.0086 vs half-split sd 0.0083 (ratio 1.04); seed-to-seed sd 0.0079 (ratio 1.09) | PASS |
| ref_fast | proxy sign = gold sign (factors with significant gold) | 4/4 | 4/4 | PASS |
| ref_fast | proxy vs gold rank agreement (Spearman over factor set) | > 0.8 | 1.000 | PASS |
| ref_fast | mean |proxy - gold| (single train split vs held-out seeds) | report only | 0.0016 | PASS |
| ref_noisy | oracle gold = analytic E[RankIC] | +0.4502 | +0.4512 +/- 0.0012 (z=+0.83) | PASS |
| ref_noisy | momentum div(last(close(13)), first(close(13))) gold = analytic | -0.2147 | -0.2133 +/- 0.0018 (z=+0.77) | PASS |
| ref_noisy | oracle mispricing_corr = 1 | 1.0000 | 1.000000 | PASS |
| ref_noisy | neg_oracle = -oracle (gold and proxy) | -0.451202 / -0.456553 | -0.451202 / -0.456553 | PASS |
| ref_noisy | neg(momentum) = -momentum (DSL path) | +0.213302 | +0.213302 | PASS |
| ref_noisy | random factor gold = 0 | 0 within 3 se | +0.0015 +/- 0.0011 (z=+1.31) | PASS |
| ref_noisy | constant factor rejected | valid=False | valid=False | PASS |
| ref_noisy | invalid DSL op rejected | None | None | PASS |
| ref_noisy | block-bootstrap se calibrated to within-seed time sampling (oracle) | ratio in [0.67, 1.5] | mean boot se 0.0112 vs half-split sd 0.0088 (ratio 1.27); seed-to-seed sd 0.0093 (ratio 1.20) | PASS |
| ref_noisy | block-bootstrap se calibrated to within-seed time sampling (momentum) | ratio in [0.67, 1.5] | mean boot se 0.0141 vs half-split sd 0.0135 (ratio 1.05); seed-to-seed sd 0.0142 (ratio 0.99) | PASS |
| ref_noisy | block-bootstrap se calibrated to within-seed time sampling (random) | ratio in [0.67, 1.5] | mean boot se 0.0087 vs half-split sd 0.0092 (ratio 0.94); seed-to-seed sd 0.0089 (ratio 0.98) | PASS |
| ref_noisy | proxy sign = gold sign (factors with significant gold) | 4/4 | 4/4 | PASS |
| ref_noisy | proxy vs gold rank agreement (Spearman over factor set) | > 0.8 | 1.000 | PASS |
| ref_noisy | mean |proxy - gold| (single train split vs held-out seeds) | report only | 0.0136 | PASS |
| ref_placebo | oracle gold = analytic E[RankIC] | +0.0000 | +0.0034 +/- 0.0024 (z=+1.44) | PASS |
| ref_placebo | momentum div(last(close(13)), first(close(13))) gold = analytic | +0.0000 | +0.0003 +/- 0.0023 (z=+0.12) | PASS |
| ref_placebo | oracle mispricing_corr = 1 | 1.0000 | 1.000000 | PASS |
| ref_placebo | neg_oracle = -oracle (gold and proxy) | -0.003388 / +0.003938 | -0.003388 / +0.003938 | PASS |
| ref_placebo | neg(momentum) = -momentum (DSL path) | -0.000283 | -0.000283 | PASS |
| ref_placebo | random factor gold = 0 | 0 within 3 se | +0.0004 +/- 0.0009 (z=+0.43) | PASS |
| ref_placebo | constant factor rejected | valid=False | valid=False | PASS |
| ref_placebo | invalid DSL op rejected | None | None | PASS |
| ref_placebo | block-bootstrap se calibrated to within-seed time sampling (oracle) | ratio in [0.67, 1.5] | mean boot se 0.0198 vs half-split sd 0.0198 (ratio 1.00); seed-to-seed sd 0.0188 (ratio 1.05) | PASS |
| ref_placebo | block-bootstrap se calibrated to within-seed time sampling (momentum) | ratio in [0.67, 1.5] | mean boot se 0.0178 vs half-split sd 0.0169 (ratio 1.06); seed-to-seed sd 0.0182 (ratio 0.98) | PASS |
| ref_placebo | block-bootstrap se calibrated to within-seed time sampling (random) | ratio in [0.67, 1.5] | mean boot se 0.0086 vs half-split sd 0.0083 (ratio 1.04); seed-to-seed sd 0.0071 (ratio 1.22) | PASS |
| ref_placebo | proxy sign = gold sign (factors with significant gold) | 0/0 | 0/0 | PASS |
| ref_placebo | no truth-derived factor (oracle, neg_oracle) has significant gold (placebo) | 0 significant | 0 significant | PASS |
| ref_placebo | price-only factors with significant gold (placebo; microstructure, not F) | report only | none | PASS |
| ref_placebo | mean |proxy - gold| (single train split vs held-out seeds) | report only | 0.0102 | PASS |
| uwtc_informed | oracle gold > 0 (informed) / = 0 (placebo) | > 0 | +0.4288 +/- 0.0015 | PASS |
| uwtc_informed | momentum div(last(close(13)), first(close(13))) gold (no closed form) | report only | +0.0652 +/- 0.0024 | PASS |
| uwtc_informed | oracle mispricing_corr = 1 | 1.0000 | 1.000000 | PASS |
| uwtc_informed | neg_oracle = -oracle (gold and proxy) | -0.428771 / -0.434608 | -0.428771 / -0.434608 | PASS |
| uwtc_informed | neg(momentum) = -momentum (DSL path) | -0.065200 | -0.065200 | PASS |
| uwtc_informed | random factor gold = 0 | 0 within 3 se | +0.0008 +/- 0.0011 (z=+0.71) | PASS |
| uwtc_informed | constant factor rejected | valid=False | valid=False | PASS |
| uwtc_informed | invalid DSL op rejected | None | None | PASS |
| uwtc_informed | block-bootstrap se calibrated to within-seed time sampling (oracle) | ratio in [0.67, 1.5] | mean boot se 0.0077 vs half-split sd 0.0078 (ratio 0.98); seed-to-seed sd 0.0118 (ratio 0.65) | PASS |
| uwtc_informed | block-bootstrap se calibrated to within-seed time sampling (momentum) | ratio in [0.67, 1.5] | mean boot se 0.0166 vs half-split sd 0.0189 (ratio 0.88); seed-to-seed sd 0.0194 (ratio 0.85) | PASS |
| uwtc_informed | block-bootstrap se calibrated to within-seed time sampling (random) | ratio in [0.67, 1.5] | mean boot se 0.0075 vs half-split sd 0.0083 (ratio 0.90); seed-to-seed sd 0.0085 (ratio 0.87) | PASS |
| uwtc_informed | proxy sign = gold sign (factors with significant gold) | 4/4 | 4/4 | PASS |
| uwtc_informed | proxy vs gold rank agreement (Spearman over factor set) | > 0.8 | 1.000 | PASS |
| uwtc_informed | mean |proxy - gold| (single train split vs held-out seeds) | report only | 0.0156 | PASS |
| uwtc_placebo | oracle gold > 0 (informed) / = 0 (placebo) | 0 within 3 se | -0.0030 +/- 0.0022 | PASS |
| uwtc_placebo | momentum div(last(close(13)), first(close(13))) gold (no closed form) | report only | -0.0105 +/- 0.0019 | PASS |
| uwtc_placebo | oracle mispricing_corr = 1 | 1.0000 | 1.000000 | PASS |
| uwtc_placebo | neg_oracle = -oracle (gold and proxy) | +0.002990 / -0.002558 | +0.002990 / -0.002558 | PASS |
| uwtc_placebo | neg(momentum) = -momentum (DSL path) | +0.010500 | +0.010500 | PASS |
| uwtc_placebo | random factor gold = 0 | 0 within 3 se | +0.0008 +/- 0.0010 (z=+0.82) | PASS |
| uwtc_placebo | constant factor rejected | valid=False | valid=False | PASS |
| uwtc_placebo | invalid DSL op rejected | None | None | PASS |
| uwtc_placebo | block-bootstrap se calibrated to within-seed time sampling (oracle) | ratio in [0.67, 1.5] | mean boot se 0.0169 vs half-split sd 0.0185 (ratio 0.91); seed-to-seed sd 0.0178 (ratio 0.95) | PASS |
| uwtc_placebo | block-bootstrap se calibrated to within-seed time sampling (momentum) | ratio in [0.67, 1.5] | mean boot se 0.0151 vs half-split sd 0.0163 (ratio 0.93); seed-to-seed sd 0.0150 (ratio 1.01) | PASS |
| uwtc_placebo | block-bootstrap se calibrated to within-seed time sampling (random) | ratio in [0.67, 1.5] | mean boot se 0.0076 vs half-split sd 0.0081 (ratio 0.93); seed-to-seed sd 0.0079 (ratio 0.96) | PASS |
| uwtc_placebo | proxy sign = gold sign (factors with significant gold) | 2/2 | 2/2 | PASS |
| uwtc_placebo | no truth-derived factor (oracle, neg_oracle) has significant gold (placebo) | 0 significant | 0 significant | PASS |
| uwtc_placebo | price-only factors with significant gold (placebo; microstructure, not F) | report only | dsl:div(last(close(13)), first(close(13))) -0.0105; dsl:neg(div(last(close(13)), first(close(13)))) +0.0105 | PASS |
| uwtc_placebo | mean |proxy - gold| (single train split vs held-out seeds) | report only | 0.0113 | PASS |
| e2e | reward placed only on last response token | True | True | PASS |
| e2e | reward = evaluator proxy for the row's task window | +0.233746 (window 2024-01-03..2024-01-04) | +0.233746 | PASS |
| e2e | invalid / missing expr -> -1 | -1.0, -1.0 | -1.0, -1.0 | PASS |
| e2e | reward_extra_info logs gold/proxy_se per row | lists of len 3 | gold=[0.2471, nan, nan], proxy_se=[0.0281, 0.0, 0.0] | PASS |
| e2e | prompt rows carry task windows | start/end in extra_info | [('2024-01-03', '2024-01-04')] | PASS |

## Factor scores

| world | factor | proxy | se | gold | gold_se | mispricing_corr |
|---|---|---|---|---|---|---|
| ref_slow | `oracle` | +0.6456 | 0.0125 | +0.6542 | 0.0014 | +1.0000 |
| ref_slow | `neg_oracle` | -0.6456 | 0.0125 | -0.6542 | 0.0014 | -1.0000 |
| ref_slow | `random` | -0.0065 | 0.0088 | +0.0006 | 0.0011 | +0.0001 |
| ref_slow | `constant` | invalid | | | | |
| ref_slow | `dsl:div(last(close(13)), first(close(13)))` | +0.2245 | 0.0179 | +0.2520 | 0.0031 | +0.3588 |
| ref_slow | `dsl:neg(div(last(close(13)), first(close(13))))` | -0.2245 | 0.0179 | -0.2520 | 0.0031 | -0.3588 |
| ref_fast | `oracle` | +0.3272 | 0.0085 | +0.3261 | 0.0010 | +1.0000 |
| ref_fast | `neg_oracle` | -0.3272 | 0.0085 | -0.3261 | 0.0010 | -1.0000 |
| ref_fast | `random` | +0.0048 | 0.0084 | +0.0020 | 0.0010 | -0.0008 |
| ref_fast | `constant` | invalid | | | | |
| ref_fast | `dsl:div(last(close(13)), first(close(13)))` | +0.0396 | 0.0176 | +0.0379 | 0.0025 | +0.0994 |
| ref_fast | `dsl:neg(div(last(close(13)), first(close(13))))` | -0.0396 | 0.0176 | -0.0379 | 0.0025 | -0.0994 |
| ref_noisy | `oracle` | +0.4566 | 0.0101 | +0.4512 | 0.0012 | +1.0000 |
| ref_noisy | `neg_oracle` | -0.4566 | 0.0101 | -0.4512 | 0.0012 | -1.0000 |
| ref_noisy | `random` | -0.0060 | 0.0095 | +0.0015 | 0.0011 | +0.0007 |
| ref_noisy | `constant` | invalid | | | | |
| ref_noisy | `dsl:div(last(close(13)), first(close(13)))` | -0.2381 | 0.0131 | -0.2133 | 0.0018 | -0.4378 |
| ref_noisy | `dsl:neg(div(last(close(13)), first(close(13))))` | +0.2381 | 0.0131 | +0.2133 | 0.0018 | +0.4378 |
| ref_placebo | `oracle` | -0.0039 | 0.0204 | +0.0034 | 0.0024 | +1.0000 |
| ref_placebo | `neg_oracle` | +0.0039 | 0.0204 | -0.0034 | 0.0024 | -1.0000 |
| ref_placebo | `random` | -0.0022 | 0.0098 | +0.0004 | 0.0009 | +0.0002 |
| ref_placebo | `constant` | invalid | | | | |
| ref_placebo | `dsl:div(last(close(13)), first(close(13)))` | -0.0165 | 0.0175 | +0.0003 | 0.0023 | -0.0815 |
| ref_placebo | `dsl:neg(div(last(close(13)), first(close(13))))` | +0.0165 | 0.0175 | -0.0003 | 0.0023 | +0.0815 |
| uwtc_informed | `oracle` | +0.4346 | 0.0084 | +0.4288 | 0.0015 | +1.0000 |
| uwtc_informed | `neg_oracle` | -0.4346 | 0.0084 | -0.4288 | 0.0015 | -1.0000 |
| uwtc_informed | `random` | -0.0015 | 0.0068 | +0.0008 | 0.0011 | -0.0009 |
| uwtc_informed | `constant` | invalid | | | | |
| uwtc_informed | `dsl:div(last(close(13)), first(close(13)))` | +0.0331 | 0.0183 | +0.0652 | 0.0024 | +0.0977 |
| uwtc_informed | `dsl:neg(div(last(close(13)), first(close(13))))` | -0.0331 | 0.0183 | -0.0652 | 0.0024 | -0.0977 |
| uwtc_placebo | `oracle` | +0.0026 | 0.0171 | -0.0030 | 0.0022 | +1.0000 |
| uwtc_placebo | `neg_oracle` | -0.0026 | 0.0171 | +0.0030 | 0.0022 | -1.0000 |
| uwtc_placebo | `random` | +0.0056 | 0.0079 | +0.0008 | 0.0010 | -0.0027 |
| uwtc_placebo | `constant` | invalid | | | | |
| uwtc_placebo | `dsl:div(last(close(13)), first(close(13)))` | -0.0308 | 0.0145 | -0.0105 | 0.0019 | -0.0358 |
| uwtc_placebo | `dsl:neg(div(last(close(13)), first(close(13))))` | +0.0308 | 0.0145 | +0.0105 | 0.0019 | +0.0358 |

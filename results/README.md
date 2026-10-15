# Aggregated results

All files are aggregates; none contains learner, game or item identifiers or individual responses. CSV files use a comma separator and a dot as decimal mark. Effects are differences in probability (multiply by 100 for percentage points) unless a column name says otherwise. The main `README.md` maps the files to the tables and figures of the article.

## `descriptive/`

Preprocessing funnel (Table 1) and descriptive statistics of the analysis sample.

| File | Rows × columns | Columns |
|---|---|---|
| `preprocessing_funnel.csv` | 17 × 2 | step, n |
| `table1_descriptives.csv` | 25 × 3 | group, statistic, value |

## `primary/deep_modern/`

AKT and HawkesKT: test metrics and implied effects per window; `modern_levels.csv`, `modern_pairs.csv` and `modern_curves.csv` = three windows pooled; `F3_boot` = window-F3 bootstrap including both models.

| File | Rows × columns | Columns |
|---|---|---|
| `modern_curves.csv` | 189 × 5 | model, group, lag_bin, effect, n |
| `modern_levels.csv` | 18 × 9 | model, group, AUC, AUC_lo, AUC_hi, D_pp, lo, hi, P_best_D |
| `modern_pairs.csv` | 9 × 17 | reference, model, dAUC, dAUC_lo, dAUC_hi, dD_all, dD_all_lo, dD_all_hi, P_model_better_all, dD_after correct, dD_after correct_lo, dD_after correct_hi, P_mod… |

## `primary/deep_modern/F3/`

AKT and HawkesKT: test metrics and implied effects per window; `modern_levels.csv`, `modern_pairs.csv` and `modern_curves.csv` = three windows pooled; `F3_boot` = window-F3 bootstrap including both models.

| File | Rows × columns | Columns |
|---|---|---|
| `modern_f3.csv` | 2 × 14 | model, n, AUC, LogLoss, RMSE, ECE_all, ECE_reexp, cal_intercept, cal_slope, slope_lo, slope_hi, implied_pp, D_F3_pp, bins_inside |
| `rq2_implied_effects_deep.csv` | 56 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_metrics_test_deep.csv` | 2 × 12 | model, n, AUC, RMSE, LogLoss, ECE, val_AUC, dim, lr, dropout, window, reweight |

## `primary/deep_modern/F3_boot/`

AKT and HawkesKT: test metrics and implied effects per window; `modern_levels.csv`, `modern_pairs.csv` and `modern_curves.csv` = three windows pooled; `F3_boot` = window-F3 bootstrap including both models.

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_bootstrap_pairs.csv` | 26 × 8 | reference, model, dAUC, dAUC_lo, dAUC_hi, d_rmse, d_rmse_lo, d_rmse_hi |
| `rq2_bootstrap_summary.csv` | 14 × 12 | model, AUC, AUC_lo, AUC_hi, implied_all, implied_lo, implied_hi, rmse_lag, rmse_lo, rmse_hi, P(best AUC), P(best consistency) |

## `primary/deep_modern/fold_2025-07/`

AKT and HawkesKT: test metrics and implied effects per window; `modern_levels.csv`, `modern_pairs.csv` and `modern_curves.csv` = three windows pooled; `F3_boot` = window-F3 bootstrap including both models.

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_implied_effects_deep.csv` | 56 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_metrics_test_deep.csv` | 2 × 12 | model, n, AUC, RMSE, LogLoss, ECE, val_AUC, dim, lr, dropout, window, reweight |

## `primary/deep_modern/fold_2025-10/`

AKT and HawkesKT: test metrics and implied effects per window; `modern_levels.csv`, `modern_pairs.csv` and `modern_curves.csv` = three windows pooled; `F3_boot` = window-F3 bootstrap including both models.

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_implied_effects_deep.csv` | 56 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_metrics_test_deep.csv` | 2 × 12 | model, n, AUC, RMSE, LogLoss, ECE, val_AUC, dim, lr, dropout, window, reweight |

## `primary/design_checks/`

Randomization and design checks of the item draw (Section 3.3, Figure 3).

| File | Rows × columns | Columns |
|---|---|---|
| `deck_global_recency.csv` | 13 × 7 | learner_status, global_recency, observed, expected, ratio, ci_low, ci_high |
| `difficulty_vs_ability.csv` | 3 × 5 | term, coef, ci_low, ci_high, outcome_sd |
| `outcome_lag_balance.csv` | 9 × 12 | lag_bin, observed_after_error, ratio_after_error, lo_err, hi_err, observed_after_correct, ratio_after_correct, lo_cor, hi_cor, difference, diff_lo, diff_hi |
| `randomization_balance.csv` | 198 × 11 | subset, window, property, n, learners, observed, expected, diff, ci_low, ci_high, rel_diff |
| `recency_rule_time_x_games.csv` | 40 × 7 | time_since, games_since, observed, expected, ratio, ci_low, ci_high |

## `primary/review/calib/`

Two-way learner × item bootstrap (pigeonhole) and calibration slopes (Appendix B, Table 7).

| File | Rows × columns | Columns |
|---|---|---|
| `calibration_slope_intercept_F3.csv` | 11 × 7 | model, intercept, intercept_lo, intercept_hi, slope, slope_lo, slope_hi |

## `primary/review/pigeonhole/`

Two-way learner × item bootstrap (pigeonhole) and calibration slopes (Appendix B, Table 7).

| File | Rows × columns | Columns |
|---|---|---|
| `pigeonhole_levels.csv` | 12 × 5 | model, group, D_pp, lo, hi |
| `pigeonhole_pairs.csv` | 4 × 17 | reference, model, dAUC, dAUC_lo, dAUC_hi, dD_all_pp, dD_all_lo, dD_all_hi, P_model_better_all, dD_after correct_pp, dD_after correct_lo, dD_after correct_hi,… |

## `primary/rq1/`

Causal effects of a previous encounter by lag and previous outcome (RQ1). `rq1_effects*.csv`: estimates (coef, se, CI) per specification and estimator; suffix `_test` = window F3, `_test_2025-07` / `_test_2025-10` = windows F1 / F2, `_fine` = 14-bin lag grid of the whole period. `rq1_cov_*.npy` and `rq1d_cov_*.npy`: covariance matrices of the estimates (terms listed in the matching `*_terms.csv`), used to redraw the causal reference in the bootstrap.

| File | Rows × columns | Columns |
|---|---|---|
| `rq1_cov_fine_B.npy` | 18 × 18 | covariance matrix of the estimates |
| `rq1_cov_fine_B_terms.csv` | 18 × 1 | 0 |
| `rq1_cov_fine_C.npy` | 32 × 32 | covariance matrix of the estimates |
| `rq1_cov_fine_C_terms.csv` | 32 × 1 | 0 |
| `rq1_cov_test_2025-07_A.npy` | 5 × 5 | covariance matrix of the estimates |
| `rq1_cov_test_2025-07_A_terms.csv` | 5 × 1 | 0 |
| `rq1_cov_test_2025-07_B.npy` | 13 × 13 | covariance matrix of the estimates |
| `rq1_cov_test_2025-07_B_terms.csv` | 13 × 1 | 0 |
| `rq1_cov_test_2025-07_C.npy` | 22 × 22 | covariance matrix of the estimates |
| `rq1_cov_test_2025-07_C_terms.csv` | 22 × 1 | 0 |
| `rq1_cov_test_2025-10_A.npy` | 5 × 5 | covariance matrix of the estimates |
| `rq1_cov_test_2025-10_A_terms.csv` | 5 × 1 | 0 |
| `rq1_cov_test_2025-10_B.npy` | 13 × 13 | covariance matrix of the estimates |
| `rq1_cov_test_2025-10_B_terms.csv` | 13 × 1 | 0 |
| `rq1_cov_test_2025-10_C.npy` | 22 × 22 | covariance matrix of the estimates |
| `rq1_cov_test_2025-10_C_terms.csv` | 22 × 1 | 0 |
| `rq1_cov_test_A.npy` | 5 × 5 | covariance matrix of the estimates |
| `rq1_cov_test_A_terms.csv` | 5 × 1 | 0 |
| `rq1_cov_test_B.npy` | 13 × 13 | covariance matrix of the estimates |
| `rq1_cov_test_B_terms.csv` | 13 × 1 | 0 |
| `rq1_cov_test_C.npy` | 22 × 22 | covariance matrix of the estimates |
| `rq1_cov_test_C_terms.csv` | 22 × 1 | 0 |
| `rq1_descriptive.csv` | 2 × 3 | seen, mean, count |
| `rq1_descriptive_fine.csv` | 2 × 3 | seen, mean, count |
| `rq1_descriptive_test.csv` | 2 × 3 | seen, mean, count |
| `rq1_descriptive_test_2025-07.csv` | 2 × 3 | seen, mean, count |
| `rq1_descriptive_test_2025-10.csv` | 2 × 3 | seen, mean, count |
| `rq1_effects.csv` | 151 × 7 | spec, estimator, term, coef, se, ci_low, ci_high |
| `rq1_effects_fine.csv` | 50 × 7 | spec, estimator, term, coef, se, ci_low, ci_high |
| `rq1_effects_test.csv` | 40 × 7 | spec, estimator, term, coef, se, ci_low, ci_high |
| `rq1_effects_test_2025-07.csv` | 40 × 7 | spec, estimator, term, coef, se, ci_low, ci_high |
| `rq1_effects_test_2025-10.csv` | 40 × 7 | spec, estimator, term, coef, se, ci_low, ci_high |
| `rq1_fine_lag_hours.csv` | 14 × 2 | lag_bin, hours |
| `rq1_model_info.csv` | 12 × 5 | spec, estimator, n, clusters, iters |
| `rq1_model_info_fine.csv` | 2 × 5 | spec, estimator, n, clusters, iters |
| `rq1_model_info_test.csv` | 3 × 5 | spec, estimator, n, clusters, iters |
| `rq1_model_info_test_2025-07.csv` | 3 × 5 | spec, estimator, n, clusters, iters |
| `rq1_model_info_test_2025-10.csv` | 3 × 5 | spec, estimator, n, clusters, iters |
| `rq1_robustness.csv` | 2 × 9 | term, coef, se, ci_low, ci_high, subset, n, clusters, iters |
| `rq1b_cell_counts.csv` | 16 × 3 | tb, gb, n |
| `rq1b_corr.csv` | 3 × 4 | Unnamed: 0, hours, games, cat_games |
| `rq1b_decay_interference.csv` | 99 × 7 | spec, estimator, term, coef, se, ci_low, ci_high |
| `rq1b_decay_interference_refined.csv` | 41 × 7 | spec, estimator, term, coef, se, ci_low, ci_high |
| `rq1c_error_sensitivity.csv` | 28 × 5 | term, coef, se, ci_low, ci_high |
| `rq1c_strata.csv` | 6 × 5 | prev_y, tert, n, mean_p_prev, mean_y |
| `rq1d_clogit.csv` | 20 × 9 | spec, term, beta, se, odds_ratio, pp_at_054, pp_lo, pp_hi, groups |
| `rq1d_clogit_test.csv` | 43 × 9 | spec, term, beta, se, odds_ratio, pp_at_054, pp_lo, pp_hi, groups |
| `rq1d_clogit_test_2025-07.csv` | 43 × 9 | spec, term, beta, se, odds_ratio, pp_at_054, pp_lo, pp_hi, groups |
| `rq1d_clogit_test_2025-10.csv` | 43 × 9 | spec, term, beta, se, odds_ratio, pp_at_054, pp_lo, pp_hi, groups |
| `rq1d_cov_test_2025-07_A.npy` | 6 × 6 | covariance matrix of the estimates |
| `rq1d_cov_test_2025-07_A_terms.csv` | 6 × 1 | 0 |
| `rq1d_cov_test_2025-07_B.npy` | 14 × 14 | covariance matrix of the estimates |
| `rq1d_cov_test_2025-07_B_terms.csv` | 14 × 1 | 0 |
| `rq1d_cov_test_2025-07_C.npy` | 23 × 23 | covariance matrix of the estimates |
| `rq1d_cov_test_2025-07_C_terms.csv` | 23 × 1 | 0 |
| `rq1d_cov_test_2025-10_A.npy` | 6 × 6 | covariance matrix of the estimates |
| `rq1d_cov_test_2025-10_A_terms.csv` | 6 × 1 | 0 |
| `rq1d_cov_test_2025-10_B.npy` | 14 × 14 | covariance matrix of the estimates |
| `rq1d_cov_test_2025-10_B_terms.csv` | 14 × 1 | 0 |
| `rq1d_cov_test_2025-10_C.npy` | 23 × 23 | covariance matrix of the estimates |
| `rq1d_cov_test_2025-10_C_terms.csv` | 23 × 1 | 0 |
| `rq1d_cov_test_A.npy` | 6 × 6 | covariance matrix of the estimates |
| `rq1d_cov_test_A_terms.csv` | 6 × 1 | 0 |
| `rq1d_cov_test_B.npy` | 14 × 14 | covariance matrix of the estimates |
| `rq1d_cov_test_B_terms.csv` | 14 × 1 | 0 |
| `rq1d_cov_test_C.npy` | 23 × 23 | covariance matrix of the estimates |
| `rq1d_cov_test_C_terms.csv` | 23 × 1 | 0 |

## `primary/rq2/`

Model comparison on window F3 (Table 5, Figures 5–7): metrics, implied effects, bootstrap summaries and pairs, implied-vs-causal curves, calibration bins and the review-interval analysis (RQ3, Figure 8). `rq2_folds_*`: three windows pooled (Table 6).

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_bootstrap_pairs.csv` | 22 × 8 | reference, model, dAUC, dAUC_lo, dAUC_hi, d_rmse, d_rmse_lo, d_rmse_hi |
| `rq2_bootstrap_summary.csv` | 12 × 12 | model, AUC, AUC_lo, AUC_hi, implied_all, implied_lo, implied_hi, rmse_lag, rmse_lo, rmse_hi, P(best AUC), P(best consistency) |
| `rq2_calibration_bins.csv` | 396 × 6 | model, subset, bin, mean_pred, obs, n |
| `rq2_consistency.csv` | 12 × 11 | model, implied_all, causal_all, bias_all, rmse_lag, rmse_lag_prev, coverage_lag, AUC, LogLoss, rank_AUC, rank_consistency |
| `rq2_consistency_logit_equal_curves.csv` | 108 × 4 | group, lag_bin, model, logodds |
| `rq2_consistency_logit_equal_pairs.csv` | 18 × 7 | group, reference, model, d_rmse, lo, hi, P_model_better |
| `rq2_consistency_logit_equal_summary.csv` | 12 × 9 | group, model, implied_logodds, implied_lo, implied_hi, rmse_logodds, rmse_lo, rmse_hi, causal_logodds_pooled |
| `rq2_consistency_pp_count_curves.csv` | 108 × 4 | group, lag_bin, model, logodds |
| `rq2_consistency_pp_count_pairs.csv` | 18 × 7 | group, reference, model, d_rmse, lo, hi, P_model_better |
| `rq2_consistency_pp_count_summary.csv` | 12 × 9 | group, model, implied_logodds, implied_lo, implied_hi, rmse_logodds, rmse_lo, rmse_hi, causal_logodds_pooled |
| `rq2_consistency_pp_equal_curves.csv` | 108 × 4 | group, lag_bin, model, logodds |
| `rq2_consistency_pp_equal_pairs.csv` | 18 × 7 | group, reference, model, d_rmse, lo, hi, P_model_better |
| `rq2_consistency_pp_equal_summary.csv` | 12 × 9 | group, model, implied_logodds, implied_lo, implied_hi, rmse_logodds, rmse_lo, rmse_hi, causal_logodds_pooled |
| `rq2_curves_vs_causal.csv` | 108 × 7 | model, subset, implied, causal, ci_low, ci_high, n |
| `rq2_folds_by_fold.csv` | 12 × 6 | fold, model, AUC, rmse_lag, causal_all, implied_all |
| `rq2_folds_by_fold_comonotone.csv` | 12 × 6 | fold, model, AUC, rmse_lag, causal_all, implied_all |
| `rq2_folds_consistency_by_group.csv` | 9 × 4 | group, model, rmse_pp, share_of_reexposures |
| `rq2_folds_curves.csv` | 108 × 7 | model, group, lag_bin, effect, ci_low, ci_high, n |
| `rq2_folds_pairs.csv` | 6 × 9 | reference, model, dAUC, dAUC_lo, dAUC_hi, d_rmse, d_rmse_lo, d_rmse_hi, P_model_more_consistent |
| `rq2_folds_pairs_comonotone.csv` | 6 × 9 | reference, model, dAUC, dAUC_lo, dAUC_hi, d_rmse, d_rmse_lo, d_rmse_hi, P_model_more_consistent |
| `rq2_folds_summary.csv` | 4 × 13 | model, AUC, AUC_lo, AUC_hi, implied_all, implied_lo, implied_hi, rmse_lag, rmse_lo, rmse_hi, P(best AUC), P(best consistency), causal_all_pooled |
| `rq2_folds_summary_comonotone.csv` | 4 × 13 | model, AUC, AUC_lo, AUC_hi, implied_all, implied_lo, implied_hi, rmse_lag, rmse_lo, rmse_hi, P(best AUC), P(best consistency), causal_all_pooled |
| `rq2_gbm_lag_shift.csv` | 9 × 4 | model, hours, mean_p, implied_effect_vs_unseen |
| `rq2_implied_effects.csv` | 196 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_implied_effects_das3h.csv` | 56 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_implied_effects_deep.csv` | 84 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_lagshift_DKT-time.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_lagshift_M5.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_lagshift_M6b.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_logit_curves.csv` | 108 × 4 | group, lag_bin, model, logodds |
| `rq2_logit_pairs.csv` | 18 × 7 | group, reference, model, d_rmse, lo, hi, P_model_better |
| `rq2_logit_summary.csv` | 12 × 9 | group, model, implied_logodds, implied_lo, implied_hi, rmse_logodds, rmse_lo, rmse_hi, causal_logodds_pooled |
| `rq2_metrics_full_F3.csv` | 11 × 16 | model, AUC, LogLoss, RMSE, ECE, ECE_reexposures, AUC_lo, LogLoss_lo, RMSE_lo, ECE_lo, ECE_reexposures_lo, AUC_hi, LogLoss_hi, RMSE_hi, ECE_hi, ECE_reexposure… |
| `rq2_metrics_full_pooled.csv` | 4 × 16 | model, AUC, LogLoss, RMSE, ECE, ECE_reexposures, AUC_lo, LogLoss_lo, RMSE_lo, ECE_lo, ECE_reexposures_lo, AUC_hi, LogLoss_hi, RMSE_hi, ECE_hi, ECE_reexposure… |
| `rq2_metrics_pairs.csv` | 8 × 18 | window, reference, model, dAUC, dLogLoss, dRMSE, dECE, dECE_reexposures, dAUC_lo, dLogLoss_lo, dRMSE_lo, dECE_lo, dECE_reexposures_lo, dAUC_hi, dLogLoss_hi, … |
| `rq2_metrics_test.csv` | 7 × 6 | model, n, AUC, RMSE, LogLoss, ECE |
| `rq2_metrics_test_das3h.csv` | 2 × 6 | model, n, AUC, RMSE, LogLoss, ECE |
| `rq2_metrics_test_deep.csv` | 3 × 12 | model, n, AUC, RMSE, LogLoss, ECE, val_AUC, dim, lr, dropout, window, reweight |
| `rq3_curves_asym-2_fine.csv` | 154 × 5 | group, source, hours, effect, effect_isotonic |
| `rq3_curves_fine.csv` | 154 × 5 | group, source, hours, effect, effect_isotonic |
| `rq3_policy_asym-2_fine.csv` | 16 × 11 | group, model, half_life_h, half_life_lo, half_life_hi, retained_at_review, retained_lo, retained_hi, review_rate_ratio, ratio_lo, ratio_hi |
| `rq3_policy_fine.csv` | 16 × 11 | group, model, half_life_h, half_life_lo, half_life_hi, retained_at_review, retained_lo, retained_hi, review_rate_ratio, ratio_lo, ratio_hi |

## `primary/rq2/fold_2025-07/`

Same outputs for test window F1 (July–September 2025).

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_gbm_lag_shift.csv` | 9 × 4 | model, hours, mean_p, implied_effect_vs_unseen |
| `rq2_implied_effects.csv` | 196 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_implied_effects_das3h.csv` | 56 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_implied_effects_deep.csv` | 28 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_lagshift_DKT-time.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_lagshift_M5.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_lagshift_M6b.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_metrics_test.csv` | 7 × 6 | model, n, AUC, RMSE, LogLoss, ECE |
| `rq2_metrics_test_das3h.csv` | 2 × 6 | model, n, AUC, RMSE, LogLoss, ECE |
| `rq2_metrics_test_deep.csv` | 1 × 12 | model, n, AUC, RMSE, LogLoss, ECE, val_AUC, dim, lr, dropout, window, reweight |
| `rq3_curves_fine.csv` | 154 × 5 | group, source, hours, effect, effect_isotonic |
| `rq3_policy_fine.csv` | 16 × 11 | group, model, half_life_h, half_life_lo, half_life_hi, retained_at_review, retained_lo, retained_hi, review_rate_ratio, ratio_lo, ratio_hi |

## `primary/rq2/fold_2025-10/`

Same outputs for test window F2 (October–December 2025).

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_gbm_lag_shift.csv` | 9 × 4 | model, hours, mean_p, implied_effect_vs_unseen |
| `rq2_implied_effects.csv` | 196 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_implied_effects_das3h.csv` | 56 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_implied_effects_deep.csv` | 28 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_lagshift_DKT-time.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_lagshift_M5.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_lagshift_M6b.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_metrics_test.csv` | 7 × 6 | model, n, AUC, RMSE, LogLoss, ECE |
| `rq2_metrics_test_das3h.csv` | 2 × 6 | model, n, AUC, RMSE, LogLoss, ECE |
| `rq2_metrics_test_deep.csv` | 1 × 12 | model, n, AUC, RMSE, LogLoss, ECE, val_AUC, dim, lr, dropout, window, reweight |
| `rq3_curves_fine.csv` | 154 × 5 | group, source, hours, effect, effect_isotonic |
| `rq3_policy_fine.csv` | 16 × 11 | group, model, half_life_h, half_life_lo, half_life_hi, retained_at_review, retained_lo, retained_hi, review_rate_ratio, ratio_lo, ratio_hi |

## `primary/stage_deep/F3/`

Implied effects of the deep models (DKT, SAKT, DKT-time) under the direct protocol, per window.

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_implied_effects_deep.csv` | 84 × 5 | model, subset, n, implied_effect, item_specific_path |

## `primary/stage_deep/fold_2025-07/`

Implied effects of the deep models (DKT, SAKT, DKT-time) under the direct protocol, per window.

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_implied_effects_deep.csv` | 28 × 5 | model, subset, n, implied_effect, item_specific_path |

## `primary/stage_deep/fold_2025-10/`

Implied effects of the deep models (DKT, SAKT, DKT-time) under the direct protocol, per window.

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_implied_effects_deep.csv` | 28 × 5 | model, subset, n, implied_effect, item_specific_path |

## `sensitivity_original_spec/balance/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `balance_by_strata.csv` | 54 × 11 | dimension, stratum, property, n, learners, observed, expected, diff, ci_low, ci_high, rel_diff |
| `positivity_by_experience.csv` | 5 × 14 | experience, count, mean, std, min, 5%, 25%, 50%, 75%, 95%, max, share_strictly_between_0_and_1, share_above_0_9, observed_seen_share |

## `sensitivity_original_spec/calib/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `calibration_slope_intercept_F3.csv` | 11 × 7 | model, intercept, intercept_lo, intercept_hi, slope, slope_lo, slope_hi |

## `sensitivity_original_spec/cf/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `cf_protocols_summary.csv` | 14 × 13 | model, protocol, implied_all, D_m_pp, imp <1h, imp 1-6h, imp 6-24h, imp 1-3d, imp 3-7d, imp 7-14d, imp 14-30d, imp 30-90d, imp >90d |
| `elo_leakage_gbm_metrics.csv` | 4 × 7 | elo, model, iters, AUC, RMSE, LogLoss, ECE |
| `pfa_item_share_of_kc_counts.csv` | 1 × 5 | n, mean_share, median_share, p90_share, median_kc_prior_n |

## `sensitivity_original_spec/cf_das3h_direct/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_implied_effects_das3h.csv` | 28 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_lagshift_M6b.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_metrics_test_das3h.csv` | 1 × 6 | model, n, AUC, RMSE, LogLoss, ECE |

## `sensitivity_original_spec/cf_das3h_direct/fold_2025-07/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_implied_effects_das3h.csv` | 28 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_lagshift_M6b.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_metrics_test_das3h.csv` | 1 × 6 | model, n, AUC, RMSE, LogLoss, ECE |

## `sensitivity_original_spec/cf_das3h_direct/fold_2025-10/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_implied_effects_das3h.csv` | 28 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_lagshift_M6b.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_metrics_test_das3h.csv` | 1 × 6 | model, n, AUC, RMSE, LogLoss, ECE |

## `sensitivity_original_spec/pigeonhole/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `pigeonhole_levels.csv` | 12 × 5 | model, group, D_pp, lo, hi |
| `pigeonhole_pairs.csv` | 4 × 17 | reference, model, dAUC, dAUC_lo, dAUC_hi, dD_all_pp, dD_all_lo, dD_all_hi, P_model_better_all, dD_after correct_pp, dD_after correct_lo, dD_after correct_hi,… |

## `sensitivity_original_spec/pooled_variants/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `pooled_variants_levels.csv` | 36 × 7 | variant, model, group, AUC, D_pp, lo, hi |
| `pooled_variants_pairs.csv` | 9 × 15 | variant, reference, model, dAUC, dAUC_lo, dAUC_hi, dD_all, dD_all_lo, dD_all_hi, dD_after correct, dD_after correct_lo, dD_after correct_hi, dD_after error, … |

## `sensitivity_original_spec/round_pools/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `counter_ratio_by_category.csv` | 26 × 5 | cat_id, median_ratio, R_c, items, expected_if_shared_pool |
| `counter_ratio_by_cohort.csv` | 2 × 4 | new_item, items, median_ratio, mean_ratio |

## `sensitivity_original_spec/rq1/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `lag_hours_by_activity.csv` | 3 × 9 | act_terc, count, mean, std, min, 25%, 50%, 75%, max |
| `review_rq1_effects.csv` | 117 × 5 | model, term, coef, se_learner, se_twoway |
| `review_rq1_info.csv` | 1 × 5 | model, n, learners, items, iters |
| `unrecorded_exposure_descriptives.csv` | 11 × 6 | group, n, mean_lambda, mean_p_unrecorded, share_p_gt_0_05, share_p_gt_0_25 |

## `sensitivity_original_spec/rq2_leakfree/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_gbm_lag_shift.csv` | 9 × 4 | model, hours, mean_p, implied_effect_vs_unseen |
| `rq2_implied_effects.csv` | 196 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_lagshift_M5.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_metrics_test.csv` | 7 × 6 | model, n, AUC, RMSE, LogLoss, ECE |
| `rq3_curves_fine.csv` | 70 × 5 | group, source, hours, effect, effect_isotonic |
| `rq3_policy_fine.csv` | 12 × 11 | group, model, half_life_h, half_life_lo, half_life_hi, retained_at_review, retained_lo, retained_hi, review_rate_ratio, ratio_lo, ratio_hi |

## `sensitivity_original_spec/rq2_leakfree/fold_2025-07/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_gbm_lag_shift.csv` | 9 × 4 | model, hours, mean_p, implied_effect_vs_unseen |
| `rq2_implied_effects.csv` | 196 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_lagshift_M5.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_metrics_test.csv` | 7 × 6 | model, n, AUC, RMSE, LogLoss, ECE |
| `rq3_curves_fine.csv` | 70 × 5 | group, source, hours, effect, effect_isotonic |
| `rq3_policy_fine.csv` | 12 × 11 | group, model, half_life_h, half_life_lo, half_life_hi, retained_at_review, retained_lo, retained_hi, review_rate_ratio, ratio_lo, ratio_hi |

## `sensitivity_original_spec/rq2_leakfree/fold_2025-10/`

Original specification (per-response Elo updates, full-history deletion) and robustness checks reported in Appendices A–B.

| File | Rows × columns | Columns |
|---|---|---|
| `rq2_gbm_lag_shift.csv` | 9 × 4 | model, hours, mean_p, implied_effect_vs_unseen |
| `rq2_implied_effects.csv` | 196 × 5 | model, subset, n, implied_effect, item_specific_path |
| `rq2_lagshift_M5.csv` | 42 × 5 | model, prev_y, hours, n, implied |
| `rq2_metrics_test.csv` | 7 × 6 | model, n, AUC, RMSE, LogLoss, ECE |
| `rq3_curves_fine.csv` | 70 × 5 | group, source, hours, effect, effect_isotonic |
| `rq3_policy_fine.csv` | 12 × 11 | group, model, half_life_h, half_life_lo, half_life_hi, retained_at_review, retained_lo, retained_hi, review_rate_ratio, ratio_lo, ratio_hi |

# Causal validation of knowledge-tracing models with naturally randomized item exposure

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23392742.svg)](https://doi.org/10.5281/zenodo.23392742)

Code and aggregated results for the article

> Kulatay, A.; Zhaisanova, D.; Kulambayev, B.; Mambetov, S.; Zhaksylyk, A.; Karataeva, Z.; Skakova, A. *Causal Validation of Knowledge-Tracing Models with Naturally Randomized Item Exposure in Game-Based Learning.* Machine Learning and Knowledge Extraction (submitted).

The study uses a natural experiment in the logs of a game-based quiz platform, where questions are drawn at random within a learner-chosen category. It estimates the causal effect of a previous encounter with an item and its decay over time. Fourteen knowledge-tracing (KT) models then answer the same counterfactual query, and their implied effects are compared with the causal reference (*causal inconsistency* D_m). The repository contains every script of the analysis and all aggregated results that back the tables and figures of the article.

## Data availability

The response data are **not included**. They were provided by the platform operator and contain personal information of platform users, so they cannot be shared publicly. The scripts run on a MySQL export of the platform database (gzip CSV files, NULL = `\N`), and access for replication can be requested from the corresponding author with the approval of the data provider.

The `results/` folder contains only aggregated outputs: effect estimates, covariance matrices of estimates, model-implied curves, metrics and bootstrap summaries. No file contains learner, game or item identifiers or individual responses. Record-level intermediate files (parquet, npz) and trained model weights were excluded because they are derived from the personal data.

## Repository layout

```
code/                      all analysis scripts (Python 3.12); each script header documents its inputs and outputs
results/primary/           primary specification of the article (leak-free Elo, direct item-memory protocol)
  design_checks/           randomization and design checks (Section 3.3, Figure 3)
  rq1/                     causal effects of re-exposure and time (RQ1, Section 4.2, Figure 4)
  rq2/                     model comparison, implied effects, bootstrap (RQ2, Section 4.3, Tables 5-7, Figures 5-7)
  rq2/fold_2025-07, fold_2025-10   rolling-origin test windows F1 and F2 (F3 = rq2/)
  stage_deep/              deep models under the direct protocol (summary tables)
  deep_modern/             AKT and HawkesKT (Tables 5-6, Figure 5)
  review/                  two-way bootstrap and calibration slopes (Appendix B)
results/sensitivity_original_spec/   original specification and robustness checks (Appendices A-B)
results/descriptive/       preprocessing funnel (Table 1) and descriptive statistics
results/README.md          inventory of every result file with its size and columns
figures/                   Figures 1-8 of the article (PNG)
```

## Where the numbers in the article come from

| Article | Files |
|---|---|
| Table 1 (preprocessing funnel) | `results/descriptive/preprocessing_funnel.csv` |
| Figure 3, Section 4.1 (design checks) | `results/primary/design_checks/randomization_balance.csv`, `deck_global_recency.csv`; `results/sensitivity_original_spec/balance/`, `round_pools/` |
| Section 4.2, Figure 4 (RQ1) | `results/primary/rq1/rq1_effects*.csv`, `rq1d_clogit*.csv`, `rq1b_*.csv`, `rq1c_*.csv`, `rq1_cov_*.npy` |
| Table 5, Figure 5a (window F3) | `results/primary/rq2/rq2_bootstrap_summary.csv`, `results/primary/deep_modern/F3_boot/rq2_bootstrap_summary.csv`, `results/primary/deep_modern/F3/modern_f3.csv` |
| Table 6, Figure 5b (three windows pooled) | `results/primary/rq2/rq2_folds_summary.csv`, `rq2_folds_pairs.csv`, `rq2_folds_consistency_by_group.csv`, `results/primary/deep_modern/modern_levels.csv`, `modern_pairs.csv` |
| Figure 6 (implied vs. causal curves) | `results/primary/rq2/rq2_folds_curves.csv`, `results/primary/deep_modern/modern_curves.csv` |
| Table 7, Figure 7 (predictive quality) | `results/primary/rq2/rq2_metrics_full_F3.csv`, `rq2_metrics_full_pooled.csv`, `rq2_metrics_pairs.csv`, `rq2_calibration_bins.csv`, `results/primary/review/calib/` |
| Figure 8, Section 4.4 (RQ3) | `results/primary/rq2/rq3_policy_fine.csv` and the same file in `fold_2025-07/`, `fold_2025-10/` |
| Appendix A (log-odds scale, weighting, tuning) | `results/primary/rq2/rq2_logit_*.csv`, `rq2_consistency_*.csv` |
| Appendix B (protocols, Elo correction, two-way bootstrap, window dependence) | `results/sensitivity_original_spec/`, `results/primary/review/pigeonhole/`, `results/primary/rq2/rq2_folds_*_comonotone.csv` |

## Environment

Python 3.12 with the packages in `requirements.txt` (exact versions used). The deep models (DKT, DKT-time, SAKT, AKT, HawkesKT) were trained with PyTorch 2.11 and CUDA 12.8 on an NVIDIA RTX 5070 Ti laptop GPU. All other stages run on a CPU with 32 GB of RAM. All random seeds are fixed (42), and the pipeline is deterministic given the seed.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # on Windows: .venv\Scripts\pip install -r requirements.txt
```

## Running the analysis

Set `EXPORT` to the unpacked database export, `WORK` to a working directory for the original specification and `WORK2` to the directory of the primary specification. Run the commands in this order from the repository root. Each script prints its usage when called with wrong arguments, and its header explains inputs, outputs and design decisions.

**Stage 1. Preprocessing (Section 3.2, Table 1).** Deterministic rules P1–P6, complete records only, no imputation.

```bash
python code/preprocess.py $EXPORT $WORK/clean
python code/eda.py $WORK/clean $WORK/eda
```

**Stage 2. Design checks (Section 3.3).**

```bash
python code/randomization_check.py $WORK/clean $EXPORT $WORK/design_checks 0.1
python code/randomization_recency.py $WORK/clean $EXPORT $WORK/design_checks 0.2
python code/randomization_deck.py $WORK/clean $EXPORT $WORK/design_checks 0.2
python code/randomization_outcome_lag.py $WORK/clean $EXPORT $WORK/design_checks 0.2
python code/review_balance.py $WORK/clean $EXPORT $WORK/review/balance 0.1
python code/review_round_pools.py $EXPORT $WORK/review/round_pools
```

**Stage 3. Causal reference (RQ1, Section 3.4).** Whole period, the three test windows and the fine lag grid used by RQ3.

```bash
python code/rq1_causal.py $WORK/clean $EXPORT $WORK/rq1
python code/rq1_causal.py $WORK/clean $EXPORT $WORK/rq1 --test-period                          # F3
python code/rq1_causal.py $WORK/clean $EXPORT $WORK/rq1 --test-period --window 2025-07-01 2025-10-01   # F1
python code/rq1_causal.py $WORK/clean $EXPORT $WORK/rq1 --test-period --window 2025-10-01 2026-01-01   # F2
python code/rq1_causal.py $WORK/clean $EXPORT $WORK/rq1 --fine-lags
python code/rq1b_decay_interference.py $WORK/clean $EXPORT $WORK/rq1
python code/rq1c_error_sensitivity.py $WORK/clean $EXPORT $WORK/rq1
python code/rq1d_clogit.py $WORK/clean $EXPORT $WORK/rq1
python code/rq1d_clogit.py $WORK/clean $EXPORT $WORK/rq1 --window 2026-01-01 2026-04-01   # and the F1, F2 windows
python code/review_rq1_checks.py $WORK/clean $EXPORT $WORK/review/rq1
```

**Stage 4. KT models (RQ2, Section 3.5).** Window F3 (default split) and the rolling-origin windows F1 and F2.

```bash
python code/rq2_models.py $WORK/clean $WORK/rq2
python code/rq2_models.py $WORK/clean $WORK/rq2 --fold 2025-04-01 2025-07-01 2025-10-01
python code/rq2_models.py $WORK/clean $WORK/rq2 --fold 2025-07-01 2025-10-01 2026-01-01
python code/rq2_das3h.py  $WORK/clean $WORK/rq2                 # same --fold options
python code/rq2_deep_data.py $WORK/clean $WORK/rq2              # token cache; same --fold options
python code/rq2_deep.py $WORK/clean $WORK/rq2                   # DKT, DKT-time, SAKT (GPU)
python code/rq2_deep.py $WORK/clean $WORK/rq2/fold_2025-07 --models DKT-time
python code/rq2_deep.py $WORK/clean $WORK/rq2/fold_2025-10 --models DKT-time
python code/rq3_deep_lagshift.py $WORK/rq2                      # and $WORK/rq2/fold_2025-07, fold_2025-10
```

**Stage 5. Primary specification (leak-free Elo, one counterfactual protocol for all models).**

```bash
python code/review_rq2_leakfree.py $WORK/clean $WORK/review/rq2_leakfree           # and --fold ... for F1, F2
python code/review_das3h_direct.py $WORK/clean $WORK/review/cf_das3h_direct 2000   # and --fold ... for F1, F2
python code/review_deep_direct_rows.py $WORK/rq2 $WORK2/stage_deep/F3
python code/review_deep_direct_rows.py $WORK/rq2/fold_2025-07 $WORK2/stage_deep/fold_2025-07 --models DKT-time
python code/review_deep_direct_rows.py $WORK/rq2/fold_2025-10 $WORK2/stage_deep/fold_2025-10 --models DKT-time
python code/review_build_work_v2.py $WORK $WORK2
```

**Stage 6. Causal consistency, predictive quality and review intervals (RQ2–RQ3, Sections 3.6–3.7)**, on the primary specification.

```bash
python code/rq2_consistency.py $WORK2/rq1 $WORK2/rq2
python code/rq2_bootstrap.py $WORK2/rq1 $WORK2/rq2 500
python code/rq2_bootstrap_folds.py $WORK2/rq1 $WORK2/rq2 500
python code/rq2_folds_curves.py $WORK2/rq1 $WORK2/rq2
python code/rq2_metrics_bootstrap.py $WORK2/rq2 500
python code/rq2_logit_consistency.py $WORK2/rq1 $WORK2/rq2 500 --scale logit --weights count   # and pp/equal variants
python code/rq3_policy.py $WORK2/rq1 $WORK2/rq2 2000 --fine
python code/rq3_policy.py $WORK2/rq1 $WORK2/rq2 2000 --fine --models-dir $WORK2/rq2/fold_2025-07
python code/rq3_policy.py $WORK2/rq1 $WORK2/rq2 2000 --fine --models-dir $WORK2/rq2/fold_2025-10
python code/review_pigeonhole.py $WORK2/rq1 $WORK2/rq2 $WORK2/review/pigeonhole 500
python code/review_calibration_slope.py $WORK2/rq2 $WORK2/review/calib 200
python code/review_window_dependence.py $WORK2/rq1 $WORK2/rq2 500
```

**Stage 7. AKT and HawkesKT (Section 4.3).**

```bash
python code/rq2_deep_modern.py $WORK/rq2 $WORK2/deep_modern/F3
python code/rq2_deep_modern.py $WORK/rq2/fold_2025-07 $WORK2/deep_modern/fold_2025-07
python code/rq2_deep_modern.py $WORK/rq2/fold_2025-10 $WORK2/deep_modern/fold_2025-10
python code/review_modern_kt.py $WORK2/rq1 $WORK2/rq2 $WORK2/deep_modern $WORK2/deep_modern 500
python code/review_modern_f3.py $WORK2/rq1 $WORK2/rq2 $WORK2/deep_modern/F3
```

**Sensitivity analyses of the original specification (Appendix B)** run the Stage 6 scripts on `$WORK` instead of `$WORK2`, plus `review_cf_deep.py`, `review_cf_tabular.py`, `review_cf_summary.py` and `review_pooled_variants.py` (see their headers).

**Figures.** `python code/make_figures.py $WORK2 <out_dir>` writes Figures 1–8 (TIFF and PNG, 1200 dpi).

## License

Code: MIT License (see `LICENSE`). Aggregated results and figures: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Citation

If you use this code or the results, please cite the article above and the archived version of this repository: https://doi.org/10.5281/zenodo.23392742 (all versions; version 1.0.0: https://doi.org/10.5281/zenodo.23392743). See also `CITATION.cff`.

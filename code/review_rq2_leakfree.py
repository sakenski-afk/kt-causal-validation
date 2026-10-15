"""
Leak-free replication of the RQ2 tabular models: rq2_models.py with the online Elo states updated only after all
responses of a game (review_cf_tabular.elo_batched) instead of response by response.

    python review_rq2_leakfree.py <clean_dir> <out_root> [--fold T_VAL T_TEST END]

Everything else (features, models, hyper-parameters, seed, outputs) is that of rq2_models.py; the pipeline
scripts are not modified (the Elo function is replaced at run time). Outputs: <out_root> (F3) or <out_root>/fold_<yyyy-mm>.
"""
import sys

import baselines as B
import review_cf_tabular as T
import rq2_models as R

if __name__ == "__main__":
    B.elo = T.elo_batched
    tv, tt, en, out = R.parse_fold(sys.argv, sys.argv[2])
    B.T_VAL, B.T_TEST, R.END = (tv, tt, en) if tv else ("2025-10-01", "2026-01-01", "2026-04-01")
    R.main(sys.argv[1], out, 500)

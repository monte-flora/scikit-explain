"""
Benchmark suite for scikit-explain performance optimization.

Run: python tests/benchmark_suite.py
Outputs timing results for all major compute methods.
"""

import numpy as np
import pandas as pd
import time
import json
import sys
import warnings
from sklearn.ensemble import RandomForestClassifier
import skexplain

warnings.filterwarnings("ignore")


def make_dataset(n_samples, n_features=10, random_state=42):
    np.random.seed(random_state)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = (X["f0"] * 2 + X["f1"] > 0).astype(int).values
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=6, random_state=random_state, n_jobs=1
    )
    rf.fit(X, y)
    return X, y, rf


def bench(label, fn, n_runs=3):
    """Run fn n_runs times and return median time."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    median = sorted(times)[len(times) // 2]
    print(f"  {label}: {median:.4f}s (median of {n_runs})")
    return median


def run_benchmarks(n_samples=2000):
    print(f"\n{'='*60}")
    print(f"Benchmarks: {n_samples} samples, 10 features, 50-tree RF")
    print(f"{'='*60}")

    X, y, rf = make_dataset(n_samples)
    exp = skexplain.ExplainToolkit([("RF", rf)], X=X, y=y)

    results = {}

    results["predict_proba_100x"] = bench(
        "Raw predict_proba ×100",
        lambda: [rf.predict_proba(X.values) for _ in range(100)],
    )

    results["perm_imp_5v_5p"] = bench(
        "Perm Imp (5 vars, 5 permutes)",
        lambda: exp.permutation_importance(n_vars=5, evaluation_fn="auc", n_permute=5),
    )

    results["ale_1d_all_1boot"] = bench(
        "ALE 1D (all, 20 bins, 1 boot)",
        lambda: exp.ale(features="all", n_bins=20),
    )

    results["ale_1d_all_10boot"] = bench(
        "ALE 1D (all, 20 bins, 10 boot)",
        lambda: exp.ale(features="all", n_bins=20, n_bootstrap=10),
    )

    results["pd_1d_3feat_1boot"] = bench(
        "PD 1D (3 feat, 20 bins, 1 boot)",
        lambda: exp.pd(features=["f0", "f1", "f2"], n_bins=20),
    )

    results["pd_1d_3feat_10boot"] = bench(
        "PD 1D (3 feat, 20 bins, 10 boot)",
        lambda: exp.pd(features=["f0", "f1", "f2"], n_bins=20, n_bootstrap=10),
    )

    results["ice_2feat_20bins"] = bench(
        "ICE (2 feat, 20 bins, 100 sub)",
        lambda: exp.ice(features=["f0", "f1"], n_bins=20, subsample=100),
    )

    results["ale_2d_1pair"] = bench(
        "2D ALE (1 pair, 15 bins)",
        lambda: exp.ale(features=[("f0", "f1")], n_bins=15),
    )

    return results


if __name__ == "__main__":
    all_results = {}
    for n in [2000]:
        all_results[n] = run_benchmarks(n)

    print(f"\n{'='*60}")
    print("Summary (seconds)")
    print(f"{'='*60}")
    for method, t in all_results[2000].items():
        print(f"  {method}: {t:.4f}s")

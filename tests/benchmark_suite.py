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


def run_stress_test():
    """Heavy benchmark: 10000 samples, 30 features, 100 trees."""
    N, F, T = 10000, 30, 100
    print(f"\n{'='*60}")
    print(f"STRESS TEST: {N} samples, {F} features, {T}-tree RF")
    print(f"{'='*60}")

    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(N, F),
        columns=[f"f{i}" for i in range(F)],
    )
    y = (X["f0"] * 2 + X["f1"] - X["f2"] * 0.5 > 0).astype(int).values
    rf = RandomForestClassifier(
        n_estimators=T, max_depth=8, random_state=42, n_jobs=1,
    )
    rf.fit(X, y)
    exp = skexplain.ExplainToolkit([("RF", rf)], X=X, y=y)

    results = {}

    # Baseline: raw predict overhead
    results["predict_proba_10x"] = bench(
        f"Raw predict_proba ×10 ({N} samples)",
        lambda: [rf.predict_proba(X.values) for _ in range(10)],
        n_runs=3,
    )

    # Permutation importance
    results["perm_imp_10v_10p"] = bench(
        "Perm Imp (10 vars, 10 permutes)",
        lambda: exp.permutation_importance(n_vars=10, evaluation_fn="auc", n_permute=10),
        n_runs=2,
    )

    # ALE
    results["ale_1d_all_1boot"] = bench(
        f"ALE 1D (all {F} features, 30 bins, 1 boot)",
        lambda: exp.ale(features="all", n_bins=30),
        n_runs=2,
    )

    results["ale_1d_all_10boot"] = bench(
        f"ALE 1D (all {F} features, 30 bins, 10 boot)",
        lambda: exp.ale(features="all", n_bins=30, n_bootstrap=10),
        n_runs=2,
    )

    results["ale_1d_10feat_20boot"] = bench(
        "ALE 1D (10 features, 30 bins, 20 boot)",
        lambda: exp.ale(features=[f"f{i}" for i in range(10)], n_bins=30, n_bootstrap=20),
        n_runs=2,
    )

    # PD
    results["pd_1d_5feat_1boot"] = bench(
        "PD 1D (5 feat, 30 bins, 1 boot)",
        lambda: exp.pd(features=[f"f{i}" for i in range(5)], n_bins=30),
        n_runs=2,
    )

    results["pd_1d_5feat_10boot"] = bench(
        "PD 1D (5 feat, 30 bins, 10 boot)",
        lambda: exp.pd(features=[f"f{i}" for i in range(5)], n_bins=30, n_bootstrap=10),
        n_runs=2,
    )

    results["pd_1d_5feat_20boot"] = bench(
        "PD 1D (5 feat, 30 bins, 20 boot)",
        lambda: exp.pd(features=[f"f{i}" for i in range(5)], n_bins=30, n_bootstrap=20),
        n_runs=2,
    )

    # ICE
    results["ice_3feat_30bins_200sub"] = bench(
        "ICE (3 feat, 30 bins, 200 sub)",
        lambda: exp.ice(features=["f0", "f1", "f2"], n_bins=30, subsample=200),
        n_runs=2,
    )

    # 2D ALE
    results["ale_2d_1pair_20bins"] = bench(
        "2D ALE (1 pair, 20 bins)",
        lambda: exp.ale(features=[("f0", "f1")], n_bins=20),
        n_runs=2,
    )

    results["ale_2d_3pairs_15bins"] = bench(
        "2D ALE (3 pairs, 15 bins)",
        lambda: exp.ale(features=[("f0", "f1"), ("f0", "f2"), ("f1", "f2")], n_bins=15),
        n_runs=2,
    )

    # Parallel comparison
    results["ale_1d_all_1boot_2jobs"] = bench(
        f"ALE 1D (all {F}, 30 bins, 1 boot, n_jobs=2)",
        lambda: exp.ale(features="all", n_bins=30, n_jobs=2),
        n_runs=2,
    )

    results["pd_1d_5feat_10boot_2jobs"] = bench(
        "PD 1D (5 feat, 30 bins, 10 boot, n_jobs=2)",
        lambda: exp.pd(features=[f"f{i}" for i in range(5)], n_bins=30, n_bootstrap=10, n_jobs=2),
        n_runs=2,
    )

    return results


if __name__ == "__main__":
    # Standard benchmark
    std_results = run_benchmarks(2000)

    # Stress test
    stress_results = run_stress_test()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("\nStandard (2000 samples, 10 features, 50 trees):")
    for method, t in std_results.items():
        print(f"  {method}: {t:.4f}s")
    print(f"\nStress (10000 samples, 30 features, 100 trees):")
    for method, t in stress_results.items():
        print(f"  {method}: {t:.4f}s")

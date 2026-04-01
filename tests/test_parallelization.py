"""
Test that parallelization (n_jobs > 1) works correctly across all methods.

Compares results from serial (n_jobs=1) and parallel (n_jobs=2) execution
to ensure they produce equivalent outputs.
"""

import unittest
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import skexplain

# Suppress warnings during tests
warnings.filterwarnings("ignore")


class TestParallelization(unittest.TestCase):
    """Test n_jobs > 1 across all methods that support it."""

    @classmethod
    def setUpClass(cls):
        """Create shared test fixtures once for all tests."""
        # Classification dataset
        np.random.seed(42)
        n_samples, n_features = 300, 6
        X_cls = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"f{i}" for i in range(n_features)],
        )
        y_cls = (X_cls["f0"] + X_cls["f1"] > 0).astype(int).values
        cls.X_cls = X_cls
        cls.y_cls = y_cls

        cls.rf_cls = RandomForestClassifier(
            n_estimators=15, max_depth=4, random_state=42, n_jobs=1
        )
        cls.rf_cls.fit(cls.X_cls, cls.y_cls)
        cls.estimators_cls = [("RF", cls.rf_cls)]

        # Regression dataset
        np.random.seed(42)
        X_reg = pd.DataFrame(
            np.random.randn(n_samples, 4),
            columns=["X0", "X1", "X2", "X3"],
        )
        y_reg = X_reg["X0"] * 3 + X_reg["X1"] * 2 + np.random.randn(n_samples) * 0.1
        cls.X_reg = X_reg
        cls.y_reg = y_reg.values

        cls.rf_reg = RandomForestRegressor(
            n_estimators=15, max_depth=4, random_state=42, n_jobs=1
        )
        cls.rf_reg.fit(cls.X_reg, cls.y_reg)
        cls.estimators_reg = [("RFR", cls.rf_reg)]

    def _make_explainer(self, estimators, X, y):
        return skexplain.ExplainToolkit(estimators=estimators, X=X, y=y)

    # ========== ALE ==========

    def test_ale_parallel(self):
        """ALE: n_jobs=2 produces same results as n_jobs=1."""
        explainer = self._make_explainer(self.estimators_cls, self.X_cls, self.y_cls)
        features = ["f0", "f1", "f2"]

        ale_serial = explainer.ale(features=features, n_bins=10, n_jobs=1, random_seed=42)
        ale_parallel = explainer.ale(features=features, n_bins=10, n_jobs=2, random_seed=42)

        for feat in features:
            var_name = f"{feat}__RF__ale"
            np.testing.assert_allclose(
                ale_serial[var_name].values,
                ale_parallel[var_name].values,
                rtol=1e-5,
                err_msg=f"ALE mismatch for {feat}",
            )

    def test_ale_njobs_minus1(self):
        """ALE: n_jobs=-1 (all CPUs) runs without error."""
        explainer = self._make_explainer(self.estimators_reg, self.X_reg, self.y_reg)
        ale = explainer.ale(features=["X0", "X1"], n_bins=10, n_jobs=-1, random_seed=42)
        self.assertIn("X0__RFR__ale", ale.data_vars)

    def test_ale_njobs_fraction(self):
        """ALE: n_jobs=0.5 (50% of CPUs) runs without error."""
        explainer = self._make_explainer(self.estimators_reg, self.X_reg, self.y_reg)
        ale = explainer.ale(features=["X0", "X1"], n_bins=10, n_jobs=0.5, random_seed=42)
        self.assertIn("X0__RFR__ale", ale.data_vars)

    def test_ale_regression_parallel(self):
        """ALE regression: n_jobs=2 produces same results as n_jobs=1."""
        explainer = self._make_explainer(self.estimators_reg, self.X_reg, self.y_reg)
        features = ["X0", "X1", "X2"]

        ale_serial = explainer.ale(features=features, n_bins=10, n_jobs=1, random_seed=42)
        ale_parallel = explainer.ale(features=features, n_bins=10, n_jobs=2, random_seed=42)

        for feat in features:
            var_name = f"{feat}__RFR__ale"
            np.testing.assert_allclose(
                ale_serial[var_name].values,
                ale_parallel[var_name].values,
                rtol=1e-5,
                err_msg=f"ALE regression mismatch for {feat}",
            )

    # ========== PD ==========

    def test_pd_parallel(self):
        """PD: n_jobs=2 produces same results as n_jobs=1."""
        explainer = self._make_explainer(self.estimators_cls, self.X_cls, self.y_cls)
        features = ["f0", "f1", "f2"]

        pd_serial = explainer.pd(features=features, n_bins=10, n_jobs=1, random_seed=42)
        pd_parallel = explainer.pd(features=features, n_bins=10, n_jobs=2, random_seed=42)

        for feat in features:
            var_name = f"{feat}__RF__pd"
            np.testing.assert_allclose(
                pd_serial[var_name].values,
                pd_parallel[var_name].values,
                rtol=1e-5,
                err_msg=f"PD mismatch for {feat}",
            )

    # ========== ICE ==========

    def test_ice_parallel(self):
        """ICE: n_jobs=2 produces same results as n_jobs=1."""
        explainer = self._make_explainer(self.estimators_cls, self.X_cls, self.y_cls)
        features = ["f0", "f1"]

        ice_serial = explainer.ice(
            features=features, n_bins=10, n_jobs=1, subsample=30, random_seed=42
        )
        ice_parallel = explainer.ice(
            features=features, n_bins=10, n_jobs=2, subsample=30, random_seed=42
        )

        for feat in features:
            var_name = f"{feat}__RF__ice"
            np.testing.assert_allclose(
                ice_serial[var_name].values,
                ice_parallel[var_name].values,
                rtol=1e-5,
                err_msg=f"ICE mismatch for {feat}",
            )

    # ========== Permutation Importance ==========

    def test_permutation_importance_parallel(self):
        """Permutation Importance: n_jobs=2 runs without error and produces rankings."""
        explainer = self._make_explainer(self.estimators_cls, self.X_cls, self.y_cls)

        pi_serial = explainer.permutation_importance(
            n_vars=4, evaluation_fn="auc", n_jobs=1, n_permute=2, random_seed=42
        )
        pi_parallel = explainer.permutation_importance(
            n_vars=4, evaluation_fn="auc", n_jobs=2, n_permute=2, random_seed=42
        )

        # Both should produce rankings
        self.assertIn("backward_multipass_rankings__RF", pi_serial.data_vars)
        self.assertIn("backward_multipass_rankings__RF", pi_parallel.data_vars)

        # Rankings should match
        np.testing.assert_array_equal(
            pi_serial["backward_multipass_rankings__RF"].values,
            pi_parallel["backward_multipass_rankings__RF"].values,
            err_msg="PI rankings differ between serial and parallel",
        )

    # ========== Perm-Based Interaction ==========

    def test_perm_based_interaction_parallel(self):
        """Perm-based interaction: n_jobs=2 runs without error."""
        explainer = self._make_explainer(self.estimators_reg, self.X_reg, self.y_reg)

        features = [("X0", "X1"), ("X0", "X2")]

        inter_serial = explainer.perm_based_interaction(
            features=features, evaluation_fn="mse", n_jobs=1, n_bootstrap=1
        )
        inter_parallel = explainer.perm_based_interaction(
            features=features, evaluation_fn="mse", n_jobs=2, n_bootstrap=1
        )

        self.assertIn("perm_based_interactions_rankings__RFR", inter_serial.data_vars)
        self.assertIn("perm_based_interactions_rankings__RFR", inter_parallel.data_vars)

    # ========== Local Attributions ==========

    def test_local_attributions_parallel(self):
        """Tree Interpreter: n_jobs=2 produces same results as n_jobs=1."""
        X_sub = self.X_cls.iloc[:20]
        y_sub = self.y_cls[:20]

        exp_s = self._make_explainer(self.estimators_cls, X_sub, y_sub)
        exp_p = self._make_explainer(self.estimators_cls, X_sub, y_sub)

        result_s = exp_s.local_attributions(method="tree_interpreter", n_jobs=1)
        result_p = exp_p.local_attributions(method="tree_interpreter", n_jobs=2)

        np.testing.assert_allclose(
            result_s["tree_interpreter_values__RF"].values,
            result_p["tree_interpreter_values__RF"].values,
            rtol=1e-5,
            err_msg="TreeInterpreter mismatch serial vs parallel",
        )

    # ========== Grouped Permutation Importance ==========

    def test_grouped_perm_importance_parallel(self):
        """Grouped PI: n_jobs=2 runs without error."""
        explainer = self._make_explainer(self.estimators_reg, self.X_reg, self.y_reg)

        groups = {"group1": ["X0", "X1"], "group2": ["X2", "X3"]}

        result = explainer.grouped_permutation_importance(
            perm_method="grouped",
            evaluation_fn="mse",
            groups=groups,
            n_permute=2,
            n_jobs=2,
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)

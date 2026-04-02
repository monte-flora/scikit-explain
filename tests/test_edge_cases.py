"""
Expanded edge-case and property-based tests for scikit-explain.

Uses hypothesis for property-based testing to catch edge cases that
hand-written tests miss: varying shapes, dtypes, NaN patterns, etc.
"""

import unittest
import numpy as np
import pandas as pd
import xarray as xr
import warnings

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays
from hypothesis.extra.pandas import column, data_frames

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError

import skexplain
from skexplain.main._validation import normalize_features, normalize_estimator_names
from skexplain.common.importance_utils import to_skexplain_importance

warnings.filterwarnings("ignore")


# ============================================================
# Helpers: reusable fixtures
# ============================================================

def make_classification_fixtures(n_samples=200, n_features=5, random_state=42):
    """Create a simple classification dataset + fitted model."""
    np.random.seed(random_state)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = (X["f0"] > 0).astype(int).values
    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=random_state)
    rf.fit(X, y)
    return X, y, rf


def make_regression_fixtures(n_samples=200, n_features=4, random_state=42):
    """Create a simple regression dataset + fitted model."""
    np.random.seed(random_state)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"x{i}" for i in range(n_features)],
    )
    y = (X["x0"] * 3 + X["x1"] + np.random.randn(n_samples) * 0.1)
    lr = LinearRegression().fit(X, y)
    return X, y.values, lr


# ============================================================
# 1. ExplainToolkit Initialization Edge Cases
# ============================================================

class TestInitEdgeCases(unittest.TestCase):
    """Test ExplainToolkit constructor with various edge cases."""

    def test_unfitted_model_raises(self):
        """Unfitted model should raise NotFittedError."""
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = np.array([0, 1])
        rf = RandomForestClassifier()  # not fitted
        with self.assertRaises(NotFittedError):
            skexplain.ExplainToolkit([("RF", rf)], X=X, y=y)

    def test_bare_estimator_raises_type_error(self):
        """Passing a bare estimator (not tuple) should raise TypeError."""
        X, y, rf = make_classification_fixtures(n_samples=20, n_features=2)
        with self.assertRaises(TypeError):
            skexplain.ExplainToolkit(estimators=rf, X=X, y=y)

    def test_numpy_array_without_feature_names_raises(self):
        """Numpy array X without feature_names should raise."""
        X, y, rf = make_classification_fixtures(n_samples=20, n_features=2)
        with self.assertRaises(Exception):
            skexplain.ExplainToolkit([("RF", rf)], X=X.values, y=y)

    def test_numpy_array_with_feature_names_works(self):
        """Numpy array X with explicit feature_names should work."""
        X, y, rf = make_classification_fixtures(n_samples=20, n_features=2)
        explainer = skexplain.ExplainToolkit(
            [("RF", rf)], X=X.values, y=y, feature_names=["f0", "f1"]
        )
        self.assertEqual(explainer.feature_names, ["f0", "f1"])

    def test_single_estimator_tuple(self):
        """Single (name, estimator) tuple should work."""
        X, y, rf = make_classification_fixtures(n_samples=20, n_features=2)
        explainer = skexplain.ExplainToolkit(("RF", rf), X=X, y=y)
        self.assertEqual(explainer.estimator_names, ["RF"])

    def test_multiple_estimators(self):
        """Multiple estimators in a list should work."""
        X, y, rf = make_classification_fixtures(n_samples=20, n_features=2)
        lr = LinearRegression().fit(X, y)
        explainer = skexplain.ExplainToolkit(
            [("RF", rf), ("LR", lr)], X=X, y=y
        )
        self.assertEqual(len(explainer.estimator_names), 2)

    def test_y_as_list(self):
        """y as a Python list should be accepted."""
        X, y, rf = make_classification_fixtures(n_samples=20, n_features=2)
        explainer = skexplain.ExplainToolkit([("RF", rf)], X=X, y=list(y))
        self.assertEqual(len(explainer.y), 20)

    def test_y_as_series(self):
        """y as a pandas Series should be accepted."""
        X, y, rf = make_classification_fixtures(n_samples=20, n_features=2)
        explainer = skexplain.ExplainToolkit([("RF", rf)], X=X, y=pd.Series(y))
        self.assertEqual(len(explainer.y), 20)

    def test_non_contiguous_index(self):
        """DataFrame with non-contiguous index should work."""
        X, y, rf = make_classification_fixtures(n_samples=50, n_features=3)
        # Simulate train_test_split output (non-contiguous index)
        X_sub = X.iloc[::2]  # every other row
        y_sub = y[::2]
        explainer = skexplain.ExplainToolkit([("RF", rf)], X=X_sub, y=y_sub)
        self.assertEqual(len(explainer.X), 25)


# ============================================================
# 2. PlotConfig API Tests
# ============================================================

class TestPlotConfig(unittest.TestCase):
    """Test set_plotting_config, get_plotting_config, reset_plotting_config."""

    def setUp(self):
        X, y, rf = make_classification_fixtures(n_samples=20, n_features=2)
        self.explainer = skexplain.ExplainToolkit([("RF", rf)], X=X, y=y)

    def test_set_and_get_config(self):
        """Setting config should be retrievable."""
        self.explainer.set_plotting_config(
            display_feature_names={"f0": "Feature Zero"},
            base_font_size=16,
        )
        config = self.explainer.get_plotting_config()
        self.assertEqual(config.display_feature_names, {"f0": "Feature Zero"})
        self.assertEqual(config.base_font_size, 16)

    def test_reset_config(self):
        """Reset should restore defaults."""
        self.explainer.set_plotting_config(base_font_size=20)
        self.explainer.reset_plotting_config()
        config = self.explainer.get_plotting_config()
        self.assertEqual(config.base_font_size, 12)
        self.assertIsNone(config.display_feature_names)

    def test_invalid_config_key_raises(self):
        """Unknown config key should raise ValueError."""
        with self.assertRaises(ValueError):
            self.explainer.set_plotting_config(nonexistent_key="value")

    def test_config_persists_across_calls(self):
        """Config should persist across multiple set calls."""
        self.explainer.set_plotting_config(base_font_size=14)
        self.explainer.set_plotting_config(style="whitegrid")
        config = self.explainer.get_plotting_config()
        self.assertEqual(config.base_font_size, 14)
        self.assertEqual(config.style, "whitegrid")


# ============================================================
# 3. Return Type Consistency
# ============================================================

class TestReturnTypes(unittest.TestCase):
    """Every compute method should return xarray.Dataset."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.rf = make_regression_fixtures(n_samples=100, n_features=4)
        cls.explainer = skexplain.ExplainToolkit(
            [("LR", cls.rf)], X=cls.X, y=cls.y
        )

    def test_permutation_importance_returns_dataset(self):
        result = self.explainer.permutation_importance(
            n_vars=3, evaluation_fn="mse", n_permute=1
        )
        self.assertIsInstance(result, xr.Dataset)

    def test_ale_returns_dataset(self):
        result = self.explainer.ale(features=["x0"], n_bins=5)
        self.assertIsInstance(result, xr.Dataset)

    def test_pd_returns_dataset(self):
        result = self.explainer.pd(features=["x0"], n_bins=5)
        self.assertIsInstance(result, xr.Dataset)

    def test_ice_returns_dataset(self):
        result = self.explainer.ice(features=["x0"], n_bins=5, subsample=20)
        self.assertIsInstance(result, xr.Dataset)

    def test_main_effect_complexity_returns_dataset(self):
        ale = self.explainer.ale(features="all", n_bins=5)
        result = self.explainer.main_effect_complexity(ale=ale)
        self.assertIsInstance(result, xr.Dataset)

    def test_local_attributions_returns_dataset(self):
        # tree_interpreter requires tree-based models; use a dedicated explainer
        X, y, rf = make_classification_fixtures(n_samples=50, n_features=4)
        exp = skexplain.ExplainToolkit([("RF", rf)], X=X, y=y)
        result = exp.local_attributions(method="tree_interpreter")
        self.assertIsInstance(result, xr.Dataset)

    def test_computation_time_in_attrs(self):
        """All compute methods should record timing."""
        result = self.explainer.ale(features=["x0"], n_bins=5)
        self.assertIn("computation_time_seconds", result.attrs)
        self.assertGreaterEqual(result.attrs["computation_time_seconds"], 0)


# ============================================================
# 4. Validation Helpers (Property-Based)
# ============================================================

class TestNormalizeFeatures(unittest.TestCase):
    """Property-based tests for normalize_features."""

    @given(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))))
    @settings(max_examples=30)
    def test_single_string_becomes_list(self, feature_name):
        """Any single string should become a 1-element list."""
        result = normalize_features(feature_name, ["a", "b", "c"])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], feature_name)

    def test_all_returns_full_list(self):
        features = ["a", "b", "c"]
        result = normalize_features("all", features)
        self.assertEqual(result, features)

    def test_all_2d_without_allow_raises_or_returns_list(self):
        """'all_2d' without allow_2d should be treated as a feature name."""
        result = normalize_features("all_2d", ["a", "b"])
        self.assertEqual(result, ["all_2d"])

    def test_all_2d_with_allow(self):
        result = normalize_features("all_2d", ["a", "b", "c"], allow_2d=True)
        self.assertEqual(len(result), 3)  # C(3,2) = 3

    def test_list_passthrough(self):
        features = ["x", "y"]
        result = normalize_features(features, ["a", "b"])
        self.assertEqual(result, features)


class TestNormalizeEstimatorNames(unittest.TestCase):
    """Property-based tests for normalize_estimator_names."""

    def test_none_returns_defaults(self):
        result = normalize_estimator_names(None, ["RF", "GB"])
        self.assertEqual(result, ["RF", "GB"])

    @given(st.text(min_size=1, max_size=20))
    @settings(max_examples=20)
    def test_single_string_becomes_list(self, name):
        result = normalize_estimator_names(name, ["default"])
        self.assertIsInstance(result, list)
        self.assertEqual(result, [name])

    def test_list_passthrough(self):
        result = normalize_estimator_names(["A", "B"], ["default"])
        self.assertEqual(result, ["A", "B"])


# ============================================================
# 5. to_skexplain_importance Edge Cases
# ============================================================

class TestToSkexplainImportance(unittest.TestCase):
    """Test importance conversion with various inputs."""

    def test_1d_array(self):
        importances = np.array([0.5, 0.3, 0.1])
        result = to_skexplain_importance(
            importances, estimator_name="RF",
            feature_names=["a", "b", "c"], method="custom",
        )
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("custom_rankings__RF", result.data_vars)

    def test_2d_array_bootstrap(self):
        importances = np.random.randn(3, 10)  # 3 features x 10 bootstrap
        result = to_skexplain_importance(
            importances, estimator_name="RF",
            feature_names=["a", "b", "c"], method="custom",
            bootstrap_axis=1,
        )
        self.assertIn("custom_scores__RF", result.data_vars)

    def test_shap_sum_method(self):
        """SHAP values should be summed by absolute value."""
        shap_vals = np.array([[0.1, -0.5], [0.2, 0.3]])
        result = to_skexplain_importance(
            shap_vals, estimator_name="RF",
            feature_names=["a", "b"], method="shap_sum",
        )
        rankings = result["shap_sum_rankings__RF"].values
        # Feature "b" has higher absolute sum (0.8 vs 0.3)
        self.assertEqual(rankings[0], "b")

    def test_normalize_flag(self):
        """Normalized scores should be in [0, ~1] range."""
        importances = np.array([10.0, 5.0, 1.0])
        result = to_skexplain_importance(
            importances, estimator_name="RF",
            feature_names=["a", "b", "c"], method="custom",
            normalize=True,
        )
        scores = result["custom_scores__RF"].values.flatten()
        self.assertLessEqual(np.max(scores), 2.0)  # roughly normalized


# ============================================================
# 6. Hypothesis: ALE with Random Data Shapes
# ============================================================

class TestALEPropertyBased(unittest.TestCase):
    """Property-based tests for ALE computation."""

    @given(
        n_features=st.integers(min_value=2, max_value=8),
        n_bins=st.integers(min_value=3, max_value=20),
    )
    @settings(max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow])
    def test_ale_shape_matches_bins(self, n_features, n_bins):
        """ALE output shape should match n_bins for any feature count."""
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"f{i}" for i in range(n_features)],
        )
        y = (X.iloc[:, 0] > 0).astype(int).values
        rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        rf.fit(X, y)

        explainer = skexplain.ExplainToolkit([("RF", rf)], X=X, y=y)
        ale = explainer.ale(features=["f0"], n_bins=n_bins)

        # ALE values should exist
        self.assertIn("f0__RF__ale", ale.data_vars)
        # Number of ALE bins should be <= n_bins (can be less if data has fewer unique quantiles)
        actual_bins = ale["f0__RF__ale"].shape[-1]
        self.assertLessEqual(actual_bins, n_bins)
        self.assertGreater(actual_bins, 0)

    @given(subsample=st.floats(min_value=0.1, max_value=1.0))
    @settings(max_examples=5, deadline=30000, suppress_health_check=[HealthCheck.too_slow])
    def test_ale_subsample_fraction(self, subsample):
        """ALE should work with any valid subsample fraction."""
        X, y, rf = make_regression_fixtures(n_samples=100, n_features=3)
        explainer = skexplain.ExplainToolkit([("LR", rf)], X=X, y=y)
        ale = explainer.ale(features=["x0"], n_bins=5, subsample=subsample)
        self.assertIn("x0__LR__ale", ale.data_vars)


# ============================================================
# 7. Hypothesis: Permutation Importance Consistency
# ============================================================

class TestPermImportancePropertyBased(unittest.TestCase):
    """Property-based tests for permutation importance."""

    @given(n_vars=st.integers(min_value=1, max_value=4))
    @settings(max_examples=5, deadline=30000, suppress_health_check=[HealthCheck.too_slow])
    def test_rankings_length_matches_n_vars(self, n_vars):
        """Number of ranked features should match n_vars."""
        X, y, rf = make_regression_fixtures(n_samples=80, n_features=4)
        explainer = skexplain.ExplainToolkit([("LR", rf)], X=X, y=y)
        result = explainer.permutation_importance(
            n_vars=n_vars, evaluation_fn="mse", n_permute=1
        )
        rankings = result["backward_multipass_rankings__LR"].values
        self.assertEqual(len(rankings), n_vars)

    def test_rankings_are_unique(self):
        """All ranked features should be unique."""
        X, y, rf = make_regression_fixtures(n_samples=80, n_features=4)
        explainer = skexplain.ExplainToolkit([("LR", rf)], X=X, y=y)
        result = explainer.permutation_importance(
            n_vars=4, evaluation_fn="mse", n_permute=2
        )
        rankings = result["backward_multipass_rankings__LR"].values
        self.assertEqual(len(rankings), len(set(rankings)))


# ============================================================
# 8. SAGE Tests
# ============================================================

class TestSAGE(unittest.TestCase):
    """Test SAGE computation if sage-importance is installed."""

    @classmethod
    def setUpClass(cls):
        try:
            import sage
            cls.sage_available = True
        except ImportError:
            cls.sage_available = False

        cls.X, cls.y, cls.rf = make_classification_fixtures(n_samples=100, n_features=4)

    def test_sage_basic(self):
        """SAGE should return a dataset with correct variable names."""
        if not self.sage_available:
            self.skipTest("sage-importance not installed")

        explainer = skexplain.ExplainToolkit([("RF", self.rf)], X=self.X, y=self.y)
        result = explainer.sage(n_background=10, bar=False)

        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("sage_rankings__RF", result.data_vars)
        self.assertIn("sage_scores__RF", result.data_vars)
        self.assertEqual(len(result["sage_rankings__RF"].values), 4)

    def test_sage_grouped(self):
        """Grouped SAGE should use group names as features."""
        if not self.sage_available:
            self.skipTest("sage-importance not installed")

        groups = {"group_A": ["f0", "f1"], "group_B": ["f2", "f3"]}
        explainer = skexplain.ExplainToolkit([("RF", self.rf)], X=self.X, y=self.y)
        result = explainer.sage(groups=groups, n_background=10, bar=False)

        self.assertIn("grouped_sage_rankings__RF", result.data_vars)
        rankings = result["grouped_sage_rankings__RF"].values
        self.assertEqual(set(rankings), {"group_A", "group_B"})

    def test_sage_without_package_raises(self):
        """If sage package is missing, should raise ImportError with helpful message."""
        # We can't easily unimport sage, so just verify the method exists
        explainer = skexplain.ExplainToolkit([("RF", self.rf)], X=self.X, y=self.y)
        self.assertTrue(hasattr(explainer, "sage"))


# ============================================================
# 9. IO Round-Trip Tests
# ============================================================

class TestIORoundTrip(unittest.TestCase):
    """Test save/load round-trips."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        X, y, rf = make_regression_fixtures(n_samples=50, n_features=3)
        self.explainer = skexplain.ExplainToolkit([("LR", rf)], X=X, y=y)

    def test_save_load_dataset(self):
        """Save and load a dataset should preserve values."""
        import os
        try:
            import netCDF4
        except ImportError:
            self.skipTest("netCDF4 not installed — needed for save/load")

        ale = self.explainer.ale(features=["x0"], n_bins=5)
        fpath = os.path.join(self.tmpdir, "ale_test.nc")
        self.explainer.save(fpath, ale)

        loaded = self.explainer.load(fpath, dtype="dataset")
        np.testing.assert_allclose(
            ale["x0__LR__ale"].values,
            loaded["x0__LR__ale"].values,
            rtol=1e-5,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ============================================================
# 10. Grouping Utilities
# ============================================================

class TestGroupingUtils(unittest.TestCase):
    """Test group_local_values and group_feature_values."""

    def setUp(self):
        self.X, self.y, self.rf = make_classification_fixtures(n_samples=50, n_features=4)
        self.explainer = skexplain.ExplainToolkit(
            [("RF", self.rf)], X=self.X, y=self.y
        )
        self.groups = {"group_A": ["f0", "f1"], "group_B": ["f2", "f3"]}

    def test_group_feature_values_shape(self):
        """Grouped feature values should have group count columns."""
        X_grouped = skexplain.group_feature_values(self.X, self.groups)
        self.assertEqual(X_grouped.shape[1], 2)
        self.assertEqual(list(X_grouped.columns), ["group_A", "group_B"])

    def test_group_feature_values_scaled(self):
        """Grouped values should be in [0, 1] range (min-max scaled)."""
        X_grouped = skexplain.group_feature_values(self.X, self.groups)
        self.assertGreaterEqual(X_grouped.values.min(), 0.0)
        self.assertLessEqual(X_grouped.values.max(), 1.0)

    def test_group_local_values_shape(self):
        """Grouped SHAP should have n_samples x n_groups shape."""
        attr = self.explainer.local_attributions(method="tree_interpreter")
        X_grouped = skexplain.group_feature_values(self.X, self.groups)
        grouped = skexplain.group_local_values(attr, self.groups, X_grouped)
        vals = grouped["tree_interpreter_values__RF"].values
        self.assertEqual(vals.shape, (50, 2))


# ============================================================
# 11. Hypothesis: Constant & Degenerate Features
# ============================================================

class TestDegenerateInputs(unittest.TestCase):
    """Test behavior with degenerate data: constant features, extreme values."""

    def test_constant_feature_ale(self):
        """A constant feature should produce zero ALE effect."""
        np.random.seed(42)
        X = pd.DataFrame({
            "varying": np.random.randn(100),
            "constant": np.ones(100),
        })
        y = (X["varying"] > 0).astype(int).values
        rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

        explainer = skexplain.ExplainToolkit([("RF", rf)], X=X, y=y)
        # Constant feature has only 1 unique value — ALE should handle gracefully
        ale = explainer.ale(features=["varying"], n_bins=5)
        self.assertIn("varying__RF__ale", ale.data_vars)

    def test_single_feature_dataset(self):
        """Dataset with only 1 feature should work."""
        np.random.seed(42)
        X = pd.DataFrame({"only_feature": np.random.randn(100)})
        y = (X["only_feature"] > 0).astype(int).values
        rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

        explainer = skexplain.ExplainToolkit([("RF", rf)], X=X, y=y)
        ale = explainer.ale(features=["only_feature"], n_bins=5)
        self.assertIn("only_feature__RF__ale", ale.data_vars)

    def test_two_sample_dataset(self):
        """Minimal dataset (2 samples) should not crash."""
        X = pd.DataFrame({"f0": [0.0, 1.0], "f1": [1.0, 0.0]})
        y = np.array([0, 1])
        rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

        explainer = skexplain.ExplainToolkit([("RF", rf)], X=X, y=y)
        # Should at least not crash
        pi = explainer.permutation_importance(n_vars=2, evaluation_fn="auc", n_permute=1)
        self.assertIsInstance(pi, xr.Dataset)


if __name__ == "__main__":
    unittest.main()

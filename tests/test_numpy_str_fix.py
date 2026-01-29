"""
Test for numpy.str_ vs str type fix (Issue #70)

This tests the fix for SHAP compatibility where SHAP's convert_name function
uses type(ind) == str rather than isinstance(ind, str), causing numpy.str_
types to fail.

See: https://github.com/shap/shap/issues/3304
"""
import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skexplain.common.utils import ensure_str_features
import skexplain


class TestNumpyStrFix(unittest.TestCase):
    """Test that feature names are Python strings, not numpy.str_"""

    def test_ensure_str_features_from_pandas_index(self):
        """Test conversion from pandas Index"""
        df = pd.DataFrame({'feat1': [1, 2], 'feat2': [3, 4]})
        result = ensure_str_features(df.columns)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for name in result:
            self.assertEqual(type(name), str)  # Must be exactly str, not numpy.str_
            self.assertNotEqual(type(name).__name__, 'str_')  # Not numpy.str_

    def test_ensure_str_features_from_numpy_array(self):
        """Test conversion from numpy array (which may contain numpy.str_)"""
        arr = np.array(['feat1', 'feat2'])
        result = ensure_str_features(arr)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for name in result:
            self.assertEqual(type(name), str)  # Must be exactly str
            self.assertNotEqual(type(name).__name__, 'str_')

    def test_ensure_str_features_from_list(self):
        """Test that regular lists are handled correctly"""
        lst = ['feat1', 'feat2']
        result = ensure_str_features(lst)

        self.assertIsInstance(result, list)
        for name in result:
            self.assertEqual(type(name), str)

    def test_ensure_str_features_with_unicode(self):
        """Test with unicode feature names"""
        df = pd.DataFrame({'féat1': [1, 2], 'feat_2': [3, 4]})
        result = ensure_str_features(df.columns)

        for name in result:
            self.assertEqual(type(name), str)
        self.assertIn('féat1', result)

    def test_feature_names_in_explaintoolkit(self):
        """Test that ExplainToolkit stores feature names as Python strings"""
        np.random.seed(42)
        n_samples = 100

        # Create DataFrame with features that might become numpy.str_
        X = pd.DataFrame({
            'feature_A': np.random.randn(n_samples),
            'feature_B': np.random.randn(n_samples),
            'feature_C': np.random.randn(n_samples),
        })
        y = pd.Series((X['feature_A'] + X['feature_B'] > 0).astype(int), name='target')

        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        # Create ExplainToolkit
        explainer = skexplain.ExplainToolkit(
            ('RF', model),
            X=X,
            y=y,
        )

        # Check that feature_names are Python strings
        self.assertIsInstance(explainer.feature_names, list)
        for name in explainer.feature_names:
            self.assertEqual(type(name), str,
                f"Feature name '{name}' is type {type(name)}, not str")
            self.assertNotEqual(type(name).__name__, 'str_',
                f"Feature name '{name}' is numpy.str_, not Python str")

    def test_feature_names_from_numpy_array_input(self):
        """Test feature names when X is a numpy array"""
        np.random.seed(42)
        n_samples = 50

        X_array = np.random.randn(n_samples, 3)
        feature_names = ['feat_1', 'feat_2', 'feat_3']
        y = pd.Series(np.random.randint(0, 2, n_samples), name='target')

        # Convert to DataFrame with feature names
        X = pd.DataFrame(X_array, columns=feature_names)

        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        explainer = skexplain.ExplainToolkit(
            ('RF', model),
            X=X,
            y=y,
        )

        # Verify feature names are Python strings
        for name in explainer.feature_names:
            self.assertEqual(type(name), str)

    def test_shap_compatibility_simulation(self):
        """
        Simulate SHAP's convert_name behavior to ensure compatibility.

        SHAP uses: type(ind) == str
        This test verifies our feature names pass this check.
        """
        df = pd.DataFrame({'feat1': [1, 2], 'feat2': [3, 4]})
        feature_names = ensure_str_features(df.columns)

        # Simulate SHAP's type check
        for name in feature_names:
            # This is what SHAP does - it must pass
            self.assertTrue(type(name) == str,
                f"SHAP compatibility check failed: type({name}) == str returned False")

            # isinstance would work with numpy.str_, but SHAP doesn't use it
            self.assertTrue(isinstance(name, str))

    def test_feature_names_preserved_through_list_operations(self):
        """Test that feature names remain Python strings when using list indexing"""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})

        # Get feature names through our utility
        features = ensure_str_features(df.columns)

        # After indexing with list comprehension (like in sobol ranking fix)
        indices = np.array([2, 0, 1])
        ranked_features = [features[i] for i in indices]

        # Should be Python strings
        for name in ranked_features:
            self.assertEqual(type(name), str)

    def test_numpy_array_converts_to_numpy_str(self):
        """Test that demonstrates numpy arrays convert strings to numpy.str_"""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

        # Start with Python strings
        features = ensure_str_features(df.columns)
        for name in features:
            self.assertEqual(type(name), str)

        # Converting to numpy array changes type to numpy.str_
        features_array = np.array(features)
        for name in features_array:
            # This demonstrates the problem - numpy converts str to numpy.str_
            self.assertTrue(type(name).__name__ == 'str_' or type(name) == np.str_)

        # Solution: convert back to list
        features_fixed = ensure_str_features(features_array)
        for name in features_fixed:
            self.assertEqual(type(name), str)


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure the fix doesn't break existing functionality"""

    def test_normal_workflow_still_works(self):
        """Test that normal ExplainToolkit workflow still works"""
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            'x1': np.random.randn(n_samples),
            'x2': np.random.randn(n_samples),
        })
        y = pd.Series((X['x1'] + X['x2'] > 0).astype(int), name='y')

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        explainer = skexplain.ExplainToolkit(('RF', model), X=X, y=y)

        # Should work without errors
        self.assertEqual(len(explainer.feature_names), 2)
        self.assertIn('x1', explainer.feature_names)
        self.assertIn('x2', explainer.feature_names)


if __name__ == '__main__':
    unittest.main()

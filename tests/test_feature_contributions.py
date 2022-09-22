#===================================================
# Unit test for the feature contribution 
# code in Scikit-Explain.
#===================================================

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import shap

import sys, os 
sys.path.insert(0, os.path.dirname(os.getcwd()))
import skexplain

from tests import TestSingleExampleContributions


class TestFeatureContributions(TestSingleExampleContributions):
    def test_tree_interpreter_single_example(self):
        # Test a single example with tree interpreter.
        contrib_ds = self.explainer.local_attributions(method="tree_interpreter")

    def test_lime_single_example(self):
        # Test a single example with LIME. 
        lime_kws = {'training_data' : self.X.values}
        contrib_ds = self.explainer.local_attributions(method="lime", 
                                                        lime_kws=lime_kws,
                                                       )

    def test_plot_ti_contributions(self):
        # Test plotting the treeinterpret results.
        # Should be vaild for SHAP as well.
        for method in ["tree_interpreter"]:
            contrib_ds = self.explainer.local_attributions(method=method)
            # Test that the parallelization code runs. 
            contrib_ds = self.explainer.local_attributions(method=method, n_jobs=2)    
            
            
        self.explainer.plot_contributions(
            contrib=contrib_ds,
        )

    def test_perform_based_contributions(self):
        explainer = skexplain.ExplainToolkit(
            estimators=self.rf_estimator,
            X=self.X,
            y=self.y,
        )

        for method in ["tree_interpreter", "shap", 'lime']:
            contrib_ds = explainer.average_attributions(
                method=method,
                performance_based=True,
                n_samples=10,
                lime_kws={'training_data' : self.X.values}, 
                shap_kws={
                    "masker": shap.maskers.Partition(
                        self.X, max_samples=100, clustering="correlation"
                    ),
                    "algorithm": "permutation",
                   
                },
            )

    def test_bad_method(self):
        with self.assertRaises(Exception) as ex:
            contrib_ds = self.explainer.local_attributions( method="nonsense", )
        except_msg = f"Invalid method (nonsense)! Method must be one of the following: 'shap', 'tree_interpreter', 'lime'"
        self.assertEqual(ex.exception.args[0], except_msg)


if __name__ == "__main__":
    unittest.main()

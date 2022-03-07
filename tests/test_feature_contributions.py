# Unit test for the feature contribution code in scikit-explain
import unittest
import os, sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import shap

sys.path.append(os.path.dirname(os.getcwd()))

import skexplain


class TestExplainToolkit(unittest.TestCase):
    def setUp(self):
        estimators = skexplain.load_models()
        X, y = skexplain.load_data()
        X = X.astype({'urban': 'category', 'rural':'category'})
        
        self.X = X
        self.y = y
        self.estimators = estimators
        
        self.random_state=np.random.RandomState(42)
        
        # Fit a simple 5-variable random forest regressor estimator. 
        n_X = 2000
        n_vars = 5 
        weights = [2.0, 1.5, 1.2, 0.7, 0.2]
        X = np.stack([self.random_state.uniform(-1,1, size=n_X) for _ in range(n_vars)], axis=-1)
        feature_names = [f'X_{i+1}' for i in range(n_vars)]
        X = pd.DataFrame(X, columns=feature_names)
        y = X.dot(weights)
        
        rf = RandomForestRegressor()
        rf.fit(X,y)
        
        self.X=X
        self.y=y
        self.rf = rf 
        self.rf_estimator_name = 'Random Forest'
        self.weights=weights
        
        # Computing SHAP values in scikit-explain. 
        X_sub = self.X.iloc[[100]]
        X_sub.reset_index(inplace=True, drop=True)
        self.X_sub = X_sub
        
        self.explainer = skexplain.ExplainToolkit(
                estimators=(self.rf_estimator_name, self.rf),
                X=X_sub,
            )
        
        
class TestFeatureContributions(TestExplainToolkit):
    def test_tree_interpreter_single_example(self):
        # Test a single example with tree interpreter. 
        contrib_ds = self.explainer.local_contributions(method='tree_interpreter')
    
    def test_plot_ti_contributions(self):
        # Test plotting the treeinterpret results. 
        # Should be vaild for SHAP as well. 
        for method in ['tree_interpreter', ]:                             
            contrib_ds = self.explainer.local_contributions(method=method)
    
        self.explainer.plot_contributions(
                contrib = contrib_ds, )
    
    def test_perform_based_contributions(self):
        explainer = skexplain.ExplainToolkit(
                estimators=(self.rf_estimator_name, self.rf),
                X=self.X,
                y=self.y, 
            )
        
        for method in ['tree_interpreter', 'shap']:                             
            contrib_ds = explainer.local_contributions(method=method, 
                                                            performance_based=True, 
                                                            n_samples=10, 
                                                            shap_kwargs={'masker' : 
                                      shap.maskers.Partition(self.X, max_samples=100, clustering="correlation"), 
                                     'algorithm' : 'permutation'}
                                                           )

    def test_bad_method(self):
        with self.assertRaises(Exception) as ex:                   
            contrib_ds = self.explainer.local_contributions(method='nonsense', 
                                                            performance_based=True, 
                                                            n_samples=10)
        except_msg = "Invalid method! Method must be 'shap' or 'tree_interpreter'"
        self.assertEqual(ex.exception.args[0], except_msg)
                            
    
if __name__ == "__main__":
    unittest.main()
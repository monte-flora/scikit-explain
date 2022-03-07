# Unit test for the feature ranking code in scikit-explain
import unittest
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
import shap

sys.path.append(os.path.dirname(os.getcwd()))

import skexplain
from skexplain.common.importance_utils import to_skexplain_importance

class TestExplainToolkit(unittest.TestCase):
    def setUp(self):
        estimators = skexplain.load_models()
        X, y = skexplain.load_data()
        X = X.astype({'urban': 'category', 'rural':'category'})
        
        self.X = X
        self.y = y
        self.estimators = estimators
        
        random_state=np.random.RandomState(42)
        
        # Fit a simple 5-variable linear regression estimator. 
        n_X = 2000
        n_vars = 5 
        weights = [2.0, 1.5, 1.2, 0.7, 0.2]
        X = np.stack([random_state.uniform(-1,1, size=n_X) for _ in range(n_vars)], axis=-1)
        feature_names = [f'X_{i+1}' for i in range(n_vars)]
        self.feature_names = feature_names
        X = pd.DataFrame(X, columns=feature_names)
        y = X.dot(weights)
        
        lr = LinearRegression()
        lr.fit(X,y)
        
        self.X=X
        self.y=y
        self.lr = lr 
        self.lr_estimator_name = 'Linear Regression'
        self.weights=weights
        
class TestRankings(TestExplainToolkit):
    def test_bad_evaluation_fn(self):
        # Make sure the metrics are correct
        explainer = skexplain.ExplainToolkit(
                estimators=(self.lr_estimator_name, self.lr),
                X=self.X,
                y=self.y
            )
        available_scores = ["auc", "auprc", "bss", "mse", "norm_aupdc"]
        with self.assertRaises(Exception) as ex:
            explainer.permutation_importance(n_vars=len(self.X.columns), 
                                                  evaluation_fn='bad')
        
        
        except_msg = f"evaluation_fn is not set! Available options are {available_scores}"
        self.assertEqual(ex.exception.args[0], except_msg)
     
    def test_custom_evaluation_fn(self):
        # scoring_strategy exception for custom evaluation funcs 
        explainer = skexplain.ExplainToolkit(
                estimators=(self.lr_estimator_name, self.lr),
                X=self.X,
                y=self.y
            )
        available_scores = ["auc", "auprc", "bss", "mse", "norm_aupdc"]
        with self.assertRaises(Exception) as ex:
            explainer.permutation_importance(n_vars=len(self.X.columns), 
                                                  evaluation_fn=roc_auc_score,)
        
        except_msg = """ 
                The scoring_strategy argument is None! If you are using a non-default evaluation_fn 
                then scoring_strategy must be set! If the metric is positively-oriented (a higher value is better), 
                then set scoring_strategy = "argmin_of_mean" and if it is negatively-oriented-
                (a lower value is better), then set scoring_strategy = "argmax_of_mean"
                """
        self.assertEqual(ex.exception.args[0], except_msg)
        
    def test_shape(self):
        # Shape is correct (with bootstrapping) 
        explainer = skexplain.ExplainToolkit(
                estimators=(self.lr_estimator_name, self.lr),
                X=self.X,
                y=self.y
            )
        n_vars=3
        n_permute=8
        results = explainer.permutation_importance(n_vars=n_vars, 
                                                  evaluation_fn='mse',
                                                 n_permute=n_permute)
        # shape should be (n_vars_multipass, n_permute)
        self.assertEqual( results[f'multipass_scores__{self.lr_estimator_name}'].values.shape,
                          (n_vars, n_permute) 
                        )
        # shape should be (n_vars_singlepass, n_permute)
        self.assertEqual( results[f'singlepass_scores__{self.lr_estimator_name}'].values.shape,
                          (len(self.X.columns), n_permute) 
                        )
        
    def test_correct_rankings(self):
        # rankings are correct for simple case (for multi-pass, single-pass, and ale_variance)
        explainer = skexplain.ExplainToolkit(
                estimators=(self.lr_estimator_name, self.lr),
                X=self.X,
                y=self.y
            )

        ale = explainer.ale(features=self.X.columns, n_bins=10)
        ale_var_results = explainer.ale_variance(ale, estimator_names=self.lr_estimator_name)

        # TODO: coefficients
        shap_results = explainer.shap(shap_kwargs={'masker' : 
                                      shap.maskers.Partition(self.X, max_samples=100, clustering="correlation"), 
                                     'algorithm' : 'permutation'})
        
        # Implicit test of the to_sklearn_importance method. 
        shap_imp = to_skexplain_importance(shap_results[f'shap_values__{self.lr_estimator_name}'].values, 
                                           estimator_name=self.lr_estimator_name,
                                           feature_names = self.feature_names, method='shap_sum')
        
        true_rankings = np.array( ['X_1', 'X_2', 'X_3', 'X_4', 'X_5'])

        # Check the single-pass and multi-pass permutation importance (both forward and backward)
        for direction in ['backward', 'forward']:
            results = explainer.permutation_importance(n_vars=len(self.X.columns), 
                                                  evaluation_fn='mse',
                                                 n_permute=10, direction=direction)
        
            np.testing.assert_array_equal( results[f'multipass_rankings__{self.lr_estimator_name}'].values,
                          true_rankings
                        )
            np.testing.assert_array_equal( results[f'singlepass_rankings__{self.lr_estimator_name}'].values,
                          true_rankings
                        )
            
        # Check the ALE variance. 
        np.testing.assert_array_equal( ale_var_results[f'ale_variance_rankings__{self.lr_estimator_name}'].values,
                         true_rankings
                        )

        # Check the SHAP. 
        np.testing.assert_array_equal( shap_imp[f'shap_sum_rankings__{self.lr_estimator_name}'].values,
                         true_rankings
                        )
        
 
    def test_ale_variance(self):
        explainer = skexplain.ExplainToolkit(
                estimators=(self.lr_estimator_name, self.lr),
                X=self.X,
                y=self.y
            )
        
        ale = explainer.ale(features=self.X.columns, n_bins=10)
        ale_var_results = explainer.ale_variance(ale)
        
        with self.assertRaises(Exception) as ex_1:
            ale_var_results = explainer.ale_variance(ale=np.array([0,0]))
            
        except_msg_1 = """
                                 ale must be an xarray.Dataset, 
                                 perferably generated by ExplainToolkit.ale 
                                 to be formatted correctly
                                 """
        self.assertEqual(ex_1.exception.args[0], except_msg_1)
        
        with self.assertRaises(Exception) as ex_2:
            ale_var_results = explainer.ale_variance(ale, estimator_names=[self.lr_estimator_name, 'Fake'])
        
        except_msg_2 = 'ale does not contain values for all the estimator names given!'
        
        self.assertEqual(ex_2.exception.args[0], except_msg_2)

    def test_grouped_importance(self):
        pass
    
    def test_to_skexplain_importance(self):
        pass
    
        
if __name__ == "__main__":
    unittest.main()

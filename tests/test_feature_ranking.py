#===================================================
# Unit test for the feature ranking
# code in Scikit-Explain.
#===================================================
from sklearn.metrics import roc_auc_score
import shap
import numpy as np


import skexplain
from skexplain.common.importance_utils import to_skexplain_importance

from tests import TestLR, TestRF

TRUE_RANKINGS = np.array(["X_1", "X_2", "X_3", "X_4", "X_5"], dtype=object)

class TestRankings(TestLR, TestRF):
    def test_bad_evaluation_fn(self):
        # Make sure the metrics are correct
        explainer = skexplain.ExplainToolkit(
            estimators=self.lr_estimator, X=self.X, y=self.y
        )
        available_scores = ["auc", "auprc", "bss", "mse", "norm_aupdc"]
        with self.assertRaises(Exception) as ex:
            explainer.permutation_importance(
                n_vars=len(self.X.columns), evaluation_fn="bad"
            )

        except_msg = (
            f"evaluation_fn is not set! Available options are {available_scores}"
        )
        self.assertEqual(ex.exception.args[0], except_msg)

    def test_custom_evaluation_fn(self):
        # scoring_strategy exception for custom evaluation funcs
        explainer = skexplain.ExplainToolkit(
            estimators=(self.lr_estimator_name, self.lr), X=self.X, y=self.y
        )
        available_scores = ["auc", "auprc", "bss", "mse", "norm_aupdc"]
        with self.assertRaises(Exception) as ex:
            explainer.permutation_importance(
                n_vars=len(self.X.columns),
                evaluation_fn=roc_auc_score,
            )

        except_msg = """ 
                The scoring_strategy argument is None! If you are using a non-default evaluation_fn 
                then scoring_strategy must be set! If the metric is positively-oriented (a higher value is better), 
                then set scoring_strategy = "argmin_of_mean" and if it is negatively-oriented-
                (a lower value is better), then set scoring_strategy = "argmax_of_mean"
                """
        self.assertMultiLineEqual(ex.exception.args[0], except_msg)
        

    def test_shape(self):
        # Shape is correct (with bootstrapping)
        explainer = skexplain.ExplainToolkit(
            estimators=(self.lr_estimator_name, self.lr), X=self.X, y=self.y
        )
        n_vars = 3
        n_permute = 8
        results = explainer.permutation_importance(
            n_vars=n_vars, evaluation_fn="mse", n_permute=n_permute
        )
        # shape should be (n_vars_multipass, n_permute)
        self.assertEqual(
            results[f"backward_multipass_scores__{self.lr_estimator_name}"].values.shape,
            (n_vars, n_permute),
        )
        # shape should be (n_vars_singlepass, n_permute)
        self.assertEqual(
            results[f"backward_singlepass_scores__{self.lr_estimator_name}"].values.shape,
            (len(self.X.columns), n_permute),
        )

    def test_correct_rankings(self):
        # rankings are correct for simple case (for multi-pass, single-pass, and ale_variance)
        explainer = skexplain.ExplainToolkit(
            estimators=self.lr_estimator, X=self.X, y=self.y
        )

        ale = explainer.ale(features=self.X.columns, n_bins=10)
        ale_var_results = explainer.ale_variance(
            ale, estimator_names=self.lr_estimator_name
        )


        # TODO: coefficients
        shap_results = explainer.local_attributions('shap',
            shap_kws={
                "masker": shap.maskers.Partition(
                    self.X, max_samples=100, clustering="correlation"
                ),
                "algorithm": "auto",
            }
        )
        

        # Implicit test of the to_sklearn_importance method.
        shap_imp = to_skexplain_importance(
            shap_results[f"shap_values__{self.lr_estimator_name}"].values,
            estimator_name=self.lr_estimator_name,
            feature_names=list(self.X.columns),
            method="shap_sum",
        )


        # Check the single-pass and multi-pass permutation importance (both forward and backward)
        for direction in ["backward", "forward"]:
            results = explainer.permutation_importance(
                n_vars=len(self.X.columns),
                evaluation_fn="mse",
                n_permute=50,
                direction=direction,
                n_jobs=2, 
            )

            np.testing.assert_array_equal(
                results[f"{direction}_multipass_rankings__{self.lr_estimator_name}"].values,
                TRUE_RANKINGS,
            )
            np.testing.assert_array_equal(
                results[f"{direction}_singlepass_rankings__{self.lr_estimator_name}"].values,
                TRUE_RANKINGS,
            )

        # Check the ALE variance.
        np.testing.assert_array_equal(
            ale_var_results[f"ale_variance_rankings__{self.lr_estimator_name}"].values,
            TRUE_RANKINGS,
        )

        # Check the SHAP.
        np.testing.assert_array_equal(
            shap_imp[f"shap_sum_rankings__{self.lr_estimator_name}"].values,
            TRUE_RANKINGS,
        )

    def test_ale_variance(self):
        explainer = skexplain.ExplainToolkit(
            estimators=self.lr_estimator, X=self.X, y=self.y
        )

        ale = explainer.ale(features=self.X.columns, n_bins=10)
        ale_var_results = explainer.ale_variance(ale)

        with self.assertRaises(Exception) as ex_1:
            ale_var_results = explainer.ale_variance(ale=np.array([0, 0]))

        except_msg_1 = """
                                 ale must be an xarray.Dataset, 
                                 perferably generated by ExplainToolkit.ale 
                                 to be formatted correctly
                                 """
        self.assertMultiLineEqual(ex_1.exception.args[0], except_msg_1)
        

        with self.assertRaises(Exception) as ex_2:
            ale_var_results = explainer.ale_variance(
                ale, estimator_names=[self.lr_estimator_name, "Fake"]
            )

        except_msg_2 = "ale does not contain values for all the estimator names given!"

        self.assertEqual(ex_2.exception.args[0], except_msg_2)

    def test_grouped_importance(self):
        explainer = skexplain.ExplainToolkit(
            estimators=self.lr_estimator, X=self.X, y=self.y
        )
        
        groups = {'group1' : ['X_1', 'X_2'],
          'group2' : ['X_2', 'X_3'],
          'group3' : ['X_4', 'X_5'],
          }
        
        correct_rankings = np.array(['group1','group2', 'group3'])
        
        # Catch using a wrong perm_method! 
        with self.assertRaises(Exception) as ex:
            results = explainer.grouped_permutation_importance(perm_method='grou', 
                                       evaluation_fn='mse', 
                                       n_permute=10, 
                                       groups=groups,
                                       sample_size=100, 
                                       subsample=1.0, 
                                       n_jobs=1, 
                                       clustering_kwargs={'n_clusters' : 10})
            
        except_msg = "Invalid perm_method! Available options are 'grouped' and 'grouped_only'"
        self.assertEqual(ex.exception.args[0], except_msg)
            
        # Simple test that the rankings are correct. 
        for perm_method in ['grouped', 'grouped_only']:
            results = explainer.grouped_permutation_importance(perm_method=perm_method, 
                                       evaluation_fn='mse', 
                                       n_permute=10, 
                                       groups=groups,
                                       sample_size=100, 
                                       subsample=1.0, 
                                       n_jobs=1, 
                                       clustering_kwargs={'n_clusters' : 10})
            
            np.testing.assert_array_equal(results[f'{perm_method}_rankings__{self.lr_estimator_name}'].values, 
                                          correct_rankings)
            
            
            
    def test_to_skexplain_importance(self):
        # Coefficients, Gini to Importance (SHAP is tested above).
        scores = [self.lr.coef_, self.rf.feature_importances_]
        methods = ['coefs', 'gini']
        
        for imp, method in zip(scores, methods):
            for normalize in [True, False]:
                results = to_skexplain_importance(importances=imp, 
                                estimator_name=self.lr_estimator_name, 
                                feature_names=list(self.X.columns), 
                                method=method, 
                                normalize=normalize)
        
        
                # Check that the ranking is correct.
                np.testing.assert_array_equal(
                    results[f"{method}_rankings__{self.lr_estimator_name}"],
                    TRUE_RANKINGS,
                )
        

if __name__ == "__main__":
    unittest.main()

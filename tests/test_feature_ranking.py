# Unit test for the accumulated local effect code in MintPy
import unittest
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

import sys, os 
current_dir = os.getcwd()
path = os.path.dirname(current_dir)
sys.path.append(path)

import pymint

class TestInterpretToolkit(unittest.TestCase):
    def setUp(self):
        model_objs, model_names = pymint.load_models()
        examples, targets = pymint.load_data()
        examples = examples.astype({'urban': 'category', 'rural':'category'})
        
        self.examples = examples
        self.targets = targets
        self.models = model_objs
        self.model_names = model_names
        
        random_state=np.random.RandomState(42)
        
        # Fit a simple 5-variable linear regression model. 
        n_examples = 2000
        n_vars = 5 
        weights = [2.0, 1.5, 1.2, 0.7, 0.2]
        X = np.stack([random_state.uniform(-1,1, size=n_examples) for _ in range(n_vars)], axis=-1)
        feature_names = [f'X_{i+1}' for i in range(n_vars)]
        X = pd.DataFrame(X, columns=feature_names)
        y = X.dot(weights)
        
        lr = LinearRegression()
        lr.fit(X,y)
        
        self.X=X
        self.y=y
        self.lr = lr 
        self.lr_model_name = 'Linear Regression'
        self.weights=weights
        
class TestRankings(TestInterpretToolkit):
    def test_bad_evaluation_fn(self):
        # Make sure the metrics are correct
        myInterpreter = pymint.InterpretToolkit(
                models=self.lr,
                model_names=self.lr_model_name,
                examples=self.X,
                targets=self.y
            )
        available_scores = ["auc", "auprc", "bss", "mse", "norm_aupdc"]
        with self.assertRaises(Exception) as ex:
            myInterpreter.calc_permutation_importance(n_vars=len(self.X.columns), 
                                                  evaluation_fn='bad')
        
        
        except_msg = f"evaluation_fn is not set! Available options are {available_scores}"
        self.assertEqual(ex.exception.args[0], except_msg)
     
    def test_custom_evaluation_fn(self):
        # scoring_strategy exception for custom evaluation funcs 
        myInterpreter = pymint.InterpretToolkit(
                models=self.lr,
                model_names=self.lr_model_name,
                examples=self.X,
                targets=self.y
            )
        available_scores = ["auc", "auprc", "bss", "mse", "norm_aupdc"]
        with self.assertRaises(Exception) as ex:
            myInterpreter.calc_permutation_importance(n_vars=len(self.X.columns), 
                                                  evaluation_fn=roc_auc_score,)
        
        except_msg = """ 
                The scoring_strategy argument is None! If you are using a user-define evaluation_fn 
                then scoring_strategy must be set! If a metric is positively-oriented (a higher value is better), 
                then set scoring_strategy = "argmin_of_mean" and if is negatively-oriented-
                (a lower value is better), then set scoring_strategy = "argmax_of_mean"
                """
        self.assertEqual(ex.exception.args[0], except_msg)
        
    def test_shape(self):
        # Shape is correct (with bootstrapping) 
        myInterpreter = pymint.InterpretToolkit(
                models=self.lr,
                model_names=self.lr_model_name,
                examples=self.X,
                targets=self.y
            )
        n_vars=3
        n_bootstrap=8
        results = myInterpreter.calc_permutation_importance(n_vars=n_vars, 
                                                  evaluation_fn='mse',
                                                 n_bootstrap=n_bootstrap)
        # shape should be (n_vars_multipass, n_bootstrap)
        self.assertEqual( results[f'multipass_scores__{self.lr_model_name}'].values.shape,
                          (n_vars, n_bootstrap) 
                        )
        # shape should be (n_vars_singlepass, n_bootstrap)
        self.assertEqual( results[f'singlepass_scores__{self.lr_model_name}'].values.shape,
                          (len(self.X.columns), n_bootstrap) 
                        )
        
    def test_correct_rankings(self):
        # rankings are correct for simple case (for multi-pass, single-pass, and ale_variance)
        myInterpreter = pymint.InterpretToolkit(
                models=self.lr,
                model_names=self.lr_model_name,
                examples=self.X,
                targets=self.y
            )
        results = myInterpreter.calc_permutation_importance(n_vars=len(self.X.columns), 
                                                  evaluation_fn='mse',
                                                 n_bootstrap=10)
        
        myInterpreter.calc_ale(features=self.X.columns, n_bins=40)
        ale_var_results = myInterpreter.calc_ale_variance(model_name=self.lr_model_name)

        true_rankings = np.array( ['X_1', 'X_2', 'X_3', 'X_4', 'X_5'])

        np.testing.assert_array_equal( results[f'multipass_rankings__{self.lr_model_name}'].values,
                          true_rankings
                        )
        np.testing.assert_array_equal( results[f'singlepass_rankings__{self.lr_model_name}'].values,
                          true_rankings
                        )
        np.testing.assert_array_equal( ale_var_results[f'ale_variance_rankings__{self.lr_model_name}'].values,
                         true_rankings
                        )

    def test_ale_variance(self):
        myInterpreter = pymint.InterpretToolkit(
                models=self.lr,
                model_names=self.lr_model_name,
                examples=self.X,
                targets=self.y
            )
        with self.assertRaises(Exception) as ex_1:
            ale_var_results = myInterpreter.calc_ale_variance(ale_data=np.array([]))
            
        except_msg_1 = """
                                 ale_data must be an xarray.Dataset, 
                                 perferably generated by mintpy.InterpretToolkit.calc_ale to be formatted correctly
                                 """
        self.assertEqual(ex_1.exception.args[0], except_msg_1)
        
        with self.assertRaises(Exception) as ex_2:
            ale_var_results = myInterpreter.calc_ale_variance()
        
        except_msg_2 = 'Must provide ale_data or compute ale for each feature using mintpy.InterpretToolkit.calc_ale'
        
        self.assertEqual(ex_2.exception.args[0], except_msg_2)

if __name__ == "__main__":
    unittest.main()
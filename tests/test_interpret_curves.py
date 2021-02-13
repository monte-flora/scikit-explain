# Unit test for the accumulated local effect code in MintPy
import unittest
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

import sys, os 
current_dir = os.getcwd()
path = os.path.dirname(current_dir)
sys.path.append(path)

import mintpy

class TestInterpretToolkit(unittest.TestCase):
    def setUp(self):
        model_objs, model_names = mintpy.load_models()
        examples, targets = mintpy.load_data()
        examples = examples.astype({'urban': 'category', 'rural':'category'})
        
        self.examples = examples
        self.targets = targets
        self.models = model_objs
        self.model_names = model_names
        
        random_state=np.random.RandomState(42)
        
        # Fit a simple 5-variable linear regression model. 
        n_examples = 1000
        n_vars = 5 
        weights = [2.0, 1.5, 1.2, 0.5, 0.2]
        X = np.stack([random_state.uniform(0,1, size=n_examples) for _ in range(n_vars)], axis=-1)
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
        
class TestInterpretCurves(TestInterpretToolkit):
    def test_bad_feature_names_exception(self):
        feature='bad_feature'
        myInterpreter = mintpy.InterpretToolkit(
                models=self.models[0],
                model_names=self.model_names[0],
                examples=self.examples,
                targets=self.targets
            )
        with self.assertRaises(KeyError) as ex:
            myInterpreter.calc_ale(features=feature)

        except_msg = f"'{feature}' is not a valid feature."
        self.assertEqual(ex.exception.args[0], except_msg)
        
    
    def test_too_many_bins(self):
        myInterpreter = mintpy.InterpretToolkit(
                models=self.models[0],
                model_names=self.model_names[0],
                examples=self.examples,
                targets=self.targets
            )
        n_bins=100
        with self.assertRaises(ValueError) as ex:
            myInterpreter.calc_ale(features=['temp2m'],
                                             subsample=100,
                                             n_bins=n_bins,
                                            )
        except_msg = f"""
                                 The value of n_bins ({n_bins}) is likely too 
                                 high relative to the sample size of the data. Either increase
                                 the data size (if using subsample) or use less bins. 
                                 """
        self.assertEqual(ex.exception.args[0], except_msg)
       
    def test_results_shape(self):
        # Bootstrap has correct shape
        feature='X_1'
        myInterpreter = mintpy.InterpretToolkit(
                models=self.lr,
                model_names=self.lr_model_name,
                examples=self.X,
                targets=self.y
            )
        results = myInterpreter.calc_ale(features=feature, n_bins=10, n_bootstrap=5)
        ydata = results[f'{feature}__{self.lr_model_name}__ale'].values
    
        self.assertEqual(ydata.shape, (5,10))
        
    def test_xdata(self):
        # Bin values are correct. 
        myInterpreter = mintpy.InterpretToolkit(
                models=self.lr,
                model_names=self.lr_model_name,
                examples=self.X,
                targets=self.y
            ) 
        feature='X_1'
        results = myInterpreter.calc_ale(features=feature, n_bins=5)
        xdata = results[f'{feature}__bin_values'].values

        self.assertCountEqual(np.round(xdata,8), 
                              [0.09082125, 
                               0.27714732, 
                               0.48378955, 
                               0.6950751 , 
                               0.89978646] )
    
    def test_ale_simple(self):
        # ALE is correct for a simple case 
        # The coefficient of the ALE curves must 
        # match that of the actual coefficient. 
        myInterpreter = mintpy.InterpretToolkit(
                models=self.lr,
                model_names=self.lr_model_name,
                examples=self.X,
                targets=self.y
            ) 
        feature='X_1'
        results = myInterpreter.calc_ale(features=feature, n_bins=5)
        lr = LinearRegression()
        lr.fit(results[f'{feature}__bin_values'].values.reshape(-1, 1), 
               results[f'{feature}__{self.lr_model_name}__ale'].values[0,:])

        self.assertAlmostEqual(lr.coef_[0], self.weights[0]) 

    def test_pd_simple(self):
        # PD is correct for a simple case 
        # The coefficient of the PD curves must 
        # match that of the actual coefficient. 
        myInterpreter = mintpy.InterpretToolkit(
                models=self.lr,
                model_names=self.lr_model_name,
                examples=self.X,
                targets=self.y
            ) 
        feature='X_1'
        results = myInterpreter.calc_pd(features=feature, n_bins=5)
        lr = LinearRegression()
        lr.fit(results[f'{feature}__bin_values'].values.reshape(-1, 1), 
               results[f'{feature}__{self.lr_model_name}__pd'].values[0,:])

        self.assertAlmostEqual(lr.coef_[0], self.weights[0])     
    
    def test_cat_ale_simple(self):
        # Make sure the categorical ALE is working correctly
        pass
    
    def test_ale_complex(self):
        # ALE is correct for a more complex case! 
        pass
    
    def test_2d_ale(self):
        pass
    
    def test_2d_pd(self):
        pass
    
    def test_ias(self):
        pass
    
    def test_hstat(self):
        pass 
    
if __name__ == "__main__":
    unittest.main()

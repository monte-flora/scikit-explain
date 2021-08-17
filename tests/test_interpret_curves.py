# Unit test for the accumulated local effect code in MintPy
import unittest
import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pymint

class TestInterpretToolkit(unittest.TestCase):
    def setUp(self):
        self.estimators = pymint.load_models()
        X_clf, y_clf = pymint.load_data()
        X_clf = X_clf.astype({'urban': 'category', 'rural':'category'})
        
        self.X_clf = X_clf
        self.y_clf = y_clf
        
        random_state=np.random.RandomState(42)
        
        # Fit a simple 5-variable linear regression estimator. 
        n_X = 1000
        n_vars = 5 
        weights = [2.0, 1.5, 1.2, 0.5, 0.2]
        X = np.stack([random_state.uniform(0,1, size=n_X) for _ in range(n_vars)], axis=-1)
        feature_names = [f'X_{i+1}' for i in range(n_vars)]
        X = pd.DataFrame(X, columns=feature_names)
        y = X.dot(weights)
        
        lr = LinearRegression()
        lr.fit(X,y)
        
        self.X=X
        self.y=y
        self.lr = lr 
        self.lr_estimator_name = 'Linear Regression'
        self.weights=weights
        
class TestInterpretCurves(TestInterpretToolkit):
    def test_bad_feature_names_exception(self):
        feature='bad_feature'
        explainer = pymint.InterpretToolkit(
                estimators=self.estimators[0],
                X=self.X_clf,
                y=self.y_clf
            )
        with self.assertRaises(KeyError) as ex:
            explainer.ale(features=feature)

        except_msg = f"'{feature}' is not a valid feature."
        self.assertEqual(ex.exception.args[0], except_msg)
        
    
    def test_too_many_bins(self):
        explainer = pymint.InterpretToolkit(
                estimators=self.estimators[0],
                X=self.X_clf,
                y=self.y_clf
            )
        
        n_bins=100
        with self.assertRaises(ValueError) as ex:
            explainer.ale(features=['temp2m'],
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
        explainer = pymint.InterpretToolkit(
                estimators=(self.lr_estimator_name, self.lr),
                X=self.X,
                y=self.y
            )
        results = explainer.ale(features=feature, n_bins=10, n_bootstrap=5)
        ydata = results[f'{feature}__{self.lr_estimator_name}__ale'].values
    
        self.assertEqual(ydata.shape, (5,10))
        
    def test_xdata(self):
        # Bin values are correct. 
        explainer = pymint.InterpretToolkit(
                estimators=(self.lr_estimator_name, self.lr),
                X=self.X,
                y=self.y
            ) 
        feature='X_1'
        results = explainer.ale(features=feature, n_bins=5)
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
        explainer = pymint.InterpretToolkit(
                estimators=(self.lr_estimator_name, self.lr),
                X=self.X,
                y=self.y
            ) 
        feature='X_1'
        results = explainer.ale(features=feature, n_bins=5)
        lr = LinearRegression()
        lr.fit(results[f'{feature}__bin_values'].values.reshape(-1, 1), 
               results[f'{feature}__{self.lr_estimator_name}__ale'].values[0,:])

        self.assertAlmostEqual(lr.coef_[0], self.weights[0]) 

    def test_pd_simple(self):
        # PD is correct for a simple case 
        # The coefficient of the PD curves must 
        # match that of the actual coefficient. 
        explainer = pymint.InterpretToolkit(
                estimators=(self.lr_estimator_name, self.lr),
                X=self.X,
                y=self.y
            ) 
        feature='X_1'
        results = explainer.pd(features=feature, n_bins=5)
        lr = LinearRegression()
        lr.fit(results[f'{feature}__bin_values'].values.reshape(-1, 1), 
               results[f'{feature}__{self.lr_estimator_name}__pd'].values[0,:])

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

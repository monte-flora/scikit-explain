#===================================================
# Fit a simple 5-variable linear regression model
# for unit testing. 
#===================================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import shap
import unittest
import numpy as np
import pandas as pd
import os, sys

# Adding the parent directory to path so that 
# skexplain can be imported without being explicitly 
sys.path.append(os.path.dirname(os.getcwd()))

import skexplain

N = 5000
NVARS = 5
INTS = [42, 123, 0, 69, 420]
WEIGHTS = [3.0, 2.5, 2.0, 1.0, 0.5]

# Creating repeatability random states for unit testing.
random_states = [np.random.RandomState(i) for i in INTS]
        
# Fit a simple 5-variable linear regression estimator.
X = np.stack(
            [random_states[i].normal(0, 1, size=N) for i in range(NVARS)], axis=-1
        )
feature_names = [f"X_{i+1}" for i in range(NVARS)]
X = pd.DataFrame(X, columns=feature_names)
y = X.dot(WEIGHTS)


class TestSciKitExplainData(unittest.TestCase):
    def setUp(self):
        # Load the bulit-in models and data in scikit-explain.
        estimators = skexplain.load_models()
        X, y = skexplain.load_data()
        X = X.astype({"urban": "category", "rural": "category"})

        self.X = X
        self.y = y
        self.estimators = estimators

class TestLR(unittest.TestCase):
    """ Set-up a linear regression model """
    def setUp(self):
        lr = LinearRegression()
        lr.fit(X.values, y.values)

        self.X = X
        self.y = y
        self.lr = lr
        self.lr_estimator_name = "Linear Regression"
        self.lr_estimator=(self.lr_estimator_name, self.lr)
        self.WEIGHTS = WEIGHTS
        
        #
        # y = x_1 + x_2 + x_1*x_2
        X_interact = X.iloc[:,:2]
        y_interact = X_interact.iloc[:,0] + X_interact.iloc[:,1] + 2.0*(X_interact.iloc[:,0] * X_interact.iloc[:,1])
        feature_names = [f"X_{i+1}" for i in range(X_interact.shape[1])]

        rf_interact = RandomForestRegressor()
        rf_interact.fit(X_interact, y_interact)
        estimator_name = 'RF'
    
        self.explainer_interact = skexplain.ExplainToolkit(
            estimators=(estimator_name, rf_interact), X=X_interact, y=y_interact, feature_names=feature_names
        )
        
class TestRF:
    """ Set-up random forest model """
    rf = RandomForestRegressor()
    rf.fit(X.values, y.values)
    rf_estimator = ("Random Forest", rf) 
    
class TestSingleExampleContributions(unittest.TestCase, TestRF):
    """ Set-up for a single example for feature contributions """
    def setUp(self):
        # Computing SHAP values in scikit-explain.
        self.X=X
        self.y=y
        X_sub = X.iloc[[100]]
        X_sub.reset_index(inplace=True, drop=True)
        self.X_sub = X_sub

        self.explainer = skexplain.ExplainToolkit(
            estimators=self.rf_estimator,
            X=X_sub,
        )

class TestMultiExampleContributions(unittest.TestCase, TestRF):
    """ Set-up for a single example for feature contributions """
    def setUp(self):
        # Computing SHAP values in scikit-explain.
        self.X=X
        self.y=y
        
        rf = RandomForestRegressor()
        rf.fit(X.values, y.values)
        self.rf_estimator = ("Random Forest", rf) 
        self.rf_estimator_name = "Random Forest"
        
        # Computing SHAP values in scikit-explain.
        inds = np.random.choice(len(self.X), size=5, replace=False)
        X_sub = self.X.iloc[inds]
        X_sub.reset_index(inplace=True, drop=True)

        self.X_sub = X_sub
        self.y_sub = self.y[inds]
        
        self.explainer = skexplain.ExplainToolkit(
            estimators=self.rf_estimator,
            X=X_sub, y=self.y_sub
        )
        
        
        self.results = self.explainer.shap(
            shap_kwargs={
                "masker": shap.maskers.Partition(
                    self.X, max_samples=100, clustering="correlation"
                ),
                "algorithm": "auto",
            }
        )

        
        
        
        
        
    
        
        
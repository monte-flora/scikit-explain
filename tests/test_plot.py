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
        
class Test1DPlotting(TestInterpretToolkit):

    def test_1d_plot(self):
        # Make sure the plot data is correct. 
        feature='X_1'
        myInterpreter = pymint.InterpretToolkit(
                models=self.lr,
                model_names=self.lr_model_name,
                examples=self.X,
                targets=self.y
            )
        results = myInterpreter.calc_ale(features=feature, n_bins=30, n_bootstrap=1)
        ydata = results[f'{feature}__{self.lr_model_name}__ale'].values[0,:]
        xdata = results[f'{feature}__bin_values'].values
        
        fig, ax = myInterpreter.plot_ale(features=feature)
        
        ## effect line
        eff_plt_data = ax.lines[0].get_xydata()
        # the x values should be the bins
        np.testing.assert_array_equal(eff_plt_data[:, 0], xdata)
        # the y values should be the effect
        np.testing.assert_array_equal(eff_plt_data[:, 1], ydata)

class Test2DPlotting(TestInterpretToolkit):
    def test_2d_plot(self):
        pass
        
class TestRankPlots(TestInterpretToolkit):
    
    # Make sure the ranking is plotted correctly
    def test_rank_plot(self):
        pass
    
class TestContributionPlots(TestInterpretToolkit):
    
    # Make sure the ranking is plotted correctly
    def test_contrib_plot(self):
        pass
    

if __name__ == "__main__":
    unittest.main()
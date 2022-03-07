# Unit test for the ExplainToolkit code in MintPy
import unittest
from sklearn.ensemble import RandomForestRegressor
import os,sys

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
        
class TestInitializeExplainToolkit(TestExplainToolkit):
    """Test for proper initialization of ExplainToolkit"""
    def test_estimator_has_been_fit(self):
        # estimators must be fit! 
        with self.assertRaises(Exception) as ex:
            skexplain.ExplainToolkit(
                estimators=('Random Forest', RandomForestRegressor()),
                X=self.X,
                y=self.y
            )
        except_msg = "One or more of the estimators given has NOT been fit!"
        self.assertEqual(ex.exception.args[0], except_msg)
        
    def test_X_and_feature_names(self):
        # Feature names must be provided if X is an numpy.array. 
        with self.assertRaises(Exception) as ex:
            skexplain.ExplainToolkit(
                estimators=self.estimators[0],
                X=self.X.values,
                y=self.y,
                feature_names=None, 
            )
        except_msg = "Feature names must be specified if using NumPy array."
        self.assertEqual(ex.exception.args[0], except_msg)
        
    def test_estimator_output(self):
        estimator_output='regression'
        available_options = ["raw", "probability"]
        with self.assertRaises(Exception) as ex:
            skexplain.ExplainToolkit(
                estimators=self.estimators[0],
                X=self.X,
                y=self.y,
                estimator_output=estimator_output,

            )
        except_msg = f"""
                                {estimator_output} is not an accepted options. 
                                 The available options are {available_options}.
                                 Check for syntax errors!
                                 """
        self.assertEqual(ex.exception.args[0], except_msg)
        
if __name__ == "__main__":
    unittest.main()
       

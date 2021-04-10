# Unit test for the InterpretToolkit code in MintPy
import unittest
from sklearn.ensemble import RandomForestRegressor



import sys, os 
current_dir = os.getcwd()
path = os.path.dirname(current_dir)
sys.path.append(path)
import pymint

class TestInterpretToolkit(unittest.TestCase):
    def setUp(self):
        estimator_objs, estimator_names = pymint.load_models()
        X, y = pymint.load_data()
        X = X.astype({'urban': 'category', 'rural':'category'})
        
        self.X = X
        self.y = y
        self.estimators = estimator_objs
        self.estimator_names = estimator_names
        
class TestInitializeInterpretToolkit(TestInterpretToolkit):
    """Test for proper initialization of InterpretToolkit"""
    def test_estimator_has_been_fit(self):
        # estimators must be fit! 
        with self.assertRaises(Exception) as ex:
            pymint.InterpretToolkit(
                estimators=RandomForestRegressor(),
                estimator_names='Random Forest',
                X=self.X,
                y=self.y
            )
        except_msg = "One or more of the estimators given has NOT been fit!"
        print(ex.exception.args[0], except_msg)
        self.assertEqual(ex.exception.args[0], except_msg)
        
    def test_estimator_and_estimator_names(self):
        # List of estimator names != list of estimator_objs 
        with self.assertRaises(Exception) as ex:
            pymint.InterpretToolkit(
                estimators=self.estimators[0],
                estimator_names=self.estimator_names[:2],
                X=self.X,
                y=self.y
            )
        except_msg = "Number of estimator objects is not equal to the number of estimator names given!"
        self.assertEqual(ex.exception.args[0], except_msg)
        
    def test_X_and_feature_names(self):
        # Feature names must be provided if X is an numpy.array. 
        with self.assertRaises(Exception) as ex:
            pymint.InterpretToolkit(
                estimators=self.estimators[0],
                estimator_names=self.estimator_names[0],
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
            pymint.InterpretToolkit(
                estimators=self.estimators[0],
                estimator_names=self.estimator_names[0],
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
       

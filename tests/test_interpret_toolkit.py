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
        model_objs, model_names = pymint.load_models()
        examples, targets = pymint.load_data()
        examples = examples.astype({'urban': 'category', 'rural':'category'})
        
        self.examples = examples
        self.targets = targets
        self.models = model_objs
        self.model_names = model_names
        
class TestInitializeInterpretToolkit(TestInterpretToolkit):
    """Test for proper initialization of InterpretToolkit"""
    def test_model_has_been_fit(self):
        # Models must be fit! 
        with self.assertRaises(Exception) as ex:
            pymint.InterpretToolkit(
                models=RandomForestRegressor(),
                model_names='Random Forest',
                examples=self.examples,
                targets=self.targets
            )
        except_msg = "One or more of the models given has NOT been fit!"
        print(ex.exception.args[0], except_msg)
        self.assertEqual(ex.exception.args[0], except_msg)
        
    def test_model_and_model_names(self):
        # List of model names != list of model_objs 
        with self.assertRaises(Exception) as ex:
            pymint.InterpretToolkit(
                models=self.models[0],
                model_names=self.model_names[:2],
                examples=self.examples,
                targets=self.targets
            )
        except_msg = "Number of model objects is not equal to the number of model names given!"
        self.assertEqual(ex.exception.args[0], except_msg)
        
    def test_examples_and_feature_names(self):
        # Feature names must be provided if examples is an numpy.array. 
        with self.assertRaises(Exception) as ex:
            pymint.InterpretToolkit(
                models=self.models[0],
                model_names=self.model_names[0],
                examples=self.examples.values,
                targets=self.targets,
                feature_names=None, 
            )
        except_msg = "Feature names must be specified if using NumPy array."
        self.assertEqual(ex.exception.args[0], except_msg)
        
    def test_model_output(self):
        model_output='regression'
        available_options = ["raw", "probability"]
        with self.assertRaises(Exception) as ex:
            pymint.InterpretToolkit(
                models=self.models[0],
                model_names=self.model_names[0],
                examples=self.examples,
                targets=self.targets,
                model_output=model_output,

            )
        except_msg = f"""
                                {model_output} is not an accepted options. 
                                 The available options are {available_options}.
                                 Check for syntax errors!
                                 """
        self.assertEqual(ex.exception.args[0], except_msg)
        
if __name__ == "__main__":
    unittest.main()
       

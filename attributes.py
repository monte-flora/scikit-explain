import numpy as np 
import pandas as pd
from collections import OrderedDict
from .utils import is_list, to_list

class Attributes:
    """
    Attributes class handles setting and check the attributes
    from InterpretToolkit, PartialDependence, AccumulatedLocalEffects
    """
    def set_model_attribute(self, model_objs, model_names):
        """
        Checks the type of the model attribute. 
        If a list or not a dict, then the model argument
        is converted to a dict for processing. 
        
        Args:
        ----------
            model_objs : object, list
                pre-fitted scikit-learn model object or list of objects
            model_names : string, list 
                List of names of the models in model_objs 
                (for plotting purposes) 
        """
        if not is_list(model_objs):
            model_objs = to_list(model_objs)
        
        if not is_list(model_names):
            model_names = to_list(model_names)
               
        assert len(model_objs) == len(model_names), "Number of model objects is not equal to the number of model names given!"    
       
        self.models = OrderedDict([(name, obj) for name, obj in zip(model_names, model_objs)])
        self.model_names = model_names
 
    def set_target_attribute(self, targets):
        """
        Checks the type of the targets attribute. 
        """
         # check that targets are assigned correctly
        if is_list(targets):
            self.targets = np.array(targets)
        elif isinstance(targets, np.ndarray):
            self.targets = targets
        elif isinstance(targets, (pd.DataFrame, pd.Series)):
            self.targets = targets.values
        else:
            if targets is not None:
                raise TypeError('Target variable must be numpy array or pandas.DataFrame.')
            
    def set_examples_attribute(self, examples, feature_names=None):
        """
        Check the type of the examples attribute.
        """
        # make sure data is the form of a pandas dataframe regardless of input type
        if isinstance(examples, np.ndarray):
            if (feature_names is None):
                raise Exception('Feature names must be specified if using NumPy array.')
            else:
                self.examples = pd.DataFrame(data=examples, columns=feature_names)
        else:
            self.examples = examples
        
        self.feature_names  = self.examples.columns.to_list()
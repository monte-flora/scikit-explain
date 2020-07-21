import numpy as np 
import pandas as pd

class Attributes:
    """
    Attributes class handles setting and check the attributes
    from InterpretToolkit, PartialDependence, AccumulatedLocalEffects
    """
    def set_model_attribute(self, model):
        """
        Checks the type of the model attribute. 
        If a list or not a dict, then the model argument
        is converted to a dict for processing. 
        
        Args:
        ----------
            model : object, list, or dict 
        """
         # if model is of type list or single objection, convert to dictionary
        if not isinstance(model, dict):
            if isinstance(model, list):
                self.models = {type(m).__name__ : m for m in model}
            else:
                self.models = {type(model).__name__ : model}
        # user provided a dict
        else:
            self.models = model
    
    def set_target_attribute(self, targets):
        """
        Checks the type of the targets attribute. 
        """
         # check that targets are assigned correctly
        if isinstance(targets, list):
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
                self.examples      = pd.DataFrame(data=examples, columns=feature_names)
        else:
            self.examples = examples
        
        self.feature_names  = self.examples.columns.to_list()
import numpy as np
import pandas as pd
from collections import OrderedDict
from .utils import is_list, to_list


class Attributes:
    """
    The Attributes class handles checking and setting the attributes
    for the InterpretToolkit, GlobalInterpret, and LocalInterpret classes.

    Attributes is a base class to be inherited by those classes and
    should never be instantiated
    """

    def set_model_attribute(self, model_objs, model_names):
        """
        Checks the type of the models and model_names attributes.
        If a list or not a dict, then the model argument
        is converted to a dict for processing.

        Args:
        ----------
            model_objs : object or list of objects
                pre-fitted scikit-learn model object or list thereof.
            model_names : string or list of strings
                List of names of the models in model_objs
                (for plotting purposes)
        """
        model_is_none = model_objs == None

        # Convert the model_objs to a list, if it is not already.
        if not is_list(model_objs):
            model_objs = to_list(model_objs)

        # Convert the name of the model_objs to a list,
        # if is not already.
        if not is_list(model_names):
            model_names = to_list(model_names)

        # Check that the model_objs and model_names are the same size.
        if not model_is_none:
            assert len(model_objs) == len(
                model_names
            ), "Number of model objects is not equal to the number of model names given!"

        # Create a dictionary from the model_objs and model_names.
        # Then set the attributes.
        self.models = OrderedDict(
            [(name, obj) for name, obj in zip(model_names, model_objs)]
        )
        self.model_names = model_names

    def set_target_attribute(self, targets):
        """
        Checks the type of the targets attribute.
        """
        # check that targets are assigned correctly
        if type(targets) == type(None):
            raise ValueError("targets are required!")
        if is_list(targets):
            self.targets = np.array(targets)
        elif isinstance(targets, np.ndarray):
            self.targets = targets
        elif isinstance(targets, (pd.DataFrame, pd.Series)):
            self.targets = targets.values
        else:
            if targets is not None:
                raise TypeError(
                    "Target variable must be numpy array or pandas.DataFrame."
                )
            else:
                self.targets = None

    def set_examples_attribute(self, examples, feature_names=None):
        """
        Check the type of the examples attribute.
        """
        # make sure data is the form of a pandas dataframe regardless of input type
        if type(examples) == type(None):
            raise ValueError("examples are required!")
        if isinstance(examples, np.ndarray):
            if feature_names is None:
                raise Exception("Feature names must be specified if using NumPy array.")
            else:
                self.examples = pd.DataFrame(data=examples, columns=feature_names)
        else:
            self.examples = examples

        if examples is not None:
            self.feature_names = self.examples.columns.to_list()

    def set_model_output(self, model_output, model):
        """
        Check the model output is given and if not try to
        assume the correct model output.
        """

        model_obj = model[0] if is_list(model) else model
        available_options = ["raw", "probability"]

        if model_output is None:
            if hasattr(model_obj, "predict_proba"):
                self.model_output = "probability"
            else:
                self.model_output = "raw"

        else:
            if model_output in available_options:
                self.model_output = model_output
            else:
                raise ValueError(
                    f"""
                                {model_output} is not an accepted options. 
                                 The available options are {available_options}.
                                 Check for syntax errors!
                                 """
                )

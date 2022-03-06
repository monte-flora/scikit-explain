import numpy as np
import pandas as pd
from collections import OrderedDict
from .utils import is_list, to_list, is_fitted

class Attributes:
    """
    The Attributes class handles checking and setting the attributes
    for the InterpretToolkit, GlobalInterpret, and LocalInterpret classes.

    Attributes is a base class to be inherited by those classes and
    should never be instantiated
    """

    def set_estimator_attribute(self, estimator_objs, estimator_names):
        """
        Checks the type of the estimators and estimator_names attributes.
        If a list or not a dict, then the estimator argument
        is converted to a dict for processing.

        Parameters
        ----------
        estimators : object, list of objects
            A fitted estimator object or list thereof implementing `predict` or 
            `predict_proba`.
            Multioutput-multiclass classifiers are not supported.
            
        estimator_names : string, list
            Names of the estimators (for internal and plotting purposes)    
        """
        estimator_is_none = estimator_objs == None
        
        # Convert the estimator_objs to a list, if it is not already.
        if not is_list(estimator_objs):
            estimator_objs = to_list(estimator_objs)

        # Convert the name of the estimator_objs to a list,
        # if is not already.
        if not is_list(estimator_names):
            estimator_names = to_list(estimator_names)

        # Check that the estimator_objs and estimator_names are the same size.
        if not estimator_is_none:
            assert len(estimator_objs) == len(
                estimator_names
            ), "Number of estimator objects is not equal to the number of estimator names given!"

        # Check that the estimator objects have been fit! 
        if not estimator_is_none:
            if not all([is_fitted(m) for m in estimator_objs]):
                raise ValueError('One or more of the estimators given has NOT been fit!') 
            
        # Create a dictionary from the estimator_objs and estimator_names.
        # Then set the attributes.
        self.estimators = OrderedDict(
            [(name, obj) for name, obj in zip(estimator_names, estimator_objs)]
        )
        self.estimator_names = estimator_names

    def set_y_attribute(self, y):
        """
        Checks the type of the y attribute.
        """
        # check that y are assigned correctly
        if type(y) == type(None):
            raise ValueError("y is required!")
        if is_list(y):
            self.y = np.array(y)
        elif isinstance(y, np.ndarray):
            self.y = y
        elif isinstance(y, (pd.DataFrame, pd.Series)):
            self.y = y.values
        else:
            if y is not None:
                raise TypeError(
                    "y must be an numpy array or pandas.DataFrame."
                )
            else:
                self.y = None

    def set_X_attribute(self, X, feature_names=None):
        """
        Check the type of the X attribute.
        """
        # make sure data is the form of a pandas dataframe regardless of input type
        if type(X) == type(None):
            raise ValueError("X are required!")
        if isinstance(X, np.ndarray):
            if feature_names is None:
                raise Exception("Feature names must be specified if using NumPy array.")
            else:
                self.X = pd.DataFrame(data=X, columns=feature_names)
        else:
            self.X = X

        if X is not None:
            self.feature_names = self.X.columns.to_list()

    def set_estimator_output(self, estimator_output, estimator):
        """
        Check the estimator output is given and if not try to
        assume the correct estimator output.
        """
        
        estimator_obj = estimator[0] if is_list(estimator) else estimator
        available_options = ["raw", "probability"]

        if estimator_output is None:
            if hasattr(estimator_obj, "predict_proba"):
                self.estimator_output = "probability"
            else:
                self.estimator_output = "raw"

        else:
            if estimator_output in available_options:
                self.estimator_output = estimator_output
            else:
                raise ValueError(
                    f"""
                                {estimator_output} is not an accepted options. 
                                 The available options are {available_options}.
                                 Check for syntax errors!
                                 """
                )

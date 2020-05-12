import numpy as np
import pandas as pd
import joblib

from utils import *

class PartialDependence:

    """
    Class for computing various ML model interpretations...blah blah blah

    Args:
        model : a single (or multiple) scikit-learn models represented as a dictionary.
            Create a dictionary such as { "RandomForest" : rf_sklearn_model }
        examples : pandas DataFrame or ndnumpy array. If ndnumpy array, make sure
            to specify the feature names
        targets: list or numpy array of targets/labels. List converted to numpy array
        classification: defaults to True for classification problems. 
            Set to false otherwise.
        feature_names : defaults to None. Should only be set if examples is a 
            nd.numpy array. Make sure it's a list
    """

    def __init__(self, model=None, examples=None, targets=None, classification=True, 
            feature_names=None):

        # if model is of type list or single objection, convert to dictionary
        if not isinstance(model, dict):
            if isinstance(model, list):
                self._models = {type(m).__name__ : m for m in model}
            else:
                self._models = {type(model).__name__ : model}

        self._examples = examples

        # check that targets are assigned correctly
        if isinstance(targets, list): 
            self._targets = np.array(targets)
        elif isinstance(targets, np.ndarray): 
            self._targets = targets
        else:
            raise TypeError('Target variable must be numpy array.')

        # make sure data is the form of a pandas dataframe regardless of input type
        if isinstance(self._examples, np.ndarray): 
            if (feature_names is None): 
                raise Exception('Feature names must be specified if using NumPy array.')    
            else:
                self._feature_names = feature_names
                self._examples      = pd.DataFrame(data=examples, columns=feature_names)
        else:
            self._feature_names  = examples.columns.to_list()

        self._classification = classification
        self._pdp_values     = None
        self._x1vals         = None
        self._x2vals         = None
        self._hist_vals      = None

        # dictionary containing information for all each feature and model
        self._dict_out = {}


    def get_final_dict(self):  return self._dict_out

#     def get_pd_values(self): return self._pdp_values
#     def get_x1vals(self):    return self._x1vals
#     def get_x2vals(self):    return self._x2vals
#     def get_hist_vals(self): return self._hist_collector

    def run_pd(self, features=None, njobs=1, subsample=1.0, nbootstrap=1, **kwargs):

        """
            Runs the partial dependence calculation and returns a dictionary with all
            necessary inputs for plotting.
        
            feature: List of strings for first-order partial dependence, or list of tuples
                     for second-order
            subsample: a float (between 0-1) for fraction of examples used in bootstrap
            nbootstrap: number of bootstrap iterations to perform. Defaults to 1 (no
                        bootstrapping).
        """

        # get number of features we are processing
        n_feats = len(features)

        # check first element of feature and see if of type tuple; assume second-order calculations
        if isinstance(features[0], tuple): 

            #parallelize routine... calculate partial dependence for each feature set.
            tdict = joblib.Parallel(n_jobs=njobs)(joblib.delayed(self._parallelize_2d) 
                (feature, subsample, nbootstrap) for feature in features)

            #convert list of dicts to dict
            for elem in tdict:
                self._dict_out.update(elem)
                
        # else, single order calculations
        else:
 
            #parallelize routine... calculate partial dependence for each feature. 
            tdict = joblib.Parallel(n_jobs=njobs)(joblib.delayed(self._parallelize_1d) 
                (feature, subsample, nbootstrap) for feature in features)

            #convert list of dicts to dict
            for elem in tdict:
                self._dict_out.update(elem)

    def _parallelize_1d(self, feature, subsample, nbootstrap):

        temp_dict = {}
        temp_dict[feature] = {}

        print(f"Processing feature {feature}...")

        for model_name, model in zip(self._models.keys(), self._models.values()):

            #print(f"Processing model {model}...")

            temp_dict[feature][model_name] = {}

            self.compute_1d_partial_dependence(feature=feature,
                                                model=model,
                                                subsample =subsample,
                                                nbootstrap=nbootstrap)
            
            #print(self._pdp_values)

            # add to a dict 
            temp_dict[feature][model_name]['pd_values'] = self._pdp_values
            temp_dict[feature][model_name]['xdata1']    = self._x1vals
            temp_dict[feature][model_name]['hist_data'] = self._hist_vals

        return temp_dict

    def _parallelize_2d(self, feature, subsample, nbootstrap):

        temp_dict = {}
        temp_dict[feature] = {}

        print(f"Processing feature {feature}...")

        for model_name, model in zip(self._models.keys(), self._models.values()):

            #print(f"Processing model {model}...")

            temp_dict[feature][model_name] = {}

            self.compute_2d_partial_dependence(feature=feature,
                                                model=model,
                                                subsample =subsample,
                                                nbootstrap=nbootstrap)
            
            #print(self._pdp_values)

            # add to a dict 
            temp_dict[feature][model_name]['pd_values'] = self._pdp_values
            temp_dict[feature][model_name]['xdata1']    = self._x1vals
            temp_dict[feature][model_name]['xdata2']    = self._x2vals

        return temp_dict

    def compute_1d_partial_dependence(self, feature=None, **kwargs):

        """
        Calculate the partial dependence.
        # Friedman, J., 2001: Greedy function approximation: a gradient boosting machine.Annals of Statistics,29 (5), 1189–1232.
        ##########################################################################
        Partial dependence plots fix a value for one or more predictors
        # for examples, passing these new data through a trained model, 
        # and then averaging the resulting predictions. After repeating this process
        # for a range of values of X*, regions of non-zero slope indicates that
        # where the ML model is sensitive to X* (McGovern et al. 2019). Only disadvantage is
        # that PDP do not account for non-linear interactions between X and the other predictors.
        #########################################################################

        Args: 
            feature : name of feature to compute PD for (string) 
        """
    
        model      = kwargs.get('model', "")
        subsample  = kwargs.get('subsample', 1.0)
        nbootstrap = kwargs.get('nbootstrap', 1)

        # check to make sure a feature is present...
        if feature is None:
            raise Exception("Specify a feature")

        # check to make sure feature is valid
        if feature not in self._feature_names:
            raise Exception(f"Feature {feature} is not a valid feature")

        # get data in numpy format
        column_of_data = self._examples[feature].to_numpy()

        # append examples for histogram use
        self._hist_vals = column_of_data

        # define bins based on 10th and 90th percentiles
        self._x1vals = np.linspace(
            np.percentile(column_of_data, 5), np.percentile(column_of_data, 95), num=20
        )

        # get the bootstrap samples
        if nbootstrap > 1:
            bootstrap_examples = compute_bootstrap_samples(self._examples, 
                                        subsample=subsample, 
                                        nbootstrap=nbootstrap)
        else:
            bootstrap_examples = [self._examples.index.to_list()]

        # define ALE array
        self._pdp_values = np.full((nbootstrap, self._x1vals.shape[0]), np.nan)

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_examples):

            # get samples
            examples = self._examples.iloc[idx, :].copy()
        
            # for each value, set all indices to the value, make prediction, store mean prediction
            for i, value in enumerate(self._x1vals):

                examples.loc[:, feature] = value

                if self._classification is True:
                    predictions = model.predict_proba(examples)[:, 1] * 100.
                else:
                    predictions = model.predict(examples)

                self._pdp_values[k,i] = np.mean(predictions)


    def compute_2d_partial_dependence(self, feature=None, **kwargs):

        """
        Calculate the partial dependence between two features.

        Args: 
            feature : tuple or list of strings of predictor names

        """

        subsample  = kwargs.get('subsample', 1.0)
        nbootstrap = kwargs.get('nbootstrap', 1)

        # make sure there are two features...
        assert(len(feature) == 2), "Size of features must be equal to 2."

        # check to make sure feature is valid
        if (feature[0] not in self._feature_names): 
            raise TypeError(f'Feature {feature[0]} is not a valid feature')
        if (feature[1] not in self._feature_names): 
            raise TypeError(f'Feature {feature[1]} is not a valid feature')

        if self._x1vals is None:
            # ensures each bin gets the same number of examples
            self._x1vals = np.percentile(self._examples[feature[0]].values, 
                                         np.arange(2.5, 97.5 + 5, 5))

        if self._x2vals is None:
            # ensures each bin gets the same number of examples
            self._x2vals = np.percentile(self._examples[feature[1]].values, 
                                         np.arange(2.5, 97.5 + 5, 5))

        # get the bootstrap samples
        if nbootstrap > 1:
            bootstrap_examples = compute_bootstrap_samples(self._examples, 
                                        subsample=subsample, 
                                        nbootstrap=nbootstrap)
        else:
            bootstrap_examples = [self._examples.index.to_list()]

        # define 2-D grid
        self._pdp_values = np.full((nbootstrap, self._x1vals.shape[0], 
                                                self._x2vals.shape[0]), np.nan)

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_examples):

            # get samples
            examples = self._examples.iloc[idx, :].copy()

            # similar concept as 1-D, but for 2-D
            for i, value1 in enumerate(self._x1vals):
                for j, value2 in enumerate(self._x2vals):

                    examples.loc[feature[0]] = value1
                    examples.loc[feature[1]] = value2

                    if self._classification is True:
                        predictions = self._model.predict_proba(examples)[:, 1] * 100.
                    else:
                        predictions = self._model.predict(examples)

                    self._pdp_values[k,i,j] = np.mean(predictions)

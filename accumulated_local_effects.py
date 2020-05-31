import numpy as np
import pandas as pd
import multiprocessing

from .utils import compute_bootstrap_samples, merge_nested_dict
from .multiprocessing_utils import run_parallel, to_iterator

class AccumulatedLocalEffects:

    def __init__(self, model=None, examples=None, classification=True, 
            feature_names=None):

        # if model is of type list or single objection, convert to dictionary
        if not isinstance(model, dict):
            if isinstance(model, list):
                self._models = {type(m).__name__ : m for m in model}
            else:
                self._models = {type(model).__name__ : model}
        # passed in a dict
        else: 
            self._models = model

        self._examples = examples

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

        # dictionary containing information for all each feature and model
        self._dict_out = {}

    def get_final_dict(self): 

        return self._dict_out

    def run_ale(self, features=None, njobs=1, subsample=1.0, nbootstrap=1, **kwargs):

        """
            Runs the accumulated local effect calculation and returns a dictionary with all
            necessary inputs for plotting.
        
            feature: List of strings for first-order partial dependence, or list of tuples
                     for second-order
            subsample: a float (between 0-1) for fraction of examples used in bootstrap
            nbootstrap: number of bootstrap iterations to perform. Defaults to 1 (no
                        bootstrapping).
        """    
        models = [name for name in list(self._models.keys())]
        
        print(f'Models for ALE: {models}')
        
        # get number of features we are processing
        n_feats = len(features)

        # check first element of feature and see if of type tuple; assume second-order calculations
        if isinstance(features[0], tuple): 
            func =  self.calculate_second_order_ale
        else:
            func = self.calculate_first_order_ale
            
        args_iterator = to_iterator(models, features)
        kwargs = {
                      "subsample": subsample, 
                      "nbootstrap": nbootstrap
                     }
        
        results = run_parallel(
                   func = func,
                   args_iterator = args_iterator,
                   kwargs = kwargs, 
                   nprocs_to_use=njobs
            )
                
        results = merge_nested_dict(results)
            
        self._dict_out = results
            
    def calculate_first_order_ale(self, model_name, feature, **kwargs):

        """
            Computes first-order ALE function on single continuous feature data.

            Parameters
            ----------
            feature : string
                The name of the feature to consider.
            xdata : array
                Quantiles of feature.
        """

        subsample  = kwargs.get('subsample', 1.0)
        nbootstrap = kwargs.get('nbootstrap', 1)
        model =  self._models[model_name]
        
        # check to make sure feature is valid
        if feature not in self._feature_names:
            raise Exception(f"Feature {feature} is not a valid feature")
        
        # Find the ranges to calculate the local effects over
        # Using xdata ensures each bin gets the same number of examples
        x1vals = np.percentile(self._examples[feature].values, np.arange(2.5, 97.5 + 5, 5))

        # get data in numpy format
        column_of_data = self._examples[feature].to_numpy()

        # append examples for histogram use
        hist_vals = column_of_data
 
        # get the bootstrap samples
        if nbootstrap > 1:
            bootstrap_examples = compute_bootstrap_samples(self._examples, 
                                        subsample=subsample, 
                                        nbootstrap=nbootstrap)
        else:
            bootstrap_examples = [self._examples.index.to_list()]

        # define ALE array
        ale = np.zeros((nbootstrap, x1vals.shape[0]-1))

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_examples):

            # get samples
            examples = self._examples.iloc[idx, :]

            # loop over all ranges
            for i in range(1, x1vals.shape[0]):

                # get subset of data
                df_subset = examples[(examples[feature] >= x1vals[i - 1]) & 
                                     (examples[feature] < x1vals[i])]

                # Without any observation, local effect on splitted area is null
                if len(df_subset) != 0:
                    lower_bound = df_subset.copy()
                    upper_bound = df_subset.copy()

                    lower_bound[feature] = x1vals[i - 1]
                    upper_bound[feature] = x1vals[i]

                    upper_bound = upper_bound.values
                    lower_bound = lower_bound.values

                    if self._classification:
                        effect = 100.0 * ( model.predict_proba(upper_bound)[:, 1]
                                        -  model.predict_proba(lower_bound)[:, 1] )
                    else:
                        effect = model.predict(upper_bound) - model.predict(lower_bound)

                    ale[k,i - 1] = np.mean(effect)

            # The accumulated effect
            ale[k,:] = ale[k,:].cumsum()
            mean_ale = ale[k,:].mean()
 
            # Now we have to center ALE function in order to obtain null expectation for ALE function
            ale[k,:] -= mean_ale
        
        results = {feature : {model_name :{}}}
        results[feature][model_name]['values'] = ale
        results[feature][model_name]['xdata1'] = 0.5 * (x1vals[1:] + x1vals[:-1])
        results[feature][model_name]['hist_data'] = hist_vals
        
        return results
    
    def calculate_second_order_ale(self, model_name, feature, **kwargs):
        """
            Computes second-order ALE function on two continuous features data.

            Parameters
            ----------
            feature : string
                The name of the feature to consider.
            xdata : array
                Quantiles of feature.
        """
        model =  self._models[model_name]
        
        # make sure there are two features...
        assert(len(feature) == 2), "Size of features must be equal to 2."

        # check to make sure both features are valid
        if (feature[0] not in self._feature_names): 
            raise TypeError(f'Feature {features[0]} is not a valid feature')

        if (feature[1] not in self._feature_names): 
            raise TypeError(f'Feature {features[1]} is not a valid feature')

        # create bins for computation for both features
        if x1vals is None:
            x1vals = np.percentile(self._examples[feature[0]].values, np.arange(2.5, 97.5 + 5, 5))
          
        if x2vals is None: 
            x2vals = np.percentile(self._examples[feature[1]].values, np.arange(2.5, 97.5 + 5, 5))

        # get the bootstrap samples
        if nbootstrap > 1:
            bootstrap_examples = compute_bootstrap_samples(self._examples, 
                                        subsample=subsample, 
                                        nbootstrap=nbootstrap)
        else:
            bootstrap_examples = [self._examples.index.to_list()]

        # define ALE array as 3D
        ale = np.zeros((nbootstrap, x1vals.shape[1] - 1, x1vals.shape[1] - 1))

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_examples):

            # get samples
            examples = self._examples.iloc[idx, :]

            # compute calculation over 2-d space
            for i in range(1, x1vals.shape[0]):
                for j in range(1, x1vals.shape[1]):

                    # Select subset of training data that falls within subset
                    df_subset = examples[ (examples[features[0]] >= x1vals[i - 1])
                                        & (examples[features[0]] <  x1vals[i])
                                        & (examples[features[1]] >= x2vals[j - 1])
                                        & (examples[features[1]] <  x2vals[j]) ]

                    # Without any observation, local effect on splitted area is null
                    if len(df_subset) != 0:

                        # get lower and upper bounds on accumulated grid
                        z_low = [df_subset.copy() for _ in range(2)]
                        z_up =  [df_subset.copy() for _ in range(2)]

                        # The main ALE idea that compute prediction difference between
                        # same data except feature's one
                        z_low[0][features[0]] = x1vals[i - 1]
                        z_low[0][features[1]] = x2vals[j - 1]
                        z_low[1][features[0]] = x1vals[i]
                        z_low[1][features[1]] = x2vals[j - 1]
                        z_up[0][features[0]]  = x1vals[i - 1]
                        z_up[0][features[1]]  = x2vals[j]
                        z_up[1][features[0]]  = x1vals[i]
                        z_up[1][features[1]]  = x2vals[j]

                        if self._classification is True:
                            effect = 100.0 * ( (model.predict_proba(z_up[1])[:, 1]
                                              - model.predict_proba(z_up[0])[:, 1])
                                             - (model.predict_proba(z_low[1])[:, 1]
                                              - model.predict_proba(z_low[0])[:, 1]) )
                        else:
                            effect = (model.predict(z_up[1]) - model.predict(z_up[0])
                                   - (model.predict(z_low[1]) - model.predict(z_low[0])))

                        ale[i - 1, j - 1] = np.mean(effect)

            # The accumulated effect
            ale[k,:,:] = np.cumsum(ale, axis=0)

            # Now we have to center ALE function in order to obtain null expectation for ALE function
            ale[k,:,:] -= ale.mean()
        
        temp_dict = {model_name : {feature :{}}}
        temp_dict[model_name][feature]['values'] = ale
        temp_dict[model_name][feature]['xdata1']  = 0.5 * (x1vals[1:] + x1vals[:-1])
        temp_dict[model_name][feature]['xdata2']  = 0.5 * (x2vals[1:] + x2vals[:-1])
        
        return temp_dict
        

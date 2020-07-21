import numpy as np
import pandas as pd
import multiprocessing
from itertools import product

from .utils import (compute_bootstrap_indices, 
                    merge_nested_dict, 
                    merge_dict
                   )
from .multiprocessing_utils import run_parallel, to_iterator
from .attributes import Attributes

def is_str(a):
    """Check if argument is a string"""
    return isinstance(a, str)

class AccumulatedLocalEffects(Attributes):
    """
    AccumulatedLocalEffects is a class for computing first- and second-order
    accumulated local effects (ALE) from Apley and Zhy et al. (2016). 
    The computations can take advantage of multiple cores for parallelization. 
    
    Attributes: 
        model : object, list, or dict
            a trained single scikit-learn model, or list of scikit-learn models, or
            dictionary of models where the key-value pairs are the model name as 
            a string identifier and prefit model object.
            
        examples : pandas.DataFrame or ndnumpy.array; shape = (n_examples, n_features)
            training or validation examples to evaluate.
            If ndnumpy array, make sure to specify the feature names
            
        model_output : "predict" or "predict_proba"
            What output of the model should be evaluated. 
            
        feature_names : defaults to None. Should only be set if examples is a
            nd.numpy array. Make sure it's a list
    
    Reference: 
        Apley, D. W., and J. Zhu, 2016: Visualizing the Effects of Predictor 
        Variables in Black Box Supervised Learning Models.
    
    """
    def __init__(self, model, examples, model_output, feature_names=None):
        # These functions from the inherited Attributes class
        self.set_model_attribute(model)
        self.set_examples_attribute(examples, feature_names)
        self.model_output = model_output
        
    def run_ale(self, features=None, nbins=25, 
                njobs=1, subsample=1.0, nbootstrap=1):
        """
        Runs the accumulated local effect calculation and returns a dictionary with all
        necessary inputs for plotting.
        
        Args:
            feature: str, list of strs, or list of 2-tuples of strs
                List of strings for first-order partial dependence, or list of tuples
                     for second-order 
            subsample: a float (between 0-1) for fraction of examples used in bootstrap
            nbootstrap: number of bootstrap iterations to perform. Defaults to 1 (no
                        bootstrapping).
            nbins : int 
                Number of bins to compute ALE in. 
        """   
        # Retrieve the string identifiers for the different models 
        model_str_ids = [name for name in self.models.keys()]
        
        #Check if features is a string
        if is_str(features):
            features = [features]
           
        # check first element of feature and see if of type tuple; assume second-order calculations
        if isinstance(features[0], tuple): 
            func =  self.compute_second_order_ale
        else:
            func = self.compute_first_order_ale
            
        args_iterator = to_iterator(model_str_ids, 
                                    features,
                                    [nbins],
                                    [subsample],
                                    [nbootstrap])
        results = run_parallel(
                   func = func,
                   args_iterator = args_iterator,
                   kwargs = {}, 
                   nprocs_to_use=njobs
            )
        
        # Unpack the results from the parallelization script
        if len(model_str_ids) > 1:
            results = merge_nested_dict(results)
        else:
            results = merge_dict(results)
            
        return results 
    
    def _get_percentiles(feature_values, nbins):
        pass
    
    def compute_first_order_ale(self, model_name, feature, nbins=30, subsample=1.0, nbootstrap=1):
        """
        Computes first-order ALE function on single continuous feature data.
        
        Script is based on the _first_order_ale_quant from 
        https://github.com/blent-ai/ALEPython/
        
        Args:
        ----------
            model_name : str
                 the string identifier for model in the attribute "models" dict
            feature : string
                The name of the feature to consider.
            nbins : int 
            subsample : float [0,1]
            nbootstrap
                
        Returns: 
        ----------
            results : nested dictionary 
                
        """
        model =  self.models[model_name]
        
        # check to make sure feature is valid
        if feature not in self.feature_names:
            raise Exception(f"Feature {feature} is not a valid feature")

        # get the bootstrap samples
        if nbootstrap > 1:
            bootstrap_indices = compute_bootstrap_indices(self.examples, 
                                        subsample=subsample, 
                                        nbootstrap=nbootstrap)
        else:
            bootstrap_indices = [self.examples.index.to_list()]

        # define ALE array
        ale = np.zeros((nbootstrap, nbins))
            
        print('starting calculations...')
        
        # for each bootstrap set
        for k, idx in enumerate(bootstrap_indices):
            examples = self.examples.iloc[idx, :]
            
            # Find the ranges to calculate the local effects over
            # Using xdata ensures each bin gets the same number of examples
            feature_values = examples[feature].values
            bin_edges = np.percentile(feature_values, 
                                  np.linspace(0.0, 100.0, nbins+1),
                                  interpolation="nearest"
                                )
            # if right=True, then the smallest value in data is not included in a bin.
            # Thus, Define the bins the feature samples fall into. Shift and clip to ensure we are
            # getting the index of the left bin edge and the smallest sample retains its index
            # of 0.
            indices = np.clip(
                np.digitize(feature_values, bin_edges, right=True) - 1, 0, None
                )

            # Assign the feature quantile values (based on its bin index) to two copied training datasets, 
            #one for each bin edge. Then compute the difference between the corresponding predictions
            predictions = []
            for offset in range(2):
                examples_temp = self.examples.copy()
                examples_temp[feature] = bin_edges[indices + offset]
                if self.model_output == 'probability':
                    # Assumes we only care about the positive class of a binary classification
                    # TODO : may need to FIX THIS! 
                    predictions.append(model.predict_proba(examples_temp)[:,1])
                elif self.model_output == 'raw':
                    predictions.append(model.predict(examples_temp))
        
            # The individual (local) effects.
            effects = predictions[1] - predictions[0]

            # Group the effects by their bin index 
            index_groupby = pd.DataFrame({"index": indices, "effects": effects}).groupby(
                "index"
            )
            # Compute the mean local effect for each bin 
            mean_effects = index_groupby.mean().to_numpy().flatten()
            
            # Accumulate (cumulative sum) the mean local effects
            # Essentially intergrating the derivative to regain
            # the original function. 
            ale[k,:] = mean_effects.cumsum()
            
            # Now we have to center ALE function in order to 
            # obtain null expectation for ALE function
            ale[k,:] -= ale[k,:].mean()
            
        results = {feature : {model_name :{}}}
        results[feature][model_name]['values'] = ale
        results[feature][model_name]['xdata1'] = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        results[feature][model_name]['hist_data'] = feature_values
        
        return results
    
    def compute_second_order_ale(self, model_name, features, nbins=30, subsample=1.0, nbootstrap=1):
        """
        Computes second-order ALE function on two continuous features data.
            
        Script is based on the _second_order_ale_quant from 
        https://github.com/blent-ai/ALEPython/
        
        To Do:
            NEEDS WORK!!! second bootstrap is all zeros!

        Args:
        ----------
            model_name : str
            features : string
                The name of the feature to consider.
            nbins : int
            subsample : float between [0,1]
            nbootstrap : int 
        
        Returns :
        ----------
            results : nested dict 
        """
        model =  self.models[model_name]
        
        # make sure there are two features...
        assert(len(features) == 2), "Size of features must be equal to 2."

        # check to make sure both features are valid
        if (features[0] not in self.feature_names): 
            raise TypeError(f'Feature {features[0]} is not a valid feature')

        if (features[1] not in self.feature_names): 
            raise TypeError(f'Feature {features[1]} is not a valid feature')

        # create bins for computation for both features
        feature_values = [self.examples[features[i]].values for i in range(2)]
        bin_edges = [np.percentile(f, np.linspace(0, 100.0, nbins+1)) for f in feature_values]

        # get the bootstrap samples
        if nbootstrap > 1:
            bootstrap_indices = compute_bootstrap_samples(self.examples, 
                                        subsample=subsample, 
                                        nbootstrap=nbootstrap)
        else:
            bootstrap_indices = [self.examples.index.to_list()]

        # define ALE array as 3D
        ale = np.zeros((nbootstrap, len(bin_edges[0]) - 1, len(bin_edges[1]) - 1))

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_indices):

            # get samples
            examples = self.examples.iloc[idx, :]
            
            # create bins for computation for both features
            feature_values = [examples[features[i]].values for i in range(2)]
            bin_edges = [np.percentile(f, np.linspace(0, 100.0, nbins+1)) for f in feature_values]
            
            # Define the bins the feature samples fall into. Shift and clip to ensure we are
            # getting the index of the left bin edge and the smallest sample retains its index
            # of 0.
            indices_list = [
                np.clip(np.digitize(f, b, right=True) - 1, 0, None)
                for f, b in zip(feature_values, bin_edges)
                ]

            # Invoke the predictor at the corners of the bins. Then compute the second order
            # difference between the predictions at the bin corners.
            predictions = {}
            for shifts in product(*(range(2),) * 2):
                examples_temp = examples.copy()
                for i in range(2):
                    examples_temp[features[i]] = bin_edges[i][indices_list[i] + shifts[i]]
                if self.model_output == 'probability':
                    predictions[shifts] = model.predict_proba(examples_temp)[:,1]
                elif self.model_output == 'raw':
                    predictions[shifts] = model.predict(examples_temp)
            
            # The individual (local) effects.
            effects = (predictions[(1, 1)] - predictions[(1, 0)]) - (
                    predictions[(0, 1)] - predictions[(0, 0)]
                    )

            # Group the effects by their indices along both axes.
            index_groupby = pd.DataFrame(
                {"index_0": indices_list[0], "index_1": indices_list[1], "effects": effects}
                    ).groupby(["index_0", "index_1"])

            # Compute mean effects.
            mean_effects = index_groupby.mean()
            
            # Get the number of samples in each bin.
            n_samples = index_groupby.size().to_numpy()

            # Get the indices of the mean values.
            group_indices = mean_effects.index
            valid_grid_indices = tuple(zip(*group_indices))
            
            # Create a 2D array of the number of samples in each bin.
            #samples_grid = np.zeros(ale.shape[1:])
            #samples_grid[valid_grid_indices] = n_samples
           
            # Extract only the data.
            mean_effects = mean_effects.to_numpy().flatten()
            ale[k,:,:][valid_grid_indices] = mean_effects
           
            # Compute the cumulative sums.
            ale[k,:,:] = np.cumsum(np.cumsum(ale[k,:,:], axis=0), axis=1)

            """
            # Subtract first order effects along both axes separately.
            for i in range(2):
                # Depending on `i`, reverse the arguments to operate on the opposite axis.
                flip = slice(None, None, 1 - 2 * i)

                # Undo the cumulative sum along the axis.
                ale_temp = ale[k,:,:]
                first_order = ale[(slice(1, None), ...)[flip]] - ale[(slice(-1), ...)[flip]]
                # Average the diffs across the other axis.
                first_order = (
                    first_order[(..., slice(1, None))[flip]]
                    + first_order[(..., slice(-1))[flip]]
                    ) / 2
                # Weight by the number of samples in each bin.
                #first_order *= samples_grid
                ## Take the sum along the axis.
                #first_order = np.sum(first_order, axis=1 - i)
                # Normalise by the number of samples in the bins along the axis.
                #first_order /= np.sum(samples_grid, axis=1 - i)
                # The final result is the cumulative sum (with an additional 0).
                first_order = np.array([0, *np.cumsum(first_order)]).reshape((-1, 1)[flip])

                # Subtract the first order effect.
                ale[k,:,:] -= first_order
            """
            # Now we have to center ALE function in order to obtain null expectation for ALE function
            ale[k,:,:] -= ale[k,:,:].mean()
            
            print(ale)
            
            results = {features : {model_name :{}}}
            results[features][model_name]['values'] = ale
            results[features][model_name]['xdata1'] = 0.5 * (bin_edges[0][1:] + bin_edges[0][:-1])
            results[features][model_name]['xdata2'] = 0.5 * (bin_edges[1][1:] + bin_edges[1][:-1])
        
            return results
            
           

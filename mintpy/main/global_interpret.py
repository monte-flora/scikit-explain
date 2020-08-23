import numpy as np
import pandas as pd
from copy import deepcopy
from math import sqrt
from scipy.spatial import cKDTree
import itertools
from functools import reduce
from operator import add

from sklearn.metrics import (roc_auc_score,
                             roc_curve,
                             average_precision_score, 
                             mean_squared_error
                            )

from ..common.utils import (compute_bootstrap_indices, 
                    merge_nested_dict, 
                    merge_dict,
                    is_str,
                    is_valid_feature,
                    is_regressor,
                    is_classifier, 
                    cartesian,
                    brier_skill_score
                   )
from ..common.multiprocessing_utils import run_parallel, to_iterator
from ..common.attributes import Attributes

from .PermutationImportance import sklearn_permutation_importance

class GlobalInterpret(Attributes):

    """
    GlobalInterpret incorporates important methods for explaining global model behavior 
    across all data examples. These include permutation importance and 
    partial dependence [Friedman (2001)]/accumulated local effects [Apley and Zhy et al. (2016)]
    which produce feature ranking and expected relationship between a feature 
    with the predict outcome, respectively.
    
    The permutation importance computations relies on the 
    PermutationImportance python package (Jergensen 2019) with slight modification for 
    use in mintpy. 
    
    Parts of the partial dependence code were based on 
    the computations in sklearn.inspection.partial_dependence (Pedregosa et al. 2011).
    
    Parts of the accumulated local effects 
    are based on ALEPython (Jumelle 2020).
    
    
    PartialDependence is a class for computing first- and second-order
    partial dependence (PD; Friedman (2001). Parts of the code were based on 
    the computations in sklearn.inspection.partial_dependence (Pedregosa et al. 2011). 
    Currently, the package handles regression and binary classification. 
    
    Attributes:
        model : pre-fit scikit-learn model object, or list of 
            Provide a list of model objects to compute the global methods for.
        
        model_names : str, list 
            List of model names for the model objects in model. 
            For internal and plotting purposes. 
            
        examples : pandas.DataFrame or ndnumpy.array. 
            Examples to explain. Typically, one should use the training dataset for 
            the global methods. 
            
        feature_names : list of strs
            If examples are ndnumpy.array, then provide the feature_names 
            (default is None; assumes examples are pandas.DataFrame).
            
        model_output : 'probability' or 'regression' 
            What is the expected model output. 'probability' uses the positive class of 
            the .predict_proba() method while 'regression' uses .predict(). 
            
        checked_attributes : boolean 
            Ignore this parameter; For internal purposes only
    
    Reference: 
        Friedman, J. H., 2001: GREEDY FUNCTION APPROXIMATION: A GRADIENT BOOSTING MACHINE. 
        Ann Statistics, 29, 1189–1232, https://doi.org/10.1214/aos/1013203451.
        
        Jumelle, M., 2020: ALEPython. 
        Github Python software library https://github.com/blent-ai/ALEPython.
        
        Jergensen, G., 2019: PermutationImportance. 
        Github Python software library https://github.com/gelijergensen/PermutationImportance.
        
        Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.  
    """
    def __init__(self, model, model_names, examples, targets=None,
                 model_output='probability', 
                 feature_names=None, checked_attributes=False):
        # These functions come from the inherited Attributes class  
        if not checked_attributes:
            self.set_model_attribute(model, model_names)
            self.set_examples_attribute(examples, feature_names)
            self.set_target_attribute(targets)
        else:
            self.models = model
            self.model_names = model_names
            self.examples = examples
            self.targets = targets
            self.feature_names = list(examples.columns)
           
        self.model_output = model_output
    
    def permutation_importance(self, n_vars=5, evaluation_fn="auprc",
            subsample=1.0, njobs=1, nbootstrap=1, scoring_strategy=None):

        """
        Performs multipass permutation importance using Eli's code.

            Parameters:
            -----------
            n_multipass_vars : integer
                number of variables to calculate the multipass permutation importance for.
            evaluation_fn : string or callable
                evaluation function
            subsample: float
                value of between 0-1 to subsample examples (useful for speedier results)
            njobs : interger or float
                if integer, interpreted as the number of processors to use for multiprocessing
                if float, interpreted as the fraction of proceesors to use
            nbootstrap: integer
                number of bootstrapp resamples
        """
        available_scores = ['auc', 'auprc', 'bss', 'mse']
        
        if not isinstance(evaluation_fn.lower(),str) and scoring_strategy is None:
            raise ValueError(
                ''' 
                The scoring_strategy argument is None! If you are using a user-define evaluation_fn 
                then scoring_strategy must be set! If a metric is positively-oriented (a higher value is better), 
                then set scoring_strategy = "argmin_of_mean" and if is negatively-oriented-
                (a lower value is better), then set scoring_strategy = "argmax_of_mean"
                ''') 
            
        if evaluation_fn.lower() == "auc":
            evaluation_fn = roc_auc_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == "auprc":
            evaluation_fn = average_precision_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == 'bss':
            evaluation_fn = brier_skill_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == 'mse':
            evaluation_fn = mean_squared_error
            scoring_strategy = "argmax_of_mean"
        else:
            raise ValueError(f"evaluation_fn is not set! Available options are {available_scores}") 
                
        targets = pd.DataFrame(data=self.targets, columns=['Test'])

        pi_dict = {}

        # loop over each model
        for model_name, model in self.models.items():

            print(f"Processing {model_name}...")

            pi_result = sklearn_permutation_importance(
                model            = model,
                scoring_data     = (self.examples.values, targets.values),
                evaluation_fn    = evaluation_fn,
                variable_names   = self.feature_names,
                scoring_strategy = scoring_strategy,
                subsample        = subsample,
                nimportant_vars  = n_vars,
                njobs            = njobs,
                nbootstrap       = nbootstrap,
            )

            pi_dict[model_name] = pi_result
              
        return pi_dict
    
    def _run_interpret_curves(self, method, features, nbins=25, njobs=1, subsample=1.0, nbootstrap=1):

        """
        Runs the interpretation curve (partial dependence or accumulated local effects) 
        calculations. Includes assessing whether the calculation is 1D or 2D and handling
        initializing the parallelization, subsampling data, and/or using bootstraping 
        to compute confidence intervals.
        
        Returns a nested dictionary with all neccesary inputs for plotting. 
        
        Args:
        ---------
            method : 'pd' or 'ale'
                determines whether to compute partial dependence ('pd') or 
                accumulated local effects ('ale').
        
            features: string, 2-tuple of strings, list of strings, or lists of 2-tuple strings
                feature names to compute partial dependence for. If 2-tuple, it will compute 
                the second-order partial dependence. 
            
            nbins : int 
                Number of evenly-spaced bins to compute PD/ALE over. 
             
            njobs : int or float 
                 if int, the number of processors to use for parallelization
                 if float, percentage of total processors to use for parallelization 
                 
            subsample : float (between 0-1)
                 Fraction of randomly sampled examples to evaluate (default is 1.0; no subsampling)
                 
            nbootstrap: number of bootstrap iterations to perform. Defaults to 1 (no
                        bootstrapping).
        """    
        #Check if features is a string
        if is_str(features) or isinstance(features, tuple):
            features = [features]
            
        if method == 'ale':
            # check first element of feature and see if of type tuple; assume second-order calculations
            if isinstance(features[0], tuple): 
                func =  self.compute_second_order_ale
            else:
                func = self.compute_first_order_ale
        elif method == 'pd':
            func = self.compute_partial_dependence
            

        args_iterator = to_iterator(self.model_names, 
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
        
        if len(self.model_names) > 1:
            results = merge_nested_dict(results)
        else:
            results = merge_dict(results)
                
        return results
    
    def compute_partial_dependence(self, model_name, features, nbins=30, subsample=1.0, nbootstrap=1):

        """
        Calculate the centered partial dependence.
        
        # Friedman, J., 2001: Greedy function approximation: a gradient boosting machine.
        Annals of Statistics, 29 (5), 1189–1232.
        ##########################################################################
        # Partial dependence plots fix a value for one or more predictors
        # for examples, passing these new data through a trained model, 
        # and then averaging the resulting predictions. After repeating this process
        # for a range of values of X*, regions of non-zero slope indicates that
        # where the ML model is sensitive to X* (McGovern et al. 2019). Only disadvantage is
        # that PDP do not account for non-linear interactions between X and the other predictors.
        #########################################################################

        Args: 
            model_name : str
                string identifier or key value for the model object dict
            features : str
                name of feature to compute PD for (string) 
            nbins : int 
                Number of evenly-spaced bins to compute PD
            subsample : float between 0-1
                Percent of randomly sampled examples to compute PD for.
            nbootstrap : int 
                Number of bootstrapping 
                
        Returns:
            pd, partial dependence values (in %, i.e., multiplied by 100.) 
        """
        # Retrieve the model object from the models dict attribute 
        model =  self.models[model_name]

         #Check if features is a string
        if is_str(features):
            features = [features]

        # Check if feature is valid
        is_valid_feature(features, self.feature_names)

        # Extract the values for the features 
        feature_values = [self.examples[f].to_numpy() for f in features]
        
        # Create a grid of values 
        grid = [np.linspace(np.amin(f), 
                           np.amax(f), 
                           nbins
                          ) for f in feature_values]

        # get the bootstrap samples
        if nbootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(self.examples, 
                                        subsample=subsample, 
                                        nbootstrap=nbootstrap)
        else:
            bootstrap_indices = [self.examples.index.to_list()]

        if self.model_output=='probability':
            prediction_method = model.predict_proba
        elif self.model_output == 'raw':
            prediction_method = model.predict

        pd_values = [ ]
        
        # for each bootstrap set
        for k, idx in enumerate(bootstrap_indices):
            # get samples
            examples = self.examples.iloc[idx, :]
            
            averaged_predictions = [ ]
            # for each value, set all indices to the value, 
            # make prediction, store mean prediction
            for value_set in cartesian(grid): 
                examples_temp = examples.copy()
                for i, feature in enumerate(features):
                    examples_temp.loc[:, feature] = value_set[i]
                    predictions = prediction_method(examples_temp.values)

                # average over samples
                averaged_predictions.append(np.mean(predictions, axis=0))        
                        
            averaged_predictions = np.array(averaged_predictions).T
          
            if self.model_output=='probability': 
                # Binary classification, shape is (2, n_points).
                # we output the effect of **positive** class
                # and convert to percentages
                averaged_predictions = averaged_predictions[1] * 100. 
            
            # Center the predictions 
            averaged_predictions -= np.mean(averaged_predictions)
        
            pd_values.append(averaged_predictions) 
        
        # Reshape the pd_values for second-order effects 
        pd_values = np.array(pd_values)
        if len(features) > 1:
            pd_values = pd_values.reshape(nbootstrap, nbins, nbins)
        else: 
            features = features[0]
        
        results = { features: {model_name : {}}}
        results[features][model_name]['values'] = pd_values
        results[features][model_name]['xdata1'] = grid[0]
        results[features][model_name]['xdata1_hist'] = feature_values[0]
        if np.shape(grid)[0] > 1:
            results[features][model_name]['xdata2'] = grid[1]
            results[features][model_name]['xdata2_hist'] = feature_values[1]
        
        return results

    def compute_first_order_ale(self, model_name, feature, nbins=30, subsample=1.0, nbootstrap=1):
        """
        Computes first-order ALE function on single continuous feature data.
        
        Script is largely the _first_order_ale_quant from 
        https://github.com/blent-ai/ALEPython/ with small modifications. 
        
        Args:
        ----------
            model_name : str
                 the string identifier for model in the attribute "models" dict
            feature : string
                The name of the feature to consider.
            nbins : int 
            subsample : float [0,1]
            nbootstrap : int
                
        Returns: 
        ----------
            results : nested dictionary 
                
        """
        model =  self.models[model_name]
        
        # check to make sure feature is valid
        if feature not in self.feature_names:
            raise Exception(f"Feature {feature} is not a valid feature")

        # get the bootstrap samples
        if nbootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(self.examples, 
                                        subsample=subsample, 
                                        nbootstrap=nbootstrap)
        else:
            bootstrap_indices = [self.examples.index.to_list()]
  

        # Using the original, unaltered feature values 
        # calculate the bin edges to be used in the bootstrapping. 
        original_feature_values = self.examples[feature].values
        bin_edges = np.unique( np.percentile(original_feature_values, 
                                  np.linspace(0, 100, nbins+1),
                                  interpolation="lower"
                                )
                                )
        
        # Initialize an empty ale array 
        ale = np.zeros((nbootstrap, len(bin_edges)-1))
        
        # for each bootstrap set
        for k, idx in enumerate(bootstrap_indices):
            examples = self.examples.iloc[idx, :]

            # Find the ranges to calculate the local effects over
            # Using xdata ensures each bin gets the same number of examples
            feature_values = examples[feature].values
            
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
                examples_temp = examples.copy()
                examples_temp[feature] = bin_edges[indices + offset]
                if self.model_output == 'probability':
                    predictions.append(model.predict_proba(examples_temp.values)[:,1]*100.) 
                elif self.model_output == 'raw':
                    predictions.append(model.predict(examples_temp.values))
        
            # The individual (local) effects.
            effects = predictions[1] - predictions[0]
            
            # Group the effects by their bin index 
            index_groupby = pd.DataFrame({"index": indices, "effects": effects}).groupby(
                "index"
            )
            
            # Compute the mean local effect for each bin 
            mean_effects = index_groupby.mean().to_numpy().flatten()
            
            # Accumulate (cumulative sum) the mean local effects.
            # Adding a 0 at the lower boundary of the first bin 
            # for the interpolation step in the next step
            ale_uninterpolated = np.array([0, *np.cumsum(mean_effects)])

            # Interpolate the ale to the center of the bins. 
            ale[k,:] = 0.5*(ale_uninterpolated[1:] + ale_uninterpolated[:-1])
            
            # Center the ALE by substracting the bin-size weighted mean. 
            ale[k,:] -= np.sum(ale[k,:]  * index_groupby.size() / examples.shape[0]) 
            
        results = {feature : {model_name :{}}}
        results[feature][model_name]['values'] = np.array(ale)
        results[feature][model_name]['xdata1'] = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        results[feature][model_name]['xdata1_hist'] = feature_values
        
        return results
    
    def _get_centers(self, x):
        return 0.5*(x[1:] + x[:-1]) 
    
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
        
        original_feature_values = [self.examples[features[i]].values for i in range(2)]

        bin_edges = [ np.unique( np.percentile(v, 
                                  np.linspace(0.0, 100.0, nbins+1),
                                  interpolation="lower")
                                )
                     for v in original_feature_values]
        
        # get the bootstrap samples
        if nbootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(self.examples, 
                                        subsample=subsample, 
                                        nbootstrap=nbootstrap)
        else:
            bootstrap_indices = [self.examples.index.to_list()]

        feature1_nbin_edges = len(bin_edges[0]) 
        feature2_nbin_edges = len(bin_edges[1]) 
            
        ale_set = [ ]    
        
        # for each bootstrap set
        for k, idx in enumerate(bootstrap_indices):

            ale = np.ma.MaskedArray(
                np.zeros((feature1_nbin_edges, feature2_nbin_edges)),
                mask=np.ones(( feature1_nbin_edges, feature2_nbin_edges)),
                fill_value = np.nan
            )    
            
            # get samples
            examples = self.examples.iloc[idx, :]
            
            # create bins for computation for both features
            feature_values = [examples[features[i]].values for i in range(2)]
            
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
            for shifts in itertools.product(*(range(2),) * 2):
                examples_temp = examples.copy()
                for i in range(2):
                    examples_temp[features[i]] = bin_edges[i][indices_list[i] + shifts[i]]
                if self.model_output == 'probability':
                    predictions[shifts] = model.predict_proba(examples_temp)[:,1]*100.
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
            
            # Get the indices of the mean values.
            group_indices = mean_effects.index
            valid_grid_indices = tuple(zip(*group_indices))
            # Extract only the data.
            mean_effects = mean_effects.to_numpy().flatten()
            
            # Get the number of samples in each bin.
            n_samples = index_groupby.size().to_numpy()

            # Create a 2D array of the number of samples in each bin.
            samples_grid = np.zeros((feature1_nbin_edges-1, feature2_nbin_edges-1))
            samples_grid[valid_grid_indices] = n_samples
           
            
            # Mark the first row/column as valid, since these are meant to contain 0s.
            ale.mask[0, :] = False
            ale.mask[:, 0] = False
            
             # Place the mean effects into the final array.
            # Since `ale` contains `len(quantiles)` rows/columns the first of which are
            # guaranteed to be valid (and filled with 0s), ignore the first row and column.
            ale[1:, 1:][valid_grid_indices] = mean_effects
            
            # Record where elements were missing.
            missing_bin_mask = ale.mask.copy()[1:, 1:]
            
            if np.any(missing_bin_mask):
                # Replace missing entries with their nearest neighbours.

                # Calculate the dense location matrices (for both features) of all bin centres.
                centers_list = np.meshgrid(
                    *(self._get_centers(quantiles) for quantiles in bin_edges), indexing="ij"
                )

                # Select only those bin centres which are valid (had observation).
                valid_indices_list = np.where(~missing_bin_mask)
                tree = cKDTree(
                    np.hstack(
                        tuple(
                            centers[valid_indices_list][:, np.newaxis]
                            for centers in centers_list
                        )
                    )
                )

                row_indices = np.hstack(
                    [inds.reshape(-1, 1) for inds in np.where(missing_bin_mask)]
                )
                # Select both columns for each of the rows above.
                column_indices = np.hstack(
                    (
                        np.zeros((row_indices.shape[0], 1), dtype=np.int8),
                    np.ones((row_indices.shape[0], 1), dtype=np.int8),
                    )
                )

                # Determine the indices of the points which are nearest to the empty bins.
                nearest_points = tree.query(tree.data[row_indices, column_indices])[1]

                nearest_indices = tuple(
                    valid_indices[nearest_points] for valid_indices in valid_indices_list
                )

                # Replace the invalid bin values with the nearest valid ones.
                ale[1:, 1:][missing_bin_mask] = ale[1:, 1:][nearest_indices]
            
            # Compute the cumulative sums.
            ale = np.cumsum(np.cumsum(ale, axis=0), axis=1)
            
            # Subtract first order effects along both axes separately.
            for i in range(2):
                # Depending on `i`, reverse the arguments to operate on the opposite axis.
                flip = slice(None, None, 1 - 2 * i)

                # Undo the cumulative sum along the axis.
                first_order = ale[(slice(1, None), ...)[flip]] - ale[(slice(-1), ...)[flip]]
                # Average the diffs across the other axis.
                first_order = (
                    first_order[(..., slice(1, None))[flip]]
                    + first_order[(..., slice(-1))[flip]]
                ) / 2
                # Weight by the number of samples in each bin.
                first_order *= samples_grid
                # Take the sum along the axis.
                first_order = np.sum(first_order, axis=1 - i)
                # Normalise by the number of samples in the bins along the axis.
                first_order /= np.sum(samples_grid, axis=1 - i)
                # The final result is the cumulative sum (with an additional 0).
                first_order = np.array([0, *np.cumsum(first_order)]).reshape((-1, 1)[flip])

                #print(first_order) 
                
                # Subtract the first order effect.
                ale -= first_order
            
            
            # Compute the ALE at the bin centres.
            ale = (
                reduce(
                    add,
                    (
                        ale[i : ale.shape[0] - 1 + i, j : ale.shape[1] - 1 + j]
                        for i, j in list(itertools.product(*(range(2),) * 2))
                    ),
                )
                / 4
            )
            
            # Center the ALE by subtracting its expectation value.
            ale -= np.sum(samples_grid * ale) /  len(examples)
            
            ale_set.append(ale)

        results = {features : {model_name :{}}}
        results[features][model_name]['values'] = ale_set
        results[features][model_name]['xdata1'] = 0.5 * (bin_edges[0][1:] + bin_edges[0][:-1])
        results[features][model_name]['xdata2'] = 0.5 * (bin_edges[1][1:] + bin_edges[1][:-1])
        results[features][model_name]['xdata1_hist'] = original_feature_values[0]
        results[features][model_name]['xdata2_hist'] = original_feature_values[1]
        
        
        return results
            
    
    def friedman_h_statistic(self, model_name, feature_tuple, nbins=15, subsample=1.0):
        """
        Compute the H-statistic for two-way interactions between two features. 
        
        Args:
            model_name : str
            feature_tuple : 2-tuple of strs
            nbins : int
        
        Returns:
        """
        self.model_names = [model_name]
        feature1, feature2 = feature_tuple
        
        features = [feature1, feature2] 
        
        results = self._run_interpret_curves(method='pd', 
                                             features=features, 
                                             nbins=nbins, 
                                             njobs=2, 
                                             subsample=subsample, 
                                             nbootstrap=1
                                            )
        
        feature1_pd = results[feature1][model_name]['values'].squeeze()
        feature2_pd = results[feature2][model_name]['values'].squeeze()

        combined_results = self.compute_partial_dependence(model_name, feature_tuple, nbins)
        combined_pd = combined_results[feature_tuple][model_name]['values'].squeeze()
       
        pd_decomposed = feature1_pd[:,np.newaxis] + feature2_pd[np.newaxis,:]
        numer = (combined_pd - pd_decomposed)**2
        denom = (combined_pd)**2
        H_squared = np.sum(numer) / np.sum(denom)
        
        return sqrt(H_squared)

        
        
        
        
        
        

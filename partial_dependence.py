import numpy as np
import pandas as pd

from .utils import (compute_bootstrap_indices, 
                    merge_nested_dict, 
                    merge_dict,
                    is_str,
                    is_valid_feature,
                    is_regressor,
                    is_classifier, 
                    cartesian
                   )
from .multiprocessing_utils import run_parallel, to_iterator
from copy import deepcopy
from .attributes import Attributes


class PartialDependence(Attributes):

    """
    PartialDependence is a class for computing first- and second-order
    partial dependence (PDP) from Friedman (2001). 
    The computations can take advantage of multiple cores for parallelization. 
    
    Args:
        model : a single (or multiple) scikit-learn models represented as a dictionary.
            Create a dictionary such as { "RandomForest" : rf_sklearn_model }
        examples : pandas DataFrame or ndnumpy array. If ndnumpy array, make sure
            to specify the feature names
        feature_names : defaults to None. Should only be set if examples is a 
            nd.numpy array. Make sure it's a list
        model_output : str 
            
            
    Reference: 
        Friedman, J. H., 2001: GREEDY FUNCTION APPROXIMATION: A GRADIENT BOOSTING MACHINE. 
        Ann Statistics, 29, 1189–1232, https://doi.org/10.1214/aos/1013203451.
    """
    def __init__(self, model, examples, model_output='probability', 
            feature_names=None):
        # These functions come from the inherited Attributes class
        self.set_model_attribute(model)
        self.set_examples_attribute(examples, feature_names)
        self.model_output = model_output
        
    def run_pd(self, features, nbins=25, njobs=1, subsample=1.0, nbootstrap=1):

        """
        Runs the partial dependence calculation by handling parallelization, 
        subsampling data, and/or using bootstraping to compute confidence intervals.
        
        Returns a nested dictionary with all neccesary inputs for plotting. 
        
        Args:
        ---------
            features: string, 2-tuple of strings, list of strings, or lists of 2-tuple strings
                feature names to compute partial dependence for. If 2-tuple, it will compute 
                the second-order partial dependence. 
             njobs : int or float 
                 if int, the number of processors to use for parallelization
                 if float, percentage of total processors to use for parallelization 
             subsample : float (between 0-1)
                 Fraction of randomly sampled examples to evaluate (default is 1.0; no subsampling)
             nbootstrap
 
 
            subsample: a float (between 0-1) for fraction of examples used in bootstrap
            nbootstrap: number of bootstrap iterations to perform. Defaults to 1 (no
                        bootstrapping).
        """    
        model_str_ids = [name for name in self.models.keys()]

        #Check if features is a string
        if is_str(features) or isinstance(features, tuple):
            features = [features]

        args_iterator = to_iterator(model_str_ids, 
                                    features,
                                    [nbins],
                                    [subsample],
                                    [nbootstrap])

        results = run_parallel(
                   func = self.compute_partial_dependence,
                   args_iterator = args_iterator,
                   kwargs = {}, 
                   nprocs_to_use=njobs
            )
        
        if len(model_str_ids) > 1:
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
        if nbootstrap > 1:
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
            examples = self.examples.iloc[idx, :].copy()
            
            averaged_predictions = [ ]
            # for each value, set all indices to the value, 
            # make prediction, store mean prediction
            for value_set in cartesian(grid): 
                examples_temp = examples.copy()
                for i, feature in enumerate(features):
                    examples_temp.loc[:, feature] = value_set[i]
                    predictions = prediction_method(examples_temp)

                # average over samples
                averaged_predictions.append(np.mean(predictions, axis=0))        
                        
            averaged_predictions = np.array(averaged_predictions).T
          
            if self.model_output=='probability': 
                #print("""Output is a probability; 
                #      Only outputing the effect of the positive class!""") 
                # Binary classification, shape is (2, n_points).
                # we output the effect of **positive** class
                averaged_predictions = averaged_predictions[1]  
        
            # Center the predictions 
            averaged_predictions -= np.mean(averaged_predictions)
        
            pd_values.append(averaged_predictions) 
        
        # Reshape the pd_values for second-order effects 
        pd_values = np.array(pd_values)
        if len(features) > 1:
            pd_values = pd_values.reshape(nbootstrap, nbins, nbins)

        key = tuple(features) 
        results = {key : {model_name : {}}}
        results[key][model_name]['values'] = pd_values
        results[key][model_name]['xdata1'] = grid[0]
        if len(features) > 1:
            results[key][model_name]['xdata2'] = grid[1]
        results[key][model_name]['hist_data'] = feature_values
        
        return results

    
    def friedman_h_statistic(self, model_name, feature_tuple, nbins=15):
        """
        Compute the H-statistic for two-way interactions between two features. 
        
        ToDo: Need to center the PD! 
        
        """
        feature1, feature2 = feature_tuple
        
        feature1_results = self.compute_1d_partial_dependence(model_name, feature1, nbins=nbins)
        feature2_results = self.compute_1d_partial_dependence(model_name, feature2, nbins=nbins)
        
        feature1_pd = feature1_results[feature1][model_name]['values'].squeeze()
        feature2_pd = feature2_results[feature2][model_name]['values'].squeeze()
        
        x1 = feature1_results[feature1][model_name]['xdata1']
        x2 = feature2_results[feature2][model_name]['xdata1']
        
        combined_results = self.compute_2d_partial_dependence(model_name, feature_tuple, nbins)
        
        combined_pd = combined_results[feature_tuple][model_name]['values'].squeeze()
        
        feature1_pd -= feature1_pd.mean()
        feature2_pd -= feature2_pd.mean()
        combined_pd -= combined_pd.mean()
        
        pd_decomposed = feature1_pd[:,np.newaxis] + feature2_pd[np.newaxis,:]
        numer = (combined_pd - pd_decomposed)**2
        denom = (combined_pd)**2
        H_squared = np.sum(numer) / np.sum(denom)
        
        return sqrt(H_squared)

        
        
        
        
        
        

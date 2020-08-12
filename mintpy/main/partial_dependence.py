import numpy as np
import pandas as pd
from copy import deepcopy
from math import sqrt

from ..common.utils import (compute_bootstrap_indices, 
                    merge_nested_dict, 
                    merge_dict,
                    is_str,
                    is_valid_feature,
                    is_regressor,
                    is_classifier, 
                    cartesian
                   )
from ..common.multiprocessing_utils import run_parallel, to_iterator
from ..common.attributes import Attributes


class PartialDependence(Attributes):

    """
    PartialDependence is a class for computing first- and second-order
    partial dependence (PD; Friedman (2001). Parts of the code were based on 
    the computations in sklearn.inspection.partial_dependence (Pedregosa et al. 2011). 
    Currently, the package handles regression and binary classification. 
    
    Attributes:
        model : pre-fit scikit-learn model object, or list of 
            Provide a list of model objects to compute PD 
            for multiple model predictions.
        
        model_names : str, list 
            List of model names for the model objects in model. 
            For internal and plotting purposes. 
            
        examples : pandas.DataFrame or ndnumpy.array. 
            Examples used to train the model.
            
        feature_names : list of strs
            If examples are ndnumpy.array, then provide the feature_names 
            (default is None; assumes examples are pandas.DataFrame).
            
        model_output : 'probability' or 'regression' 
            What is the expected model output. 'probability' uses the positive class of 
            the .predict_proba() method while 'regression' uses .predict().
            
        checked_attributes : boolean 
            For internal purposes only
    
    Reference: 
        Friedman, J. H., 2001: GREEDY FUNCTION APPROXIMATION: A GRADIENT BOOSTING MACHINE. 
        Ann Statistics, 29, 1189–1232, https://doi.org/10.1214/aos/1013203451.
        
        Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.  
    """
    def __init__(self, model, model_names, examples, 
                 model_output='probability', 
                 feature_names=None, checked_attributes=False):
        # These functions come from the inherited Attributes class  
        if not checked_attributes:
            self.set_model_attribute(model, model_names)
            self.set_examples_attribute(examples, feature_names)
        else:
            self.models = model
            self.model_names = model_names
            self.examples = examples
            self.feature_names = list(examples.columns)
           
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
            
            nbins : int 
                Number of evenly-spaced bins to compute PD over. 
             
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

        args_iterator = to_iterator(self.model_names, 
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
        if np.shape(grid)[0] > 1:
            results[features][model_name]['xdata2'] = grid[1]
        results[features][model_name]['hist_data'] = feature_values
        
        return results

    def friedman_h_statistic(self, model_name, feature_tuple, nbins=15):
        """
        Compute the H-statistic for two-way interactions between two features. 
        
        Args:
            model_name : str
            feature_tuple : 2-tuple of strs
            nbins : int
        
        Returns:
        """
        feature1, feature2 = feature_tuple
        
        feature1_results = self.compute_partial_dependence(model_name, feature1, nbins=nbins)
        feature2_results = self.compute_partial_dependence(model_name, feature2, nbins=nbins)
        
        feature1_pd = feature1_results[feature1][model_name]['values'].squeeze()
        feature2_pd = feature2_results[feature2][model_name]['values'].squeeze()

        combined_results = self.compute_partial_dependence(model_name, feature_tuple, nbins)
        combined_pd = combined_results[feature_tuple][model_name]['values'].squeeze()
       
        pd_decomposed = feature1_pd[:,np.newaxis] + feature2_pd[np.newaxis,:]
        numer = (combined_pd - pd_decomposed)**2
        denom = (combined_pd)**2
        H_squared = np.sum(numer) / np.sum(denom)
        
        return sqrt(H_squared)

        
        
        
        
        
        

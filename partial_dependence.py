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
from math import sqrt


class PartialDependence(Attributes):

    """
    PartialDependence is a class for computing first- and second-order
    partial dependence (PD; Friedman (2001). Parts of the code were based on
    the computations in sklearn.inspection.partial_dependence (Pedregosa et al. 2011).
    Currently, the package handles regression and binary classification.

    Reference:
        Friedman, J. H., 2001: GREEDY FUNCTION APPROXIMATION: A GRADIENT BOOSTING MACHINE.
        Ann Statistics, 29, 1189–1232, https://doi.org/10.1214/aos/1013203451.

        Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
    """
    def __init__(self, model, model_names, examples,
                 model_output='probability',
                 feature_names=None, checked_attributes=False):

        """
        Args:
        ----------
        model : single or list of pre-fit scikit-learn model objects
            Provide a list of model objects to compute ALE
            for multiple model predictions.

        model_names : string, or list of strings
            List of model names for the model objects in model.
            For internal and plotting purposes.

        examples : pandas.DataFrame, or ndnumpy.array.
            Examples used to train the model.

        feature_names : list of strings
            If examples are ndnumpy.array, then provide the feature_names
            (default is None; assumes examples are pandas.DataFrame).

        model_output : 'probability' or 'regression'
            What is the expected model output. 'probability' uses the positive class of
            the .predict_proba() method while 'regression' uses .predict().

        checked_attributes : boolean
            For internal purposes only
        """

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

        """Runs partial dependence calculation and populates a
           dictionary.

        Args:
        ----------
        features : single string, list of strings, single tuple, or
                   list of 2-tuples of strings
            If list of strings, computes the first-order PD for the given features
            If list of 2-tuples of strings, computes the second-order PD for the
                pairs of features.

        nbins : int
            Number of evenly-spaced bins to compute ALE. Defaults to 25.

        njobs : int or float
            Number of processes to run. Default is 1

        subsample: float (between 0-1)
            Fraction of examples used in bootstrap resampling. Default is 1.0

        nbootstrap: int
            Number of bootstrap iterations to perform. Defaults to 1 (no
                bootstrapping).

        Return:
        -------
        dictionary of PD values for each model and feature set specified.
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

        """Calculate the centered partial dependence

        Args:
        ----------
        model_name : string
             The string identifier for model in the attribute "models" dict

        feature : string
            The name of the feature to consider.

        nbins : int
            Number of evenly-spaced bins to compute ALE. Defaults to 30.

        njobs : int or float
            Number of processes to run. Default is 1

        subsample: float (between 0-1)
            Fraction of examples used in bootstrap resampling. Default is 1.0

        nbootstrap: int
            Number of bootstrap iterations to perform. Defaults to 1 (no
                bootstrapping).

        Returns:
        ----------
        results : dictionary
            A dictionary of PD values, and data used to create PD.
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
                    predictions = prediction_method(examples_temp)

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

        """Compute the H-statistic for two-way interactions between two features.

        Args:
        ----------
        model_name : string
            The model to compute the statistic for

        feature_tuple : 2-tuple of strings
            Tuple of the two features to compute the statistic for

        nbins : int
            Number of bins for PD. See compute_partial_dependence()

        Returns:
        ----------
        result: float
            The H-Statistic
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

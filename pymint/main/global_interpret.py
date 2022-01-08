import numpy as np
import pandas as pd
import xarray as xr
from math import sqrt
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import itertools
from functools import reduce
from operator import add
import traceback
from copy import copy
from inspect import currentframe, getframeinfo
from pandas.core.common import SettingWithCopyError
from joblib import delayed, Parallel


from sklearn.linear_model import LinearRegression
from sklearn import cluster
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    mean_squared_error,
    r2_score
)

from ..common.utils import (
    compute_bootstrap_indices,
    merge_nested_dict,
    merge_dict,
    is_str,
    is_valid_feature,
    is_regressor,
    is_classifier,
    cartesian,
    brier_skill_score,
    norm_aupdc,
    to_xarray,
    order_groups,
    determine_feature_dtype,
    check_is_permuted, 
    to_pymint_importance
)

from ..common.multiprocessing_utils import run_parallel, to_iterator
from ..common.attributes import Attributes

from .PermutationImportance import sklearn_permutation_importance
from .PermutationImportance.utils import bootstrap_generator


class GlobalInterpret(Attributes):

    """
    InterpretToolkit inherits functionality from GlobalInterpret and is not meant to be
    instantiated by the end-user. 
    
    GlobalInterpret incorporates important methods for explaining global estimator behavior
    across all data instances. These include permutation importance and
    partial dependence (Friedman 2001) / accumulated local effects (Apley and Zhy et al. 2016)
    which produce feature ranking and expected relationship between a feature
    with the predict outcome, respectively.

    The permutation importance computations rely on the
    PermutationImportance python package (Jergensen 2019) with slight modification for
    use in py-mint.

    Parts of the partial dependence code were based on
    the computations in sklearn.inspection.partial_dependence (Pedregosa et al. 2011).

    Parts of the accumulated local effects
    are based on ALEPython (Jumelle 2020) and PyALE.


    PartialDependence is a class for computing first- and second-order
    partial dependence (PD; Friedman (2001). Parts of the code were based on
    the computations in sklearn.inspection.partial_dependence (Pedregosa et al. 2011).
    Currently, the package handles regression and binary classification.

    Attributes
    --------------
    estimators : object, list of objects
        A fitted estimator object or list thereof implementing `predict` or 
        `predict_proba`.
        Multioutput-multiclass classifiers are not supported.
        
    estimator_names : string, list
        Names of the estimators (for internal and plotting purposes)

    X : {array-like or dataframe} of shape (n_samples, n_features)
        Training or validation data used to compute the IML methods.
        If ndnumpy array, make sure to specify the feature names

    y : {list or numpy.array} of shape = (n_samples,)
        y values.

    estimator_output : "raw" or "probability"
        What output of the estimator should be evaluated. default is None. If None, 
        InterpretToolkit will determine internally what the estimator output is. 

    feature_names : array-like of shape (n_features,), dtype=str, default=None
        Name of each feature; `feature_names[i]` holds the name of the feature
        with index `i`.
        By default, the name of the feature corresponds to their numerical
        index for NumPy array and their column name for pandas dataframe. 
        Feature names are only required if X is an ndnumpy.array, it will be 
        converted to a pandas.DataFrame internally. 
        
    Reference
    ------------
        Friedman, J. H., 2001: GREEDY FUNCTION APPROXIMATION: A GRADIENT BOOSTING MACHINE.
        Ann Statistics, 29, 1189–1232, https://doi.org/10.1214/aos/1013203451.

        Jumelle, M., 2020: ALEPython.
        Github Python software library https://github.com/blent-ai/ALEPython.

        Jergensen, G., 2019: PermutationImportance.
        Github Python software library https://github.com/gelijergensen/PermutationImportance.

        Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
    """

    def __init__(
        self,
        estimators,
        estimator_names,
        X,
        y=None,
        estimator_output=None,
        feature_names=None,
        checked_attributes=False,
    ):
        # These functions come from the inherited Attributes class
        if not checked_attributes:
            self.set_estimator_attribute(estimators, estimator_names)
            self.set_X_attribute(X, feature_names)
            self.set_y_attribute(y)
        else:
            self.estimators = estimators
            self.estimator_names = estimator_names
            self.X = X
            self.y = y
            self.feature_names = list(X.columns)

        self.estimator_output = estimator_output

        
    def _to_scorer(self, evaluation_fn):
        """
        FOR INTERNAL PURPOSES ONLY.
        Converts a string to an evaluation function.
        """
        available_scores = ["auc", "auprc", "bss", "mse", "norm_aupdc"]
        
        if evaluation_fn == "auc":
            evaluation_fn = roc_auc_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn == "auprc":
            evaluation_fn = average_precision_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn == "norm_aupdc":
            evaluation_fn = norm_aupdc
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn == "bss":
            evaluation_fn = brier_skill_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn == "mse":
            evaluation_fn = mean_squared_error
            scoring_strategy = "argmax_of_mean"
        else:
            raise ValueError(
                    f"evaluation_fn is not set! Available options are {available_scores}"
                )
        return evaluation_fn, scoring_strategy
        
        
    def calc_permutation_importance(
        self,
        n_vars=5,
        evaluation_fn="auprc",
        subsample=1.0,
        n_jobs=1,
        n_permute=1,
        scoring_strategy=None,
        direction='backward',
        verbose=False,
        return_iterations=True,
        random_seed=1, 
    ):

        """
        Performs single-pass and/or multi-pass permutation importance using the PermutationImportance
        package.

        See calc_permutation_importance in IntepretToolkit for documentation.

        """

        if isinstance(evaluation_fn, str):
            evaluation_fn = evaluation_fn.lower()
            is_str=True

        if not isinstance(evaluation_fn, str) and scoring_strategy is None:
            raise ValueError(
                """ 
                The scoring_strategy argument is None! If you are using a non-default evaluation_fn 
                then scoring_strategy must be set! If the metric is positively-oriented (a higher value is better), 
                then set scoring_strategy = "argmin_of_mean" and if it is negatively-oriented-
                (a lower value is better), then set scoring_strategy = "argmax_of_mean"
                """
            )
        
        if isinstance(evaluation_fn,str):    
            evaluation_fn, scoring_strategy = self._to_scorer(evaluation_fn)

        if is_str:
            if direction == 'forward':
                if 'max' in scoring_strategy:
                    scoring_strategy = scoring_strategy.replace('max', 'min') 
                else:
                    scoring_strategy = scoring_strategy.replace('min', 'max') 

        y = pd.DataFrame(data=self.y, columns=["Test"])

        pi_dict = {}

        # loop over each estimator
        for estimator_name, estimator in self.estimators.items():
            pi_result = sklearn_permutation_importance(
                model=estimator,
                scoring_data=(self.X.values, y.values),
                evaluation_fn=evaluation_fn,
                variable_names=self.feature_names,
                scoring_strategy=scoring_strategy,
                subsample=subsample,
                nimportant_vars=n_vars,
                njobs=n_jobs,
                n_permute=n_permute,
                verbose=verbose,
                direction=direction,
                random_seed=random_seed,
            )

            pi_dict[estimator_name] = pi_result

            del pi_result

        data = {}
        for estimator_name in self.estimator_names:
            for func in ["retrieve_multipass", "retrieve_singlepass"]:
                adict = getattr(pi_dict[estimator_name], func)()
                features = np.array(list(adict.keys()))
                rankings = np.argsort([adict[f][0] for f in features])
                top_features = features[rankings]
                scores = np.array([adict[f][1] for f in top_features])
                pass_method = func.split("_")[1]

                data[f"{pass_method}_rankings__{estimator_name}"] = (
                    [f"n_vars_{pass_method}"],
                    top_features,
                )
                data[f"{pass_method}_scores__{estimator_name}"] = (
                    [f"n_vars_{pass_method}", "n_permute"],
                    scores,
                )
            data[f"original_score__{estimator_name}"] = (
                ["n_permute"],
                pi_dict[estimator_name].original_score,
            )

        if return_iterations:
            for estimator_name in self.estimator_names:
                temp_results = getattr(pi_dict[estimator_name], 'retrieve_all_iterations')()
                temp_scores = []
                temp_features = []
                for adict in temp_results:
                    features = np.array(list(adict.keys()))
                    rankings = np.argsort([adict[f][0] for f in features])
                    top_features = features[rankings]
                    scores = np.array([adict[f][1] for f in top_features])
                    pass_method = func.split("_")[1]
                    #Just retrieve the second most important feature. 
                    temp_scores.append(scores[1])
                    temp_features.append(top_features[1])
                    
                data[f"second_place_scores__{estimator_name}"] = (
                    [f"n_vars_second_place", "n_permute"],
                    temp_scores,
                )
                data[f"second_place_rankings__{estimator_name}"] = (
                    [f"n_vars_second_place"],
                    temp_features,
                )   


        results_ds = to_xarray(data)

        return results_ds

    def _run_interpret_curves(
        self,
        method,
        features=None,
        n_bins=25,
        n_jobs=1,
        subsample=1.0,
        n_bootstrap=1,
        feature_encoder=None,
        random_seed=42, 
    ):

        """
        Runs the interpretation curve (partial dependence, accumulated local effects,
        or individual conditional expectations.) calculations.
        Includes assessing whether the calculation is 1D or 2D and handling
        initializing the parallelization, subsampling data, and/or using bootstraping
        to compute confidence intervals.

        Returns a nested dictionary with all neccesary inputs for plotting.

        Parameters
        ---------
        method : 'pd' , 'ale', or 'ice'
                determines whether to compute partial dependence ('pd'),
                accumulated local effects ('ale'), or individual conditional expectations ('ice')

        features: string, 2-tuple of strings, list of strings, or lists of 2-tuple strings
                feature names to compute for. If 2-tuple, it will compute
                the second-order results.

        n_bins : int
            Number of evenly-spaced bins to compute PD/ALE over.

        n_jobs : int or float
            if int, the number of processors to use for parallelization
            if float, percentage of total processors to use for parallelization

        subsample : float or integer
            if value between 0-1 interpreted as fraction of total X to use
                    if value > 1, interpreted as the number of X to randomly sample
                        from the original dataset.

        n_bootstrap: integer
            Number of bootstrap iterations to perform. Defaults to 1 (no bootstrapping).
                        
        Returns
        ---------
        
        results_ds : xarray.Dataset 
        
        """
        self.random_seed = random_seed
        # Check if features is a string
        if is_str(features) or isinstance(features, tuple):
            features = [features]

        if not isinstance(features[0], tuple):
            features, cat_features = determine_feature_dtype(self.X, features)
        else:
            cat_features = []

        results = []
        cat_results = []
        if len(features) > 0:
            if method == "ale":
                # check first element of feature and see if the type is a tuple; assume second-order calculations
                if isinstance(features[0], tuple):
                    func = self.compute_second_order_ale
                else:
                    func = self.compute_first_order_ale
            elif method == "pd":
                func = self.compute_partial_dependence
            elif method == "ice":
                func = self.compute_individual_cond_expect

            args_iterator = to_iterator(
                self.estimator_names,
                features,
                [n_bins],
                [subsample],
                [n_bootstrap],
            )

            total = len(features) * len(self.estimator_names) 
            results = run_parallel(
                func=func, args_iterator=args_iterator, kwargs={}, nprocs_to_use=n_jobs, 
                total=total, 
            )

        if len(cat_features) > 0:
            if method == "ale":
                func = self.compute_first_order_ale_cat
            elif method == "pd":
                func = self.compute_partial_dependence
            elif method == "ice":
                func = self.compute_individual_cond_expect

            args_iterator = to_iterator(
                self.estimator_names,
                cat_features,
                [subsample],
                [n_bootstrap],
                [feature_encoder],
            )

            total = len(cat_features) * len(self.estimator_names) 
            cat_results = run_parallel(
                func=func, args_iterator=args_iterator, kwargs={}, nprocs_to_use=n_jobs, 
                total=total,
            )

        results = cat_results + results

        results = merge_dict(results)
        results_ds = to_xarray(results)

        return results_ds

    def _store_results(
        self, method, estimator_name, features, ydata, xdata, hist_data, 
        ice_X=None, categorical=False
    ):
        """
        FOR INTERNAL PURPOSES ONLY.
        
        Store the results of the ALE/PD/ICE calculations into a dict,
        which is converted to an xarray.Dataset
        
        """
        results = {}

        feature1 = f"{features[0]}" if isinstance(features, tuple) else features
        feature2 = f"__{features[1]}" if isinstance(features, tuple) else ""

        y_shape = ["n_bootstrap", f"n_bins__{feature1}", f"n_bins{feature2}"]
        if feature2 == "":
            y_shape = y_shape[:-1]

        hist_data1 = hist_data[0]

        xdata2 = None
        if method == "pd" or method == "ice" or (method == "ale" and categorical):
            xdata1 = xdata[0]
            if np.shape(xdata)[0] > 1:
                xdata2 = xdata[1]
                hist_data2 = hist_data[1]
        else:
            if feature2 != "":
                xdata1 = 0.5 * (xdata[0][1:] + xdata[0][:-1])
                xdata2 = 0.5 * (xdata[1][1:] + xdata[1][:-1])
            else:
                xdata1 = 0.5 * (xdata[1:] + xdata[:-1])

            if feature2 != "":
                hist_data2 = hist_data[1]

        results[f"{feature1}{feature2}__{estimator_name}__{method}"] = (y_shape, ydata)
        results[f"{feature1}__bin_values"] = ([f"n_bins__{feature1}"], xdata1)
        results[f"{feature1}"] = (["n_X"], hist_data1)
        if feature2 != "":
            results[f"{feature2[2:]}__bin_values"] = ([f"n_bins{feature2[2:]}"], xdata2)
            results[f"{feature2[2:]}"] = (["n_X"], hist_data2)

        if ice_X is not None:
            results["X_sampled"] = (["n_samples", "n_features"], ice_X)
            results["features"] = (["n_features"], ice_X.columns)
            
        return results

    def compute_individual_cond_expect(
        self, estimator_name, features, n_bins=30, subsample=1.0, n_bootstrap=1, 
    ):
        """
        Compute the Individual Conditional Expectations (see https://christophm.github.io/interpretable-ml-book/ice.html)
        
        Parameters
        -----------
        
        estimator_name : string
            Name of the estimator; used as a dict keys 
            
        features : string or 2-tuple of strings
            Feature name or pair of feature names to compute ICE for. 
            
        n_bins : integer
            Number of bins 
            
        subsample: float or integer (default=1.0 for no subsampling)
        
            if value is between 0-1, it is interpreted as fraction of total X to use 
            if value > 1, interpreted as the number of X to randomly sample 
            from the original dataset.
        
        n_bootstrap: integer (default=None for no bootstrapping)
            number of bootstrap resamples for computing confidence intervals. 
        
        
        Returns
        --------
        
        results : dict 
        
        """
        random_state = np.random.RandomState(self.random_seed)
        
        # Retrieve the estimator object from the estimators dict attribute
        estimator = self.estimators[estimator_name]

        # Check if features is a string
        if is_str(features):
            features = [features]

        # Check if feature is valid
        is_valid_feature(features, self.feature_names)

        if float(subsample) != 1.0:
            n_X = len(self.X)
            size = int(n_X * subsample) if subsample <= 1.0 else subsample
            idxs = random_state.choice(n_X, size=size, replace=False)
            X = self.X.iloc[idxs, :]
            X.reset_index(drop=True, inplace=True)
        else:
            X = self.X.copy()

        # Extract the values for the features
        feature_values = [X[f].to_numpy() for f in features]

        # Create a grid of values
        grid = [np.linspace(np.amin(f), np.amax(f), n_bins) for f in feature_values]

        if self.estimator_output == "probability":
            prediction_method = estimator.predict_proba
        elif self.estimator_output == "raw":
            prediction_method = estimator.predict

        ice_values = []
        for value_set in cartesian(grid):
            X_temp = X.copy()
            X_temp.loc[:, features[0]] = value_set[0]
            ice_values.append(prediction_method(X_temp.values))

        ice_values = np.array(ice_values).T
        if self.estimator_output == "probability":
            # Binary classification, shape is (2, n_points).
            # we output the effect of **positive** class
            # and convert to percentages
            ice_values = ice_values[1]

        # center the ICE plots
        ice_values -= np.mean(ice_values, axis=1).reshape(len(ice_values), 1)

        if len(features) > 1:
            ice_values = ice_values.reshape(n_bootstrap, n_bins, n_bins)
        else:
            features = features[0]

        results = self._store_results(
            method="ice",
            estimator_name=estimator_name,
            features=features,
            ydata=ice_values,
            xdata=grid,
            hist_data=feature_values,
            ice_X=X,
        )

        return results

    def compute_partial_dependence(
        self, estimator_name, features, n_bins=30, subsample=1.0, n_bootstrap=1
    ):

        """
        Calculate the centered partial dependence.

        # Friedman, J., 2001: Greedy function approximation: a gradient boosting machine.
        Annals of Statistics, 29 (5), 1189–1232.
        ##########################################################################
        # Partial dependence plots fix a value for one or more predictors
        # for X, passing these new data through a trained estimator,
        # and then averaging the resulting predictions. After repeating this process
        # for a range of values of X*, regions of non-zero slope indicates that
        # where the ML estimator is sensitive to X* (McGovern et al. 2019). Only disadvantage is
        # that PDP do not account for non-linear interactions between X and the other predictors.
        #########################################################################

        Parameters
        --------------
        estimator_name : string
            Name of the estimator; used as a dict keys 
            
        features : string or 2-tuple of strings
            Feature name or pair of feature names to compute ICE for. 
            
        n_bins : integer
            Number of bins 
            
        subsample: float or integer (default=1.0 for no subsampling)
        
            if value is between 0-1, it is interpreted as fraction of total X to use 
            if value > 1, interpreted as the number of X to randomly sample 
            from the original dataset.
        
        n_bootstrap: integer (default=None for no bootstrapping)
            number of bootstrap resamples for computing confidence intervals. 

        Returns
        ------------------
            results : dict 
        """
        # Retrieve the estimator object from the estimators dict attribute
        estimator = self.estimators[estimator_name]

        # Check if features is a string
        if is_str(features):
            features = [features]

        # Check if feature is valid
        is_valid_feature(features, self.feature_names)

        # Extract the values for the features
        full_feature_values = [self.X[f].to_numpy() for f in features]

        # Create a grid of values
        grid = [
            np.linspace(np.amin(f), np.amax(f), n_bins) for f in full_feature_values
        ]

        # get the bootstrap samples
        if n_bootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(
                self.X, subsample=subsample, n_bootstrap=n_bootstrap, seed=self.random_seed,
            )
        else:
            bootstrap_indices = [self.X.index.to_list()]

        if self.estimator_output == "probability":
            prediction_method = estimator.predict_proba
        elif self.estimator_output == "raw":
            prediction_method = estimator.predict

        pd_values = []

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_indices):
            # get samples
            X = self.X.iloc[idx, :].reset_index(drop=True)
            feature_values = [X[f].to_numpy() for f in features]
            averaged_predictions = []
            # for each value, set all indices to the value,
            # make prediction, store mean prediction
            for value_set in cartesian(grid):
                X_temp = X.copy()
                for i, feature in enumerate(features):
                    X_temp.loc[:, feature] = value_set[i]
                    predictions = prediction_method(X_temp.values)

                # average over samples
                averaged_predictions.append(np.mean(predictions, axis=0))

            averaged_predictions = np.array(averaged_predictions).T

            if self.estimator_output == "probability":
                # Binary classification, shape is (2, n_points).
                # we output the effect of **positive** class
                # and convert to percentages
                averaged_predictions = averaged_predictions[1]

            # Center the predictions
            averaged_predictions -= np.mean(averaged_predictions)

            pd_values.append(averaged_predictions)

        # Reshape the pd_values for higher-order effects
        pd_values = np.array(pd_values)
        if len(features) > 1:
            pd_values = pd_values.reshape([n_bootstrap] + [n_bins]*len(features))
        else:
            features = features[0]

        results = self._store_results(
            method="pd",
            estimator_name=estimator_name,
            features=features,
            ydata=pd_values,
            xdata=grid,
            hist_data=feature_values,
        )

        return results

    def compute_first_order_ale(
        self, estimator_name, feature, n_bins=30, subsample=1.0, n_bootstrap=1
    ):
        """
        Computes first-order ALE function on single continuous feature data.

        Script is largely the _first_order_ale_quant from
        https://github.com/blent-ai/ALEPython/ with small modifications.

        Parameters
        ----------
        estimator_name : string
            Name of the estimator; used as a dict keys 
            
        features : string or 2-tuple of strings
            Feature name or pair of feature names to compute ICE for. 
            
        n_bins : integer
            Number of bins 
            
        subsample: float or integer (default=1.0 for no subsampling)
        
            if value is between 0-1, it is interpreted as fraction of total X to use 
            if value > 1, interpreted as the number of X to randomly sample 
            from the original dataset.
        
        n_bootstrap: integer (default=None for no bootstrapping)
            number of bootstrap resamples for computing confidence intervals. 
            
            
        Returns
        ----------
            results : nested dictionary

        """
        estimator = self.estimators[estimator_name]
        # check to make sure feature is valid
        if feature not in self.feature_names:
            raise KeyError(f"Feature {feature} is not a valid feature.")

        # get the bootstrap samples
        if n_bootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(
                self.X, subsample=subsample, n_bootstrap=n_bootstrap, seed=self.random_seed,
            )
        else:
            bootstrap_indices = [self.X.index.to_list()]

        # Using the original, unaltered feature values
        # calculate the bin edges to be used in the bootstrapping.
        original_feature_values = self.X[feature].values

        if self.X[feature].dtype.name != "category":
            bin_edges = np.unique(
                np.percentile(
                    original_feature_values,
                    np.linspace(0, 100, n_bins + 1),
                    interpolation="lower",
                )
            )
            # Initialize an empty ale array
            ale = np.zeros((n_bootstrap, len(bin_edges) - 1))
        else:
            # Use the unique values for discrete data.
            bin_edges = np.unique(original_feature_values)

            # Initialize an empty ale array
            ale = np.zeros((n_bootstrap, len(bin_edges)))

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_indices):
            X = self.X.iloc[idx, :].reset_index(drop=True)

            # Find the ranges to calculate the local effects over
            # Using xdata ensures each bin gets the same number of X
            feature_values = X[feature].values

            # if right=True, then the smallest value in data is not included in a bin.
            # Thus, Define the bins the feature samples fall into. Shift and clip to ensure we are
            # getting the index of the left bin edge and the smallest sample retains its index
            # of 0.
            if self.X[feature].dtype.name != "category":
                indices = np.clip(
                    np.digitize(feature_values, bin_edges, right=True) - 1, 0, None
                )
            else:
                # indices for discrete data
                indices = np.digitize(feature_values, bin_edges) - 1

            # Assign the feature quantile values (based on its bin index) to two copied training datasets,
            # one for each bin edge. Then compute the difference between the corresponding predictions
            predictions = []
            for offset in range(2):
                X_temp = X.copy()
                X_temp[feature] = bin_edges[indices + offset]
                if self.estimator_output == "probability":
                    predictions.append(estimator.predict_proba(X_temp.values)[:, 1])
                elif self.estimator_output == "raw":
                    predictions.append(estimator.predict(X_temp.values))

            # The individual (local) effects.
            effects = predictions[1] - predictions[0]

            # Group the effects by their bin index
            index_groupby = pd.DataFrame(
                {"index": indices, "effects": effects}
            ).groupby("index")

            # Compute the mean local effect for each bin
            mean_effects = index_groupby.mean().to_numpy().flatten()

            # Accumulate (cumulative sum) the mean local effects.
            # Adding a 0 at the lower boundary of the first bin
            # for the interpolation step in the next step
            ale_uninterpolated = np.array([0, *np.cumsum(mean_effects)])

            # Interpolate the ale to the center of the bins.
            try:
                ale[k, :] = 0.5 * (ale_uninterpolated[1:] + ale_uninterpolated[:-1])
            except Exception as e:
                traceback.print_exc()
                raise ValueError(
                    f"""
                                 The value of n_bins ({n_bins}) is likely too 
                                 high relative to the sample size of the data. Either increase
                                 the data size (if using subsample) or use less bins. 
                                 """
                )

            # Center the ALE by substracting the bin-size weighted mean.
            ale[k, :] -= np.sum(ale[k, :] * index_groupby.size() / X.shape[0])

        results = self._store_results(
            method="ale",
            estimator_name=estimator_name,
            features=feature,
            ydata=ale,
            xdata=bin_edges,
            hist_data=[original_feature_values],
        )

        return results

    def _get_centers(self, x):
        return 0.5 * (x[1:] + x[:-1])

    def compute_second_order_ale(
        self, estimator_name, features, n_bins=30, subsample=1.0, n_bootstrap=1
    ):
        """
        Computes second-order ALE function on two continuous features data.

        Script is largely the _first_order_ale_quant from
        https://github.com/blent-ai/ALEPython/ with small modifications.

        Parameters
        ----------
        estimator_name : string
            Name of the estimator; used as a dict key 
            
        features : string or 2-tuple of strings
            Feature name or pair of feature names to compute ICE for. 
            
        n_bins : integer
            Number of bins 
            
        subsample: float or integer (default=1.0 for no subsampling)
        
            if value is between 0-1, it is interpreted as fraction of total X to use 
            if value > 1, interpreted as the number of X to randomly sample 
            from the original dataset.
        
        n_bootstrap: integer (default=None for no bootstrapping)
            number of bootstrap resamples for computing confidence intervals. 
            
            
        Returns
        ----------
            results : nested dictionary

        """
        estimator = self.estimators[estimator_name]

        # make sure there are two features...
        assert len(features) == 2, "Size of features must be equal to 2."

        # check to make sure both features are valid
        if features[0] not in self.feature_names:
            raise TypeError(f"Feature {features[0]} is not a valid feature")

        if features[1] not in self.feature_names:
            raise TypeError(f"Feature {features[1]} is not a valid feature")

        # create bins for computation for both features

        original_feature_values = [self.X[features[i]].values for i in range(2)]

        bin_edges = [
            np.unique(
                np.percentile(
                    v, np.linspace(0.0, 100.0, n_bins + 1), interpolation="lower"
                )
            )
            for v in original_feature_values
        ]

        # get the bootstrap samples
        if n_bootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(
                self.X, subsample=subsample, n_bootstrap=n_bootstrap, seed=self.random_seed,
            )
        else:
            bootstrap_indices = [self.X.index.to_list()]

        feature1_nbin_edges = len(bin_edges[0])
        feature2_nbin_edges = len(bin_edges[1])

        ale_set = []

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_indices):

            ale = np.ma.MaskedArray(
                np.zeros((feature1_nbin_edges, feature2_nbin_edges)),
                mask=np.ones((feature1_nbin_edges, feature2_nbin_edges)),
                fill_value=np.nan,
            )

            # get samples
            X = self.X.iloc[idx, :].reset_index(drop=True)

            # create bins for computation for both features
            feature_values = [X[features[i]].values for i in range(2)]

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
                X_temp = X.copy()
                for i in range(2):
                    X_temp[features[i]] = bin_edges[i][
                        indices_list[i] + shifts[i]
                    ]

                if self.estimator_output == "probability":
                    predictions[shifts] = estimator.predict_proba(X_temp.values)[
                        :, 1
                    ]
                elif self.estimator_output == "raw":
                    predictions[shifts] = estimator.predict(X_temp.values)

            # The individual (local) effects.
            effects = (predictions[(1, 1)] - predictions[(1, 0)]) - (
                predictions[(0, 1)] - predictions[(0, 0)]
            )

            # Group the effects by their indices along both axes.
            index_groupby = pd.DataFrame(
                {
                    "index_0": indices_list[0],
                    "index_1": indices_list[1],
                    "effects": effects,
                }
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
            samples_grid = np.zeros((feature1_nbin_edges - 1, feature2_nbin_edges - 1))
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

            # Replace the invalid bin values with the nearest valid ones.
            if np.any(missing_bin_mask):
                # Replace missing entries with their nearest neighbours.

                # Calculate the dense location matrices (for both features) of all bin centres.
                centers_list = np.meshgrid(
                    *(self._get_centers(quantiles) for quantiles in bin_edges),
                    indexing="ij",
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
                    valid_indices[nearest_points]
                    for valid_indices in valid_indices_list
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
                first_order = (
                    ale[(slice(1, None), ...)[flip]] - ale[(slice(-1), ...)[flip]]
                )
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
                first_order = np.array([0, *np.cumsum(first_order)]).reshape(
                    (-1, 1)[flip]
                )

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
            ale -= np.sum(samples_grid * ale) / len(X)
            
            ale_set.append(ale)

        ###ale_set_ds = xarray.DataArray(ale_set).to_masked_array()

        results = self._store_results(
            method="ale",
            estimator_name=estimator_name,
            features=features,
            ydata=ale_set,
            xdata=bin_edges,
            hist_data=original_feature_values,
        )

        return results

    def compute_first_order_ale_cat(
        self,
        estimator_name,
        feature,
        subsample=1.0,
        n_bootstrap=1,
        feature_encoder=None,
    ):
        """
        Computes first-order ALE function on a single categorical feature.

        Script is largely from aleplot_1D_categorical from
        PyALE with small modifications (https://github.com/DanaJomar/PyALE).

        Parameters
        ----------
        estimator_name : string
            Name of the estimator; used as a dict keys 
            
        features : string or 2-tuple of strings
            Feature name or pair of feature names to compute ICE for. 
            
        n_bins : integer
            Number of bins 
            
        subsample: float or integer (default=1.0 for no subsampling)
        
            if value is between 0-1, it is interpreted as fraction of total X to use 
            if value > 1, interpreted as the number of X to randomly sample 
            from the original dataset.
        
        n_bootstrap: integer (default=None for no bootstrapping)
            number of bootstrap resamples for computing confidence intervals. 
            
        Returns
        ----------
            results : nested dictionary

        """
        pd.options.mode.chained_assignment = "raise"

        if feature_encoder is None:

            def feature_encoder_func(data):
                return data

            feature_encoder = feature_encoder_func

        estimator = self.estimators[estimator_name]
        # check to make sure feature is valid
        if feature not in self.feature_names:
            raise Exception(f"Feature {feature} is not a valid feature")

        # get the bootstrap samples
        if n_bootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(
                self.X, subsample=subsample, n_bootstrap=n_bootstrap, seed=self.random_seed,
            )
        else:
            bootstrap_indices = [self.X.index.to_list()]

        original_feature_values = [feature_encoder(self.X[feature].values)]
        xdata = np.array([np.unique(original_feature_values)])
        xdata.sort()

        # Initialize an empty ale array
        ale = []

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_indices):
            X = self.X.iloc[idx, :].reset_index(drop=True)

            if (X[feature].dtype.name != "category") or (
                not X[feature].cat.ordered
            ):
                X[feature] = X[feature].astype(str)
                groups_order = order_groups(X, feature)
                groups = groups_order.index.values
                X[feature] = X[feature].astype(
                    pd.api.types.CategoricalDtype(categories=groups, ordered=True)
                )

            groups = X[feature].unique()
            groups = groups.sort_values()
            feature_codes = X[feature].cat.codes
            groups_counts = X.groupby(feature).size()
            groups_props = groups_counts / sum(groups_counts)

            K = len(groups)

            # create copies of the dataframe
            X_plus = X.copy()
            X_neg = X.copy()
            # all groups except last one
            last_group = groups[K - 1]
            ind_plus = X[feature] != last_group
            # all groups except first one
            first_group = groups[0]
            ind_neg = X[feature] != first_group
            # replace once with one level up
            X_plus.loc[ind_plus, feature] = groups[feature_codes[ind_plus] + 1]
            # replace once with one level down
            X_neg.loc[ind_neg, feature] = groups[feature_codes[ind_neg] - 1]
            try:
                # predict with original and with the replaced values
                # encode the categorical feature
                X_coded = pd.concat(
                    [
                        X.drop(feature, axis=1),
                        feature_encoder(X[[feature]]),
                    ],
                    axis=1,
                )
                X_coded = np.array(X_coded[self.feature_names], dtype=float)

                # predict
                if self.estimator_output == "probability":
                    y_hat = estimator.predict_proba(X_coded)[:,1]
                else:
                    y_hat = estimator.predict(X_coded)

                # encode the categorical feature
                X_plus_coded = pd.concat(
                    [
                        X_plus.drop(feature, axis=1),
                        feature_encoder(X_plus[[feature]]),
                    ],
                    axis=1,
                )
  
                X_plus_coded = np.array(X_plus_coded[ind_plus][self.feature_names], dtype=float)
                
                # predict
                if self.estimator_output == "probability":
                    y_hat_plus = estimator.predict_proba(X_plus_coded)[:, 1]
                else:
                    y_hat_plus = estimator.predict(X_plus_coded)

                # encode the categorical feature
                X_neg_coded = pd.concat(
                    [
                        X_neg.drop(feature, axis=1),
                        feature_encoder(X_neg[[feature]]),
                    ],
                    axis=1,
                )
                
                X_neg_coded = np.array(X_neg_coded[ind_neg][self.feature_names], dtype=float)
                
                # predict
                if self.estimator_output == "probability":
                    y_hat_neg = estimator.predict_proba(X_neg_coded)[:, 1]
                else:
                    y_hat_neg = estimator.predict(X_neg_coded)

            except Exception as ex:
                raise Exception(
                    """There seems to be a problem when predicting with the estimator.
                    Please check the following: 
                    - Your estimator is fitted.
                        - The list of predictors contains the names of all the features"""
                    """ used for training the estimator.
                        - The encoding function takes the raw feature and returns the"""
                    """ right columns encoding it, including the case of a missing category.
                    """
                )

            # compute prediction difference
            Delta_plus = y_hat_plus - y_hat[ind_plus]
            Delta_neg = y_hat[ind_neg] - y_hat_neg

            # compute the mean of the difference per group
            delta_df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "eff": Delta_plus,
                            feature: groups[feature_codes[ind_plus] + 1],
                        }
                    ),
                    pd.DataFrame(
                        {"eff": Delta_neg, feature: groups[feature_codes[ind_neg]]}
                    ),
                ]
            )
                       
            res_df = delta_df.groupby([feature]).mean()
            res_df.loc[:, "ale"] = res_df.loc[:, "eff"].cumsum()

            res_df.loc[groups[0]] = 0
            # sort the index (which is at this point an ordered categorical) as a safety measure
            res_df = res_df.sort_index()

            # Subtract the mean value to get the centered value.
            ale_temp = res_df["ale"] - sum(res_df["ale"] * groups_props)
            ale.append(ale_temp)

        ale = np.array(ale, dtype=float)

        results = self._store_results(
            method="ale",
            estimator_name=estimator_name,
            features=feature,
            ydata=ale,
            xdata=xdata,
            hist_data=original_feature_values,
            categorical=True,
        )

        return results

    def compute_scalar_interaction_stats(
        self,
        method,
        data,
        estimator_names,
        data_2d=None, 
        features=None,
        **kwargs,
    ):
        """
        Wrapper function for computing the interaction strength statistic or the Friedman
        H-statistic (see below). Will perform calculation in parallel for multiple estimators.
        
        Parameters
        ------------
        
        method : 'ias' or 'hstat'
            Whether to compute the interaction strength (ias) or the 
            Friedman H-statistics (hstat)
        
        data : xarray.Dataset
        data_2d : xarray.Dataset
        
        features : string
      
        estimator_name : string
            Name of the estimator; used as a dict keys 
   
        n_bins : integer
            Number of bins 
            
        subsample: float or integer (default=1.0 for no subsampling)
        
            if value is between 0-1, it is interpreted as fraction of total X to use 
            if value > 1, interpreted as the number of X to randomly sample 
            from the original dataset.
        
        n_bootstrap: integer (default=None for no bootstrapping)
            number of bootstrap resamples for computing confidence intervals. 
            
        n_jobs : integer or float

        Returns
        --------
            results : xarray.Dataset
        
        """
        if method == "ias":
            func = self.compute_interaction_strength
            args_iterator = to_iterator(estimator_names,)
            total=len(estimator_names)
        elif method == "hstat":
            func = self.friedman_h_statistic
            args_iterator = to_iterator(estimator_names, features,)
            total=len(estimator_names)*len(features)

        self.data = data
        self.data_2d = data_2d
        n_jobs = len(estimator_names)
        results = run_parallel(
            func=func, args_iterator=args_iterator, kwargs=kwargs, nprocs_to_use=n_jobs, 
            total=total
        )
        
        results = merge_dict(results)

        if method == 'hstat':
            final_results={}
            feature_pairs = np.array([f'{f[0]}__{f[1]}' for f in features])
            for estimator_name in estimator_names:
                values = np.array([results[f'{f}__{estimator_name}_hstat'] for f in feature_pairs])
                idx = np.argsort(np.mean(values, axis=1))[::-1]
                feature_names_sorted = np.array(feature_pairs)[idx]
                values_sorted = values[idx, :]

                final_results[f"hstat_rankings__{estimator_name}"] = (
                    [f"n_vars_perm_based_interactions"],
                    feature_names_sorted,
                    )
                final_results[f"hstat_scores__{estimator_name}"] = (
                    [f"n_vars_perm_based_interactions", "n_bootstrap"],
                    values_sorted,
                )
            results = to_xarray(final_results)
        else:
            results = to_xarray(results)
            
        return results
    
    def friedman_h_statistic(self, estimator_name, features, **kwargs):
        """
        Compute the H-statistic for two-way interactions between two features.

        Parameters
        -------------
        estimator_name : str
        feature_tuple : 2-tuple of strs
        n_bins : int
        subsample : integer or float
        n_bootstrap : integer 

        Returns
        --------
        
        results : dictionary 
        """
        feature1, feature2 = features
        feature1_pd = self.data[f"{feature1}__{estimator_name}__pd"].values
        feature2_pd = self.data[f"{feature2}__{estimator_name}__pd"].values

        combined_pd = self.data_2d[
            f"{feature1}__{feature2}__{estimator_name}__pd"
        ].values

        # Calculate the H-statistics
        pd_decomposed = feature1_pd[:, :, np.newaxis] + feature2_pd[:, np.newaxis, :]

        numer = (combined_pd - pd_decomposed) ** 2
        denom = (combined_pd) ** 2
        H_squared = np.sum(np.sum(numer, axis=2), axis=1) / np.sum(np.sum(denom, axis=2), axis=1)

        results = {f"{feature1}__{feature2}__{estimator_name}_hstat" : np.sqrt(H_squared)}
        
        return results

    def compute_interaction_strength(self, estimator_name, **kwargs):
        """
        Compute the interaction strenth of a ML estimator (based on IAS from
        Quantifying estimator Complexity via Functional
        Decomposition for Better Post-Hoc Interpretability).

        Parameters
        --------------------
        estimator_names : list of strings

        Returns
        ---------
        
        ias : dict 

        """
        subsample = kwargs.get("subsample", 1.0)
        n_bootstrap = kwargs.get("n_bootstrap", 1) 
        estimator_output = kwargs.get("estimator_output", 'raw')
        
        estimator = self.estimators[estimator_name]
        feature_names = list(self.X.columns)
        data = self.data

        # Get the interpolated ALE curves
        main_effect_funcs = {}
        for f in feature_names:
            ale = np.mean(data[f"{f}__{estimator_name}__ale"].values, axis=0)
            x = data[f"{f}__bin_values"].values
            main_effect_funcs[f] = interp1d(
                x, ale, fill_value="extrapolate", kind="linear",
            )

        # get the bootstrap samples
        if n_bootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(
                self.X, subsample=subsample, n_bootstrap=n_bootstrap
            )
        else:
            bootstrap_indices = [self.X.index.to_list()]

        ias = []
        for k, idx in enumerate(bootstrap_indices):
            X = self.X.iloc[idx, :].values
            if self.estimator_output == "probability":
                predictions = estimator.predict_proba(X)[:, 1]
            else:
                predictions = estimator.predict(X)

            # Get the average estimator prediction
            avg_prediction = np.mean(predictions)

            # Get the ALE value for each feature per X
            main_effects = np.array(
                [
                   main_effect_funcs[f](np.array(X[:,i], dtype=np.float64))
                    for i, f in enumerate(feature_names)
                ]
            )

            # Sum the ALE values per X and add on the average value
            main_effects = np.sum(main_effects.T, axis=1) + avg_prediction

            num = np.sum((predictions - main_effects) ** 2)
            denom = np.sum((predictions - avg_prediction) ** 2)

            # Compute the interaction strength
            ias.append(num / denom)

        return {f'{estimator_name}_ias': (['n_bootstrap'], np.array(ias))}

    def compute_ale_variance(
        self,
        data,
        estimator_names,
        features=None,
        **kwargs
    ):
        """
        Compute the standard deviation of the ALE values
        for each feature and rank then for predictor importance.

        Parameters
        ----------
        data : xarray.Dataset
        estimator_names : list of strings
        features : str
        ale_subsample : float or integer

        Returns
        ---------

        results_ds : xarray.Dataset

        """
        feature_names = list([f for f in data.data_vars if "__" not in f])
        feature_names.sort()
        
        features, cat_features = determine_feature_dtype(self.X, feature_names)
        
        def _std(values, f, cat_features):
            """
            Using a different formula for computing standard deviation for 
            categorical features. 
            """
            if f in cat_features:
                return 0.25*(np.max(values, axis=1) - np.min(values, axis=1))
            else:
                return np.std(values, ddof=1, axis=1) 
        
        results = {}
        for estimator_name in estimator_names:
            # Compute the std over the bin axis [shape = (n_features, n_bootstrap)]
            # Input shape : (n_bootstrap, n_bins) 
            ale_std = np.array(
                [
                    _std(data[f"{f}__{estimator_name}__ale"].values, f, cat_features)
                    for f in feature_names
                ]
            )

            # Average over the bootstrap indices 
            idx = np.argsort(np.mean(ale_std, axis=1))[::-1]
    
            feature_names_sorted = np.array(feature_names)[idx]
            ale_std_sorted = ale_std[idx,:]

            results[f"ale_variance_rankings__{estimator_name}"] = (
                [f"n_vars_ale_variance"],
                feature_names_sorted,
            )
            results[f"ale_variance_scores__{estimator_name}"] = (
                [f"n_vars_ale_variance", "n_bootstrap"],
                ale_std_sorted,
            )

        results_ds = to_xarray(results)

        return results_ds

    def compute_interaction_rankings(self,data,estimator_names,features,**kwargs,):
        """
        Compute the variable interaction rankings from Greenwell et al. 2018, but
        using the purely second-order ALE rather than the second-order PD.

        For a given second-order ALE,
         1. Compute the standard deviation over one axis and then take the std of those values
         2. Compute the standard deviation over another axis and then take the std of those values
         3. Average 1 and 2 together.

        Sort by magnitudes given in (3). Larger values (relative to other values) are
        possibly indicative of  stronger feature interactions.


        Parameters
        --------------------
        data : xarray.Dataset
        estimator_names : list of strings
        features : str
        ale_subsample : float or integer

        Returns
        --------
        results_ds : xarray.Dataset
        """
        results = {}
        for estimator_name in estimator_names:
            
            interaction_effects = []
            for f in features:
                data_temp = data[f"{f[0]}__{f[1]}__{estimator_name}__ale"].values
                std_feature0 = np.std(np.std(data_temp, ddof=1, axis=1), ddof=1, axis=1)
                std_feature1 = np.std(np.std(data_temp, ddof=1, axis=2), ddof=1, axis=1)
                interaction_effects.append(0.5 * (std_feature0 + std_feature1))

            interaction_effects = np.array(interaction_effects)
            # Average over the bootstrap indices
            idx = np.argsort(np.mean(interaction_effects, axis=1))[::-1]

            feature_names_sorted = np.array(features)[idx]
            interaction_effects_sorted = interaction_effects[idx, :]
            feature_names_sorted = [f"{f[0]}__{f[1]}" for f in feature_names_sorted]

            results[f"ale_variance_interactions_rankings__{estimator_name}"] = (
                [f"n_vars_ale_variance_interactions"],
                feature_names_sorted,
            )
            results[f"ale_variance_interactions_scores__{estimator_name}"] = (
                [f"n_vars_ale_variance_interactions", "n_bootstrap"],
                interaction_effects_sorted,
            )

        results_ds = to_xarray(results)

        return results_ds

    def number_of_features_used(self, estimator, X):
        """
        Compute the number of features (NF) from Molnar et al. 2019
        """
        original_predictions = estimator.predict_proba(X)[:, 1]
        features = list(X.columns)
        for f in features:
            X_temp = X.copy()
            X_temp.loc[:, f] = np.random.permutation(X_temp[f].values)

            new_predictions = estimator.predict_proba(X_temp)[:, 1]
            change = np.absolute(new_predictions - original_predictions)
   
    
    def compute_interaction_performance_based(self, estimator, 
                                           X, y, 
                                           X_permuted, 
                                           features, 
                                           evaluation_fn,
                                           estimator_output, 
                                           verbose=False):
        """
        Compute the performance-based feature interactions from Oh (2019).
    
        Err(F_i) : Error when feature F_i is permuted as compared 
               against the original estimator performance.  
               
        Err({F_i, F_j}) : Error when features F_i & F_j are permuted as 
                       compared against the original estimator performance. 
    
        Interaction = Err(F_i) + Err(F_j) - Err({F_i, F_j}), for classification 
        Interaction = Err({F_i, F_j}) - (Err(F_i) + Err(F_j)), for regression
    
        Interaction(F_i, F_j) = 0 -> no interaction between F_i & F_j 
        Interaction(F_i, F_j) > 0 -> positive interaction between F_i & F_j 
        Interaction(F_i, F_j) < 0 -> negative interaction between F_i & F_j 
    
        Negative interaction implies that the connection between Fi and Fj reduces the prediction performance, 
        whereas positive interaction leads to an increase in the prediction performance. 
        In other words, positive or negative interactions decrease or increase the 
        prediction error, respectively.
    
        Parameters
        -----------
        estimator, 
        X : pandas.DataFrame  of shape (n_X, n_features)
        Data to compute importance over. 
        y : numpy.array
        y values
        features: list of 2-tuples 
        evaluation_fn : callable 
        random_state : integer
    
        Returns
        --------
        
        err : numpy.array
        """
        available_scores = ["auc", "auprc", "bss", "mse", "norm_aupdc"]

        if isinstance(evaluation_fn, str):
            evaluation_fn = evaluation_fn.lower()
            is_str=True

        if isinstance(evaluation_fn,str):    
            if evaluation_fn == "auc":
                evaluation_fn = roc_auc_score
                scoring_strategy = "argmin_of_mean"
            elif evaluation_fn == "auprc":
                evaluation_fn = average_precision_score
                scoring_strategy = "argmin_of_mean"
            elif evaluation_fn == "norm_aupdc":
                evaluation_fn = norm_aupdc
                scoring_strategy = "argmin_of_mean"
            elif evaluation_fn == "bss":
                evaluation_fn = brier_skill_score
                scoring_strategy = "argmin_of_mean"
            elif evaluation_fn == "mse":
                evaluation_fn = mean_squared_error
                scoring_strategy = "argmax_of_mean"
            else:
                raise ValueError(
                    f"evaluation_fn is not set! Available options are {available_scores}"
                )
                
        if estimator_output == 'probability':
            #Classification
            predictions = estimator.predict_proba(X)[:,1]
        else:
            #Regression
            predictions = estimator.predict(X)
        
        original_score = evaluation_fn(y, predictions)
        
        err=0
        X_permuted_both = X.copy()
        # Compute the change in estimator performance for the two features separately 
        for feature in features:
            X_temp = X.copy()
            X_temp.loc[:,feature] = X_permuted.loc[:,feature]
            X_permuted_both.loc[:,feature] = X_permuted.loc[:,feature]

            # Get the predictions for the dataset with a permuted feature. 
            if estimator_output == 'probability':
                #Classification
                predictions = estimator.predict_proba(X_temp)[:,1]
            else:
                #Regression
                predictions = estimator.predict(X_temp)

            # Compute the permuted score 
            permuted_score = evaluation_fn(y, predictions)
            
            if estimator_output == 'probability':
                #Classification
                err_i = original_score - permuted_score
            else:
                #Regression
                err_i = permuted_score - original_score

            if verbose:
                permuted_features = check_is_permuted(X, X_temp)
                print(f'Reduced performance by {feature} : {err_i}')
                print(f'Permuted Features: {permuted_features}')
                
            err+=err_i
        
        # Compute the change in estimator performance with both features permuted. 
        if estimator_output == 'probability':
            predictions_both = estimator.predict_proba(X_permuted_both)[:,1]
        else:
            predictions_both = estimator.predict(X_permuted_both)
        
        # Compute the permuted score 
        permuted_score_both = evaluation_fn(y, predictions_both)
        
        if estimator_output == 'probability':
            err_both = original_score - permuted_score_both
        else:
            err_both = permuted_score_both - original_score

        if verbose:
            permuted_features = check_is_permuted(X, X_permuted_both)
            print(f'Reduced performance by {features} : {err_both}')    
            print(f'Permuted Features: {permuted_features}')
            
        # Combine for the feature interaction between the two features 
        if estimator_output == 'probability':
            err-=err_both
        else:
            err = err_both - err  
    
        return err
    
    def compute_interaction_rankings_performance_based(self, estimator_names, 
                                           features, 
                                           evaluation_fn,
                                           estimator_output, 
                                           subsample=1.0,
                                           n_bootstrap=1,
                                           n_jobs=1,                     
                                           verbose=False):
        """
        Wrapper function for performance_based_feature_interactions
        """
        unique_features = list(set([item for t in features for item in t]))
        n_feature_pairs = len(features)
        X_permuted = self.X.copy()
        # Permute all features up front to save on computation cost
        for f in unique_features:
            X_permuted.loc[:,f] = np.random.permutation(self.X.loc[:,f])
            
        args_iterator = to_iterator(
                estimator_names,
                features,
                [X_permuted],
                [evaluation_fn],
                [estimator_output], 
                [subsample],
                [n_bootstrap],
                [verbose]
            )

        results = run_parallel(
                func=self._feature_interaction_worker, args_iterator=args_iterator, kwargs={}, nprocs_to_use=n_jobs
            )
        
        results = merge_dict(results)
        
        final_results={}
        feature_pairs = np.array([f'{f[0]}__{f[1]}' for f in features])
        for estimator_name in estimator_names:
            values = np.array([results[f'{f}_interaction__{estimator_name}'] for f in feature_pairs])
            idx = np.argsort(np.absolute(np.mean(values, axis=1)))[::-1]
            feature_names_sorted = np.array(feature_pairs)[idx]
            values_sorted = values[idx, :]

            final_results[f"perm_based_interactions_rankings__{estimator_name}"] = (
                [f"n_vars_perm_based_interactions"],
                feature_names_sorted,
            )
            final_results[f"perm_based_interactions_scores__{estimator_name}"] = (
                [f"n_vars_perm_based_interactions", "n_bootstrap"],
                values_sorted,
            )

        results_ds = to_xarray(final_results)
        
        return results_ds
    
    def _feature_interaction_worker(self, estimator_name, features, 
                                    X_permuted, evaluation_fn, 
                                    estimator_output, 
                                    subsample=1.0, n_bootstrap=1, verbose=False):
        """
        Internal worker function for parallel computations. 
        """
        estimator = self.estimators[estimator_name]
            
        # get the bootstrap samples
        if n_bootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(
                        self.X, subsample=subsample, n_bootstrap=n_bootstrap
                    )
        else:
            bootstrap_indices = [np.arange(self.X.shape[0])]
            
        results={}
        err_set = np.zeros((n_bootstrap,))
        for k, idx in enumerate(bootstrap_indices):
            # get samples
            X = self.X.iloc[idx, :].reset_index(drop=True)
            y= self.y[idx]

            err = self.compute_interaction_performance_based(estimator, 
                                          X, y, X_permuted, 
                                          features, 
                                          evaluation_fn,
                                          estimator_output,                        
                                         verbose=verbose)
            err_set[k]=err

        results[f'{features[0]}__{features[1]}_interaction__{estimator_name}'] = err_set

        return results       


    def compute_main_effect_complexity(self, estimator_name, ale_ds,  
                            features, post_process=False, max_segments=10, approx_error=0.05, 
                                      debug=False):
        """
        Compute the Main Effect Complexity (MEC; Molnar et al. 2019). 
        MEC is the number of linear segements required to approximate 
        the first-order ALE curves; averaged over all features. 
        The MEC is weighted-averged by the variance. Higher values indicate
        a more complex estimator (less interpretable). 
        
        References 
        -----------
            Molnar, C., G. Casalicchio, and B. Bischl, 2019: Quantifying estimator Complexity via 
            Functional Decomposition for Better Post-Hoc Interpretability. ArXiv.
        
        
        Parameters
        ----------------
        
        ale_ds : xarray.Dataset
            
            The results xarray.Dataset from computing 1D ALE using .calc_ale(). 
            Must be computed for all features in X. 

        estimator_names : string, list of strings
        
            If using multiple estimators, you can pass a single (or subset of) estimator name(s) 
            to compute the MEC for. 
            
        max_segments : integer
            
            Maximum number of linear segments used to approximate the main/first-order 
            effect of a feature. default is 10. Used to limit the computational expense. 
            
        approx_error : float
        
            The accepted error of the R squared between the piece-wise linear function 
            and the true ALE curve. If the R square is within the approx_error, then 
            no additional segments are added. default is 0.05

        Returns
        ----------
            mec_avg, float
                Average Main Effect Complexity 
            best_break_dict, dict
                For each feature, the list of "optimal" breakpoints
                Used to plot and verify the code is running correctly. 
        """
        mec = np.zeros((len(features)))
        var = np.zeros((len(features)))
    
        # Based on line 6 from Algorithm 2: Main Effect Complexity (MEC) in Molnar et al. 2019 
        # we ignore categorical features (set the slopes to zero). 
        categorical_features = [self.X[f].dtype.name == "category" for f in self.feature_names]
        best_breaks_dict = {}
        best_g={}
        for j, f in enumerate(features):
            ale = np.mean(ale_ds[f'{f}__{estimator_name}__ale'].values,axis=0)
            x = ale_ds[f'{f}__bin_values'].values.reshape(-1, 1)
    
            b_max=len(x)
            # Approximate ALE with linear estimator
            lr = LinearRegression()
            lr.fit(x.reshape(-1, 1), ale)
            g = lr.predict(x)
    
            best_g[f] = g
    
            coef = lr.coef_[0]
            best_score = r2_score(g, ale)

            # Increase num. of segements until approximation is good enough. 
            k = 1
            best_breaks=[]
            while k < max_segments and (r2_score(g, ale) < (1-approx_error)):
                # Find intervals Z_k through exhaustive search along ALE curve breakpoints
                for b in range(1, b_max-1):
                    if b not in best_breaks:
                        temp_breaks=copy(best_breaks)
                        temp_breaks.append(b)
                        temp_breaks.sort()

                        idxs_set = [range(0,temp_breaks[0]+1)] + \
                               [range(temp_breaks[i], temp_breaks[i+1]+1) for i in range(len(temp_breaks)-1)] +\
                               [range(temp_breaks[-1], b_max)]

                        estimator_set = [LinearRegression(normalize=True) for _ in range(len(idxs_set))]
                        
                        estimator_fit_set_tmp = [estimator_set[i].fit(x[idxs,:], ale[idxs])
                                             for i, idxs in enumerate(idxs_set)
                            ]

                        # For categorical features, set the slope to zero (but keep the intercept).
                        if f in categorical_features:
                            estimator_fit_set=[]
                            for e in estimator_fit_set_tmp:
                                e.coef_ = np.array([0.])
                                estimator_fit_set.append(e) 
                        else:
                            estimator_fit_set = [e for e in estimator_fit_set_tmp]
                  
                     
                        predict_set = [
                            estimator_fit_set[i].predict(x[idxs,:])
                                for i, idxs in enumerate(idxs_set)
                        ]
                        
                        # Combine the predictions from the different breaks into 
                        # a single piece-wise function (line 7 for Molnar et al. 2019)
                        for i, idxs in enumerate(idxs_set):
                            g[idxs] = predict_set[i]

                        current_score = r2_score(g, ale)
                        if current_score > best_score:
                            best_score = current_score
                            best_break = b 
                            # Is approx good enough? Then stop iterating
                            if best_score > 1-approx_error:
                                best_g[f] = g
                                break
                           
                best_breaks.append(best_break)
                k+=1

            # Sum of non-zero coefficients minus first intercept 
            best_breaks_dict[f] = best_breaks
            mec[j] = k #+ add
            var[j] = np.var(ale)
    
            print(f, k, f'{np.var(ale):.06f}')
    
        mec_avg = np.average(mec, weights=var) 
        
 
        if debug:
            return mec_avg, best_breaks_dict, best_g
        else:
            return mec_avg, best_breaks_dict

        

    def scorer(self, estimator, X, y, evaluation_fn):
        prediction = estimator.predict_proba(X)[:,1]
        return evaluation_fn(y, prediction)

    def _compute_grouped_importance(self, X, y, evaluation_fn, group, feature_subset, estimator, only, all_permuted_score):
        scores={group : []}
        X_permuted = X.copy()
        for rs in self.random_states:
            inds = rs.permutation(len(X))
            # Jointly permute all features expect those in the feature_subset
            
            if only:
                # For group only version, jointly permute all features except those in the feature subset.
                X_permuted = np.array([X[:, i] if i in feature_subset else X[inds,i] for i in range(X.shape[1])]).T
            else:
                # Else, only jointly permute the features in the feature subset.
                X_permuted = np.array([X[inds, i] if i in feature_subset else X[:,i] for i in range(X.shape[1])]).T
            
            group_permute_score = self.scorer(estimator, X_permuted, y, evaluation_fn)
            
            if only:
                imp = group_permute_score - all_permuted_score
            else:
                imp = all_permuted_score - group_permute_score
            
            scores[group].append(imp)
            
        return scores

    def grouped_feature_importance(self, 
                                   evaluation_fn,
                                   perm_method, 
                               n_permute=1, 
                               groups=None, 
                               sample_size=100, 
                               subsample=1.0,
                               n_jobs=1, 
                               clustering_kwargs={'n_clusters':10},
                              ):
        """
        The group only permutation feature importance (GOPFI) from Au et al. 2021 
        (see their equations 10,11). This function has a built-in method for clustering 
        features using the sklearn.cluster.FeatureAgglomeration. It also has the ability to 
        compute the results over multiple permutations to improve the feature importance 
        estimate (and provide uncertainty). 
    
        Description of the parameters provided in InterpretToolkit. 
    
        """
        parallel = Parallel(n_jobs=n_jobs)
        only = True if perm_method == 'grouped_only' else False
        
        if isinstance(evaluation_fn, str):
            evaluation_fn, scoring_strategy = self._to_scorer(evaluation_fn)
    
        self.random_states = bootstrap_generator(n_bootstrap=n_permute)
    
        # Get the feature names from the dataframe.
        feature_names = np.array(self.X.columns)
    
        # Convert to numpy array
        X = self.X.values
    
        subsample = subsample if subsample > 1 else int(subsample*X.shape[0])
        inds = self.random_states[-1].choice(len(X), size=subsample, replace=False)
    
        # Subsample the examples for computational efficiency. 
        X = X[inds]
        y = self.y[inds]

        if groups is None:
            # Feature clustering is based on a subset of examples for computational efficiency. 
            sample_size = sample_size if sample_size > 1 else int(sample_size*X.shape[0])
            inds = np.random.choice(len(X), size=sample_size, replace=False)
            agglo = cluster.FeatureAgglomeration(**clustering_kwargs)
            agglo.fit(X[inds,:])

            groups = {f'group {i}' : np.where(agglo.labels_== i)[0] for i in range(np.max(agglo.labels_)+1) }
            names = {f'group {i}' : feature_names[agglo.labels_==i] for i in range(np.max(agglo.labels_)+1) }
        else:
            contains_str = isinstance(list(groups.values())[0][0], str)
            if contains_str:
                names = copy(groups)
                # It is the feature names and thus need to be converted to indices
                for key, items in groups.items():
                    N=np.where(np.isin(feature_names, items))[0]
                    groups[key] = N
            else:
                names = {key : feature_names[inds] for key, inds in groups.items() }

        X_permuted = X.copy()
        if only:
            # for the group only version, we jointly permute all features and then
            # determine the original score
            inds = np.random.permutation(len(X))
            X_permuted = np.array([X[inds, i] for i in range(X.shape[1])]).T

            
        results = []
        for estimator_name, estimator in self.estimators.items():    
            # Score after jointly permuting all features. 
            all_permuted_score = self.scorer(estimator, X_permuted, y, evaluation_fn)

            scores = parallel(delayed(self._compute_grouped_importance)(X,y,evaluation_fn, 
                                                                    group, feature_subset, estimator, only, all_permuted_score) 
                          for group, feature_subset in groups.items())
            scores = merge_dict(scores)

            group_names = list(groups.keys())
            importances = np.array([scores[g] for g in group_names])
    
            group_rank = to_pymint_importance(importances, estimator_name=estimator_name, 
                                      feature_names=group_names, method=perm_method)
    
            results.append(group_rank)
    
        results = xr.merge(results, combine_attrs="override", compat="override")
 
        return results, names

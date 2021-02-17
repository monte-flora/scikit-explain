import numpy as np
import pandas as pd
from math import sqrt
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import itertools
from functools import reduce
from operator import add
import traceback

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    mean_squared_error,
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
    determine_feature_dtype
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

    def __init__(
        self,
        models,
        model_names,
        examples,
        targets=None,
        model_output="probability",
        feature_names=None,
        checked_attributes=False,
    ):
        # These functions come from the inherited Attributes class
        if not checked_attributes:
            self.set_model_attribute(models, model_names)
            self.set_examples_attribute(examples, feature_names)
            self.set_target_attribute(targets)
        else:
            self.models = models
            self.model_names = model_names
            self.examples = examples
            self.targets = targets
            self.feature_names = list(examples.columns)

        self.model_output = model_output

    def permutation_importance(
        self,
        n_vars=5,
        evaluation_fn="auprc",
        subsample=1.0,
        n_jobs=1,
        n_bootstrap=None,
        scoring_strategy=None,
        perm_method='marginal',
        verbose=False,
        random_state=None,
    ):

        """
        Performs single-pass and/or multi-pass permutation importance using the PermutationImportance
        package.

        See calc_permutation_importance in IntepretToolkit for documentation.

        """
        available_scores = ["auc", "auprc", "bss", "mse", "norm_aupdc"]

        if isinstance(evaluation_fn, str):
            evaluation_fn=evaluation_fn.lower()
        
        if not isinstance(evaluation_fn, str) and scoring_strategy is None:
            raise ValueError(
                """ 
                The scoring_strategy argument is None! If you are using a user-define evaluation_fn 
                then scoring_strategy must be set! If a metric is positively-oriented (a higher value is better), 
                then set scoring_strategy = "argmin_of_mean" and if is negatively-oriented-
                (a lower value is better), then set scoring_strategy = "argmax_of_mean"
                """
            )

        if evaluation_fn.lower() == "auc":
            evaluation_fn = roc_auc_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == "auprc":
            evaluation_fn = average_precision_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == "norm_aupdc":
            evaluation_fn = norm_aupdc
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == "bss":
            evaluation_fn = brier_skill_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == "mse":
            evaluation_fn = mean_squared_error
            scoring_strategy = "argmax_of_mean"
        else:
            raise ValueError(
                f"evaluation_fn is not set! Available options are {available_scores}"
            )

        if subsample != 1.0 and n_bootstrap is None:
            n_bootstrap = 1
            
        targets = pd.DataFrame(data=self.targets, columns=["Test"])

        pi_dict = {}

        # loop over each model
        for model_name, model in self.models.items():

            ### print(f"Processing {model_name}...")

            pi_result = sklearn_permutation_importance(
                model=model,
                scoring_data=(self.examples.values, targets.values),
                evaluation_fn=evaluation_fn,
                variable_names=self.feature_names,
                scoring_strategy=scoring_strategy,
                subsample=subsample,
                nimportant_vars=n_vars,
                njobs=n_jobs,
                nbootstrap=n_bootstrap,
                verbose=verbose,
                perm_method=perm_method,
                random_state=random_state,
            )

            pi_dict[model_name] = pi_result
            
            del pi_result

        data = {}
        for model_name in self.model_names:
            for func in ["retrieve_multipass", "retrieve_singlepass"]:
                adict = getattr(pi_dict[model_name], func)()
                features = np.array(list(adict.keys()))
                rankings = np.argsort([adict[f][0] for f in features])
                top_features = features[rankings]
                scores = np.array([adict[f][1] for f in top_features])
                pass_method = func.split("_")[1]

                data[f"{pass_method}_rankings__{model_name}"] = (
                    [f"n_vars_{pass_method}"],
                    top_features,
                )
                data[f"{pass_method}_scores__{model_name}"] = (
                    [f"n_vars_{pass_method}", "n_bootstrap"],
                    scores,
                )
            data[f"original_score__{model_name}"] = (
                ["n_bootstrap"],
                pi_dict[model_name].original_score,
            )

        results_ds = to_xarray(data)

        return results_ds

    def _run_interpret_curves(
        self, method, features=None, cat_features=None, n_bins=25, n_jobs=1, subsample=1.0, n_bootstrap=1, 
        feature_encoder=None, 
    ):

        """
        Runs the interpretation curve (partial dependence, accumulated local effects, 
        or individual conditional expectations.) calculations. 
        Includes assessing whether the calculation is 1D or 2D and handling
        initializing the parallelization, subsampling data, and/or using bootstraping
        to compute confidence intervals.

        Returns a nested dictionary with all neccesary inputs for plotting.

        Args:
        ---------
            method : 'pd' , 'ale', or 'ice'
                determines whether to compute partial dependence ('pd'),
                accumulated local effects ('ale'), or individual conditional expectations ('ice')

            features: string, 2-tuple of strings, list of strings, or lists of 2-tuple strings
                feature names to compute for. If 2-tuple, it will compute
                the second-order results.
                
            cat_features: string, or list of strings
                categorical features to compute ALE for. ALE requires a special distinction 
                for categorical features. 

            n_bins : int
                Number of evenly-spaced bins to compute PD/ALE over.

            n_jobs : int or float
                 if int, the number of processors to use for parallelization
                 if float, percentage of total processors to use for parallelization

            subsample : float or integer
                    if value between 0-1 interpreted as fraction of total examples to use
                    if value > 1, interpreted as the number of examples to randomly sample
                        from the original dataset.

            n_bootstrap: integer
                number of bootstrap iterations to perform. Defaults to 1 (no
                        bootstrapping).
        """
        # Check if features is a string
        if is_str(features) or isinstance(features, tuple):
            features = [features]
        
        if not isinstance(features[0], tuple):
            features, cat_features = determine_feature_dtype(self.examples, features)
        else:
            cat_features=[]
            
        results=[]
        cat_results=[]
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
            self.model_names, features, [n_bins], [subsample], [n_bootstrap], 
            )
            
            results = run_parallel(
                func=func, args_iterator=args_iterator, kwargs={}, nprocs_to_use=n_jobs
            )
        
        if len(cat_features) > 0:
            if method == 'ale':
                func = self.compute_first_order_ale_cat
            elif method == 'pd':
                func = self.compute_partial_dependence
            elif method =='ice':
                func = self.compute_individual_cond_expect
                
            args_iterator = to_iterator(
                self.model_names, cat_features,  [subsample], [n_bootstrap], [feature_encoder]
                )

            cat_results = run_parallel(
                func=func, args_iterator=args_iterator, kwargs={}, nprocs_to_use=n_jobs
                )
           
        results=cat_results+results

        results = merge_dict(results)
        results_ds = to_xarray(results)

        return results_ds

    def _store_results(self, method, model_name, features, ydata, xdata, hist_data, categorical=False):
        """
        Store the results of the ALE/PD/ICE calculations into dict,
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
        if method == "pd" or method == 'ice' or (method=='ale' and categorical):
            xdata1 = xdata[0]
            if np.shape(xdata)[0] > 1:
                xdata2 = xdata[1]
                hist_data2 = hist_data[1]
        else:
            if np.ndim(xdata) > 1:
                xdata1 = 0.5 * (xdata[0][1:] + xdata[0][:-1])
                xdata2 = 0.5 * (xdata[1][1:] + xdata[1][:-1])
            else:
                xdata1 = 0.5 * (xdata[1:] + xdata[:-1])
            if np.ndim(xdata) > 1:
                hist_data2 = hist_data[1]
                
                
        results[f"{feature1}{feature2}__{model_name}__{method}"] = (y_shape, ydata)
        results[f"{feature1}__bin_values"] = ([f"n_bins__{feature1}"], xdata1)
        results[f"{feature1}"] = (["n_examples"], hist_data1)
        if xdata2 is not None:
            results[f"{feature2[2:]}__bin_values"] = ([f"n_bins{feature2[2:]}"], xdata2)
            results[f"{feature2[2:]}"] = (["n_examples"], hist_data2)

        return results

    def compute_individual_cond_expect(
        self, model_name, features, n_bins=30, subsample=1.0, n_bootstrap=1
    ):
        """
        Compute the Individual Conditional Expectations (see https://christophm.github.io/interpretable-ml-book/ice.html)
        """
        # Retrieve the model object from the models dict attribute
        model = self.models[model_name]

        # Check if features is a string
        if is_str(features):
            features = [features]

        # Check if feature is valid
        is_valid_feature(features, self.feature_names)

        if float(subsample) != 1.0:
            n_examples = len(self.examples)
            size = int(n_examples * subsample) if subsample <= 1.0 else subsample
            idx = np.random.choice(n_examples, size=size)
            examples = self.examples.iloc[idx, :]
            examples.reset_index(drop=True, inplace=True)
        else:
            examples = self.examples.copy()

        # Extract the values for the features
        feature_values = [examples[f].to_numpy() for f in features]

        # Create a grid of values
        grid = [np.linspace(np.amin(f), np.amax(f), n_bins) for f in feature_values]

        if self.model_output == "probability":
            prediction_method = model.predict_proba
        elif self.model_output == "raw":
            prediction_method = model.predict

        ice_values = []
        for value_set in cartesian(grid):
            examples_temp = examples.copy()
            examples_temp.loc[:, features[0]] = value_set[0]
            ice_values.append(prediction_method(examples_temp.values))

        ice_values = np.array(ice_values).T
        if self.model_output == "probability":
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
            model_name=model_name,
            features=features,
            ydata=ice_values,
            xdata=grid,
            hist_data=feature_values,
        )

        return results

    def compute_partial_dependence(
        self, model_name, features, n_bins=30, subsample=1.0, n_bootstrap=1
    ):

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
            n_bins : int
                Number of evenly-spaced bins to compute PD
            subsample : float between 0-1
                Percent of randomly sampled examples to compute PD for.
            n_bootstrap : int
                Number of bootstrapping

        Returns:
            pd, partial dependence values (in %, i.e., multiplied by 100.)
        """
        # Retrieve the model object from the models dict attribute
        model = self.models[model_name]

        # Check if features is a string
        if is_str(features):
            features = [features]

        # Check if feature is valid
        is_valid_feature(features, self.feature_names)

        # Extract the values for the features
        full_feature_values = [self.examples[f].to_numpy() for f in features]

        # Create a grid of values
        grid = [
            np.linspace(np.amin(f), np.amax(f), n_bins) for f in full_feature_values
        ]

        # get the bootstrap samples
        if n_bootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(
                self.examples, subsample=subsample, n_bootstrap=n_bootstrap
            )
        else:
            bootstrap_indices = [self.examples.index.to_list()]

        if self.model_output == "probability":
            prediction_method = model.predict_proba
        elif self.model_output == "raw":
            prediction_method = model.predict

        pd_values = []

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_indices):
            # get samples
            examples = self.examples.iloc[idx, :].reset_index(drop=True)
            feature_values = [examples[f].to_numpy() for f in features]
            averaged_predictions = []
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

            if self.model_output == "probability":
                # Binary classification, shape is (2, n_points).
                # we output the effect of **positive** class
                # and convert to percentages
                averaged_predictions = averaged_predictions[1]

            # Center the predictions
            averaged_predictions -= np.mean(averaged_predictions)

            pd_values.append(averaged_predictions)

        # Reshape the pd_values for second-order effects
        pd_values = np.array(pd_values)
        if len(features) > 1:
            pd_values = pd_values.reshape(n_bootstrap, n_bins, n_bins)
        else:
            features = features[0]

        results = self._store_results(
            method="pd",
            model_name=model_name,
            features=features,
            ydata=pd_values,
            xdata=grid,
            hist_data=feature_values,
        )

        return results

    def compute_first_order_ale(
        self, model_name, feature, n_bins=30, subsample=1.0, n_bootstrap=1
    ):
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
            n_bins : int
            subsample : float [0,1]
            n_bootstrap : int

        Returns:
        ----------
            results : nested dictionary

        """
        model = self.models[model_name]
        # check to make sure feature is valid
        if feature not in self.feature_names:
            raise KeyError(f"Feature {feature} is not a valid feature.")

        # get the bootstrap samples
        if n_bootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(
                self.examples, subsample=subsample, n_bootstrap=n_bootstrap
            )
        else:
            bootstrap_indices = [self.examples.index.to_list()]

        # Using the original, unaltered feature values
        # calculate the bin edges to be used in the bootstrapping.
        original_feature_values = self.examples[feature].values

        if (self.examples[feature].dtype.name != "category"):
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
            examples = self.examples.iloc[idx, :].reset_index(drop=True)

            # Find the ranges to calculate the local effects over
            # Using xdata ensures each bin gets the same number of examples
            feature_values = examples[feature].values

            # if right=True, then the smallest value in data is not included in a bin.
            # Thus, Define the bins the feature samples fall into. Shift and clip to ensure we are
            # getting the index of the left bin edge and the smallest sample retains its index
            # of 0.
            if (self.examples[feature].dtype.name != "category"):
                indices = np.clip(
                    np.digitize(feature_values, bin_edges, right=True) - 1, 0, None
                )
            else:
                # indices for discrete data 
                indices = np.digitize(feature_values, bin_edges)-1

            # Assign the feature quantile values (based on its bin index) to two copied training datasets,
            # one for each bin edge. Then compute the difference between the corresponding predictions
            predictions = []
            for offset in range(2):
                examples_temp = examples.copy()
                examples_temp[feature] = bin_edges[indices + offset]
                if self.model_output == "probability":
                    predictions.append(model.predict_proba(examples_temp.values)[:, 1])
                elif self.model_output == "raw":
                    predictions.append(model.predict(examples_temp.values))

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
            ale[k, :] -= np.sum(ale[k, :] * index_groupby.size() / examples.shape[0])

        results = self._store_results(
            method="ale",
            model_name=model_name,
            features=feature,
            ydata=ale,
            xdata=bin_edges,
            hist_data=[original_feature_values],
        )

        return results

    def _get_centers(self, x):
        return 0.5 * (x[1:] + x[:-1])

    def compute_second_order_ale(
        self, model_name, features, n_bins=30, subsample=1.0, n_bootstrap=1
    ):
        """
        Computes second-order ALE function on two continuous features data.

        Script is based on the _second_order_ale_quant from
        https://github.com/blent-ai/ALEPython/

        Args:
        ----------
            model_name : str
            features : string
                The name of the feature to consider.
            n_bins : int
            subsample : float between [0,1]
            n_bootstrap : int

        Returns :
        ----------
            results : nested dict
        """
        model = self.models[model_name]

        # make sure there are two features...
        assert len(features) == 2, "Size of features must be equal to 2."

        # check to make sure both features are valid
        if features[0] not in self.feature_names:
            raise TypeError(f"Feature {features[0]} is not a valid feature")

        if features[1] not in self.feature_names:
            raise TypeError(f"Feature {features[1]} is not a valid feature")

        # create bins for computation for both features

        original_feature_values = [self.examples[features[i]].values for i in range(2)]

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
                self.examples, subsample=subsample, n_bootstrap=n_bootstrap
            )
        else:
            bootstrap_indices = [self.examples.index.to_list()]

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
            examples = self.examples.iloc[idx, :].reset_index(drop=True)

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
                    examples_temp[features[i]] = bin_edges[i][
                        indices_list[i] + shifts[i]
                    ]

                if self.model_output == "probability":
                    predictions[shifts] = model.predict_proba(examples_temp.values)[
                        :, 1
                    ]
                elif self.model_output == "raw":
                    predictions[shifts] = model.predict(examples_temp.values)

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
            ale -= np.sum(samples_grid * ale) / len(examples)

            ale_set.append(ale)

        ###ale_set_ds = xarray.DataArray(ale_set).to_masked_array()

        results = self._store_results(
            method="ale",
            model_name=model_name,
            features=features,
            ydata=ale_set,
            xdata=bin_edges,
            hist_data=original_feature_values,
        )

        return results

    def compute_first_order_ale_cat(
        self, model_name, feature, subsample=1.0, n_bootstrap=1, feature_encoder=None,
    ):
        """
        Computes first-order ALE function on a single categorical feature.

        Script is largely from aleplot_1D_categorical from
        PyALE with small modifications (https://github.com/DanaJomar/PyALE). 

        Args:
        ----------
            model_name : str
                 the string identifier for model in the attribute "models" dict
            feature : string
                The name of the feature to consider.
            n_bins : int
            subsample : float [0,1]
            n_bootstrap : int

        Returns:
        ----------
            results : nested dictionary

        """
        from inspect import currentframe, getframeinfo
        from pandas.core.common import SettingWithCopyError
        pd.options.mode.chained_assignment = 'raise'
        
        if feature_encoder is None:
            def feature_encoder_func(data):
                return data 
            feature_encoder = feature_encoder_func
        
        model = self.models[model_name]
        # check to make sure feature is valid
        if feature not in self.feature_names:
            raise Exception(f"Feature {feature} is not a valid feature")

        # get the bootstrap samples
        if n_bootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(
                self.examples, subsample=subsample, n_bootstrap=n_bootstrap
            )
        else:
            bootstrap_indices = [self.examples.index.to_list()]
   
        original_feature_values = [feature_encoder(self.examples[feature].values)]
        xdata = np.array([np.unique(original_feature_values)])
        xdata.sort()
        
        # Initialize an empty ale array
        ale = []

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_indices):
            examples = self.examples.iloc[idx, :].reset_index(drop=True)

            if (examples[feature].dtype.name != "category") or (not examples[feature].cat.ordered):
                examples[feature] = examples[feature].astype(str)          
                groups_order = order_groups(examples, feature)
                groups = groups_order.index.values
                examples[feature] = examples[feature].astype(
                pd.api.types.CategoricalDtype(categories=groups, ordered=True)
                   )
        
            groups = examples[feature].unique()
            groups = groups.sort_values()
            feature_codes = examples[feature].cat.codes
            groups_counts = examples.groupby(feature).size()
            groups_props = groups_counts / sum(groups_counts)

            K = len(groups)
            
            # create copies of the dataframe
            examples_plus = examples.copy()
            examples_neg = examples.copy()
            # all groups except last one
            last_group = groups[K - 1]
            ind_plus = examples[feature] != last_group
            # all groups except first one
            first_group = groups[0]
            ind_neg = examples[feature] != first_group
            # replace once with one level up
            examples_plus.loc[ind_plus, feature] = groups[feature_codes[ind_plus] + 1]
            # replace once with one level down
            examples_neg.loc[ind_neg, feature] = groups[feature_codes[ind_neg] - 1]
            try:
                # predict with original and with the replaced values
                # encode the categorical feature
                examples_coded = pd.concat([examples.drop(feature, axis=1), 
                                            feature_encoder(examples[[feature]])], axis=1)
                # predict
                if self.model_output =='probability':
                    y_hat = model.predict_proba(examples_coded[self.feature_names])[:,1]
                else:
                    y_hat = model.predict(examples_coded[self.feature_names])

                # encode the categorical feature
                examples_plus_coded = pd.concat(
                    [examples_plus.drop(feature, axis=1), 
                     feature_encoder(examples_plus[[feature]])], axis=1
                    )
                # predict
                if self.model_output =='probability':
                    y_hat_plus = model.predict_proba(examples_plus_coded[ind_plus][self.feature_names])[:,1]
                else:
                    y_hat_plus = model.predict(examples_plus_coded[ind_plus][self.feature_names])
                    
                # encode the categorical feature
                examples_neg_coded = pd.concat(
                        [examples_neg.drop(feature, axis=1), 
                         feature_encoder(examples_neg[[feature]])], axis=1
                    )
                # predict
                if self.model_output =='probability':
                    y_hat_neg = model.predict_proba(examples_neg_coded[ind_neg][self.feature_names])[:,1]
                else:
                    y_hat_neg = model.predict(examples_neg_coded[ind_neg][self.feature_names])
                
            except Exception as ex:
                raise Exception(
                    """There seems to be a problem when predicting with the model.
                    Please check the following: 
                    - Your model is fitted.
                        - The list of predictors contains the names of all the features"""
                    """ used for training the model.
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
                        {"eff": Delta_plus, feature: groups[feature_codes[ind_plus] + 1]}
                    ),
                    pd.DataFrame({"eff": Delta_neg, feature: groups[feature_codes[ind_neg]]}),
                ]
                )
            res_df = delta_df.groupby([feature]).mean()
            res_df.loc[:,"ale"] = res_df.loc[:, "eff"].cumsum()

            res_df.loc[groups[0]] = 0
            # sort the index (which is at this point an ordered categorical) as a safety measure
            res_df = res_df.sort_index()
            
            # Subtract the mean value to get the centered value. 
            ale_temp = res_df["ale"] - sum(res_df["ale"] * groups_props)
            ale.append(ale_temp)

        ale = np.array(ale)
        
        results = self._store_results(
            method="ale",
            model_name=model_name,
            features=feature,
            ydata=ale,
            xdata=xdata,
            hist_data=original_feature_values,
            categorical = True,
        )
        
        return results
    
    def friedman_h_statistic(self, model_name, feature_tuple, n_bins=30, subsample=1.0):
        """
        Compute the H-statistic for two-way interactions between two features.

        Args:
            model_name : str
            feature_tuple : 2-tuple of strs
            n_bins : int

        Returns:
        """
        self.model_names = [model_name]
        feature1, feature2 = feature_tuple
        features = [feature1, feature2]

        # Compute the first-order effects for the two features.
        results = self._run_interpret_curves(
            method="pd",
            features=features,
            n_bins=n_bins,
            n_jobs=2,
            subsample=subsample,
            n_bootstrap=1,
        )

        feature1_pd = results[f"{feature1}__{model_name}__pd"].values.squeeze()
        feature2_pd = results[f"{feature2}__{model_name}__pd"].values.squeeze()

        # Compute the second-order effects between the two features
        combined_results = self.compute_partial_dependence(
            model_name, feature_tuple, n_bins, subsample=subsample
        )

        combined_results = to_xarray(combined_results)

        combined_pd = combined_results[
            f"{feature1}__{feature2}__{model_name}__pd"
        ].values.squeeze()

        # Calculate the H-statistics
        pd_decomposed = feature1_pd[:, np.newaxis] + feature2_pd[np.newaxis, :]
        numer = (combined_pd - pd_decomposed) ** 2
        denom = (combined_pd) ** 2
        H_squared = np.sum(numer) / np.sum(denom)

        return sqrt(H_squared)


    def compute_scalar_interaction_stats(self, method, data, model_names, n_bins=30, subsample=1.0, n_bootstrap=1, n_jobs=1, **kwargs):
        """
        Wrapper function for computing the interaction strength statistic (see below).
        Will perform calculation in parallel for multiple models. 
        """
        if method == 'ias':
            func=self.compute_interaction_strength
        elif method == 'h_stat':
            func = self.friedman_h_statistic

        self.data = data
        args_iterator = to_iterator(
            model_names, [n_bins], [subsample], [n_bootstrap]
        )
        kwargs['n_jobs'] = n_jobs
        results = run_parallel(
            func=func, 
            args_iterator=args_iterator, 
            kwargs=kwargs, nprocs_to_use=n_jobs
        )
        
        results = merge_dict(results)

        return results 


    def compute_interaction_strength(
        self, model_name, n_bins=30, subsample=1.0, n_bootstrap=1, **kwargs
    ):
        """
        Compute the interaction strenth of a ML model (based on IAS from
        Quantifying Model Complexity via Functional
        Decomposition for Better Post-Hoc Interpretability).

        Args:
        --------------------
            results : xarray.Dataset 

            model_names : list of strings

            n_bins : integer

            subsample : float or integer

            n_jobs : float or integer

            n_bootstrap : integer

            ale_subsample : float or integer


        """
        ale_subsample = kwargs.get("ale_subsample", subsample)
        model = self.models[model_name]
        feature_names = list(self.examples.columns)
        
        if 'Run Date' in feature_names:
            feature_names.remove('Run Date')

        
        try:
            data = self.data
        except:
            # Get the ALE curve for each feature    
            feature_names = list(self.examples.columns)
            n_jobs = kwargs.get('n_jobs', 1) 
            data = self._run_interpret_curves(
                    method="ale",
                    features=feature_names,
                    n_bins=n_bins,
                    n_jobs=n_jobs,
                    subsample=ale_subsample,
                    n_bootstrap=1,
                    )
       

        # Get the interpolated ALE curves
        ale_main_effects = {}
        for f in feature_names:
            try:
                main_effect = data[f"{f}__{model_name}__ale"].values.squeeze()
            except:
                continue 
            x_values = data[f"{f}__bin_values"].values

            ale_main_effects[f] = interp1d(
                x_values, main_effect, fill_value="extrapolate", kind="linear"
            )

        # get the bootstrap samples
        if n_bootstrap > 1 or float(subsample) != 1.0:
            bootstrap_indices = compute_bootstrap_indices(
                self.examples, subsample=subsample, n_bootstrap=n_bootstrap
            )
        else:
            bootstrap_indices = [self.examples.index.to_list()]

        ias = []
        for k, idx in enumerate(bootstrap_indices):
            examples = self.examples.iloc[idx, :].values
            if self.model_output == "probability":
                predictions = model.predict_proba(examples)[:, 1]
            else:
                predictions = model.predict(examples)

            # Get the average model prediction
            avg_prediction = np.mean(predictions)

            # Get the ALE value for each feature per example
            main_effects = np.array([ale_main_effects[f](examples[:,i]) for i,f in enumerate(feature_names)])
            # Sum the ALE values per example and add on the average value 
            main_effects = np.sum(main_effects.T, axis=1) + avg_prediction 

            num = np.sum((predictions - main_effects) ** 2)
            denom = np.sum((predictions - avg_prediction) ** 2)

            # Compute the interaction strength
            ias.append(num / denom)

        return {model_name : np.array(ias)}


    def compute_ale_variance(
        self, data, model_names, n_bins=30, subsample=1.0, n_bootstrap=1, **kwargs
    ):
        """
        Compute the standard deviation of the ALE values 
        for each feature and rank then for predictor importance. 

        Args:
        --------------------
            results : xarray.Dataset 

            model_names : list of strings

            n_bins : integer

            subsample : float or integer

            n_jobs : float or integer

            n_bootstrap : integer

            ale_subsample : float or integer


        """
        results={}
        for model_name in model_names:
            if data is None:
                # Get the ALE curve for each feature    
                ale_subsample = kwargs.get("ale_subsample", subsample)
                feature_names = list(self.examples.columns)
                model = self.models[model_name]
                feature_names = list(self.examples.columns)
                n_jobs = kwargs.get('n_jobs', 1)
                data = self._run_interpret_curves(
                    method="ale",
                    features=feature_names,
                    n_bins=n_bins,
                    n_jobs=n_jobs,
                    subsample=ale_subsample,
                    n_bootstrap=n_bootstrap,
                    )
            else:
                feature_names = list(set([f.split('__')[0] for f in data.data_vars if 'ale' in f ]))

            # Compute the std over the bin axis
            ale_std = np.array([np.std(data[f"{f}__{model_name}__ale"].values, ddof=1, axis=1) for f in feature_names])

            # Average over the bootstrap indices 
            idx = np.argsort(np.mean(ale_std, axis=1))[::-1]

            feature_names_sorted = np.array(feature_names)[idx]
            ale_std_sorted = ale_std[idx, :]

            results[f"ale_variance_rankings__{model_name}"] = (
                    [f"n_vars_ale_variance"],
                    feature_names_sorted,
                )
            results[f"ale_variance_scores__{model_name}"] = (
                [f"n_vars_ale_variance", 'n_bootstrap'],
                    ale_std_sorted,
                )

        results_ds = to_xarray(results)

        return results_ds 
            












import numpy as np
import pandas as pd
from PermutationImportance.permutation_importance import sklearn_permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from treeinterpreter import treeinterpreter as ti

list_of_acceptable_tree_models = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
]


class ModelClarify:

    """
    Class for computing various ML model interpretations...blah blah blah

    Args:
        model : a scikit-learn model
        examples : pandas DataFrame or ndnumpy array. If ndnumpy array, make sure
            to specify the feature names
        targets: numpy array of targets/labels
        classification: defaults to True for classification problems. 
            Set to false otherwise.
        feature_names : defaults to None. Should only be set if examples is a 
            nd.numpy array. Make sure it's a list
    """
    def __init__(self, model, examples, targets=None, classification=True, 
            feature_names=None):

        self._model    = model
        self._examples = examples
        self._targets  = targets

        # make sure data is the form of a pandas dataframe regardless of input type
        if isinstance(self._examples, np.ndarray): 
            if (feature_names is None): 
                raise Exception('Feature names must be specified.')    
            else:
                self._feature_names  = feature_names
                self._examples = pd.DataFrame(data=examples, columns=feature_names)
        else:
            self._feature_names  = examples.columns.to_list()

        self._classification = classification
     

    def calc_ale(self, feature, xdata=None, subsample=1.0, nbootstrap=1):
        """
            Calculates the Accumulated local effect.
        """
        self.subsample = subsample
        self.nbootstrap = nbootstrap
        compute_func = self.calculate_first_order_ale 
        ale, xdata = self.compute_first_order_interpretation_curve(feature, compute_func, xdata=xdata)
        
        return ale, xdata

    def calc_pdp(self, feature, xdata=None, subsample=1.0, nbootstrap=1):
        """
            Calculates the Accumulated local effect.
        """
        self.subsample = subsample
        self.nbootstrap = nbootstrap
        compute_func = self.compute_1d_partial_dependence                       
        pdp, xdata = self.compute_first_order_interpretation_curve(feature, compute_func, xdata=xdata)
        
        return pdp, xdata


    def get_indices_based_on_performance(self, n_examples=10):

        """
        Determines the best 'hits' (forecast probabilties closest to 1)
        or false alarms (forecast probabilities furthest from 0 )
        or misses (forecast probabilties furthest from 1 )

        The returned dictionary below can be passed into interpert_tree_based_model()

        Args:
        ------------------
            n_examples : Integer representing the number of indices (examples) to return.
                          Default is 10
        """

       #get indices for each binary class
        positive_idx = np.where(self._targets > 0)
        negative_idx = np.where(self._targets < 1)

        #get targets for each binary class
        positive_class = self._targets[positive_idx[0]]
        negative_class = self._targets[negative_idx[0]]    
 
        #compute forecast probabilities for each binary class
        forecast_probabilities_on_pos_class = self._model.predict_proba(self._examples.loc[positive_idx[0], :])[:,1]
        forecast_probabilities_on_neg_class = self._model.predict_proba(self._examples.loc[negative_idx[0], :])[:,1]        
    
        #compute the absolute difference
        diff_from_pos = abs(positive_class - forecast_probabilities_on_pos_class)
        diff_from_neg = abs(negative_class - forecast_probabilities_on_neg_class)
    
        #sort based on difference and store in array
        sorted_diff_for_hits = np.array( sorted( zip(diff_from_pos, positive_idx[0]), key = lambda x:x[0]))
        sorted_diff_for_misses = np.array( sorted( zip(diff_from_pos, positive_idx[0]), key = lambda x:x[0], reverse=True ))
        sorted_diff_for_false_alarms = np.array( sorted( zip(diff_from_neg, negative_idx[0]), key = lambda x:x[0], reverse=True )) 
        sorted_diff_for_corr_negs = np.array(
            sorted(zip(diff_from_neg, negative_idx[0]), key=lambda x: x[0])
        )

        #store all resulting indicies in one dictionary
        adict =  { 
                    'hits': [ sorted_diff_for_hits[i][1] for i in range(n_examples) ],
                    'false_alarms': [ sorted_diff_for_false_alarms[i][1] for i in range(n_examples) ],
                    'misses': [ sorted_diff_for_misses[i][1] for i in range(n_examples) ],
                      "corr_negs": [
                sorted_diff_for_corr_negs[i][1] for i in range(n_examples)
            ],

                    } 

        for key in list(adict.keys()):
            adict[key] = np.array(adict[key]).astype(int)

        return adict  
    
    def _sort_df(self, df):
        """
        sort a dataframe by the absolute value 
        """
        return df.reindex(df.abs().sort_values(ascending=False).index)

    def get_top_contributors(self, n_examples=100):
        """
        Return the "num" number of top contributors (based on absolute value)

            Parameters:
            -----------
                ncontributors: integer
                    number of top contributors to return 
                n_examples: integer
                    see get_
        """
        performance_dict = self.get_indices_based_on_performance(n_examples)
        dict_of_dfs = self.tree_interpreter_performance_based(
            performance_dict=performance_dict
        )
        adict = {}
        for key in list(dict_of_dfs.keys()):
            df = dict_of_dfs[key]
            series = df.mean(axis=0)
            sorted_df = self._sort_df(series)
            idxs = performance_dict[key]
            top_vars = {}
            for var in list(sorted_df.index):
                if var == 'Bias':
                    top_vars[var] = {
                                    'Mean Value': None,
                                    'Mean Contribution' : series[var]
                                    }
                else:
                    top_vars[var] = {
                        "Mean Value": np.mean(self._examples[var].values[idxs]),
                        "Mean Contribution": series[var],
                    }
            adict[key] = top_vars

        return adict

    def tree_interpreter_performance_based(self, performance_dict=None):

        """
        Method for intrepreting tree based ML models using treeInterpreter. 
        Uses indices from dictionary returned by get_indices_based_on_performance()

        ADD REFERENCE HERE SOMEWHERE

        """

        # check to make sure model is of type Tree
        if type(self._model).__name__ not in list_of_acceptable_tree_models:
            raise Exception(f"{model_name} model is not accepted for this method.")

        if performance_dict is None:
            performance_dict = self.get_indices_based_on_performance()

        # will be returned; a list of pandas dataframes, one for each performance dict key
        dict_of_dfs = {}

        for key, values in zip(performance_dict.keys(), performance_dict.values()):

            print(key)
            # number of examples
            n_examples = values.shape[0]

            # get examples for key
            tmp_examples = self._examples.loc[values, :]

            print(f"Interpreting {n_examples} examples from {key}")
            prediction, bias, contributions = ti.predict(self._model, tmp_examples)

            forecast_probabilities = (
                self._model.predict_proba(tmp_examples)[:, 1] * 100.0
            )
            positive_class_contributions = contributions[:, :, 1]
            positive_class_bias = bias[1][1]

            tmp_data = []

            # loop over each case appending each feature and value to a dictionary
            for i in range(n_examples):

                key_list = []
                var_list = []

                for c, feature in zip(
                    positive_class_contributions[i, :], self._feature_names
                ):

                    key_list.append(feature)
                    var_list.append(round(100.0 * c, 2))

                key_list.append('Bias')
                var_list.append(round(100. * positive_class_bias, 2))
                tmp_data.append(dict(zip(key_list, var_list)))

            # return a pandas DataFrame to do analysis on
            contributions_dataframe = pd.DataFrame(data=tmp_data)

            dict_of_dfs[key] = contributions_dataframe

        return dict_of_dfs

    def tree_interpreter(self):

        """
        Method for intrepreting tree based ML models using treeInterpreter.
        Uses all data passed in to constructor
 
        ADD REFERENCE HERE SOMEWHERE

        """

        # check to make sure model is of type Tree
        if type(self._model).__name__ not in list_of_acceptable_tree_models:
            raise Exception(f"{model_name} model is not accepted for this method.")

        # number of examples
        n_examples = self._examples.shape[0]

        print(f"Interpreting {n_examples} examples...")
        prediction, bias, contributions = ti.predict(self._model, self._examples)

        forecast_probabilities = self._model.predict_proba(self._examples)[:, 1] * 100.0
        positive_class_contributions = contributions[:, :, 1]
        positive_class_bias = bias[1][1]        

        tmp_data = []

        # loop over each case appending each feature and value to a dictionary
        for i in range(n_examples):

            key_list = []
            var_list = []

            for c, feature in zip(
                positive_class_contributions[i, :], self._feature_names
            ):

                key_list.append(feature)
                var_list.append(round(100.0 * c, 2))

            key_list.append('Bias')
            var_list.append(round(100. * positive_class_bias, 2))
            
            tmp_data.append(dict(zip(key_list, var_list)))

        # return a pandas DataFrame to do analysis on
        contributions_dataframe = pd.DataFrame(data=tmp_data)

        return contributions_dataframe

    def compute_1d_partial_dependence(self, examples, feature=None, xdata=None, **kwargs):

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

        # check to make sure a feature is present...
        if feature is None:
            raise Exception("Specify a feature")

        # check to make sure feature is valid
        if feature not in self._feature_names:
            raise Exception(f"Feature {feature} is not a valid feature")

        # get data in numpy format
        column_of_data = examples[feature].to_numpy()

        # define bins based on 10th and 90th percentiles
        xdata = np.linspace(
            np.percentile(column_of_data, 5), np.percentile(column_of_data, 95), num=20
        )

        # define output array to store partial dependence values
        pdp_values = np.full(xdata.shape[0], np.nan)

        # for each value, set all indices to the value, make prediction, store mean prediction
        for i, value in enumerate(xdata):

            copy_df = examples.copy()
            copy_df.loc[:, feature] = value

            if self._classification is True:
                # Convert to percentages
                predictions = self._model.predict_proba(copy_df)[:, 1] * 100.
            else:
                predictions = self._model.predict(copy_df)

            pdp_values[i] = np.mean(predictions)

        return pdp_values, xdata

    def compute_2d_partial_dependence(self, features, **kwargs):

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
            feature : tuple of type string of predictor names

        """

        # make sure there are two features...
        if len(features) > 2:
            raise Exception(f"tuple of size {len(features)} is greater than 2")
        if len(features) < 2:
            raise Exception(f"tuple of size {len(features)} is less than 2")

        # make sure both features are valid...
        if feature[0] is None or feature[1] is None:
            raise Exception("One or more features is of type None.")

        # check to make sure feature is valid
        if (
            feature[0] not in self._feature_names
            or feature[1] not in self._feature_names
        ):
            raise Exception(f"Feature {feature} is not a valid feature")

        # get data for both features
        values_for_var1 = self._examples[features[0]].to_numpy()
        values_for_var2 = self._examples[features[1]].to_numpy()

        # get ranges of data for both features
        var1_range = np.linspace(
            np.percentile(values_for_var1, 10),
            np.percentile(values_for_var1, 90),
            num=20,
        )
        var2_range = np.linspace(
            np.percentile(values_for_var2, 10),
            np.percentile(values_for_var2, 90),
            num=20,
        )

        # define 2-D grid
        pdp_values = np.full((var1_range.shape[0], var2_range.shape[0]), np.nan)

        # similar concept as 1-D, but for 2-D
        for i, value1 in enumerate(var1_range):
            for k, value2 in enumerate(var2_range):
                copy_df = self._examples.copy()
                copy_df.loc[features[0]] = value1
                copy_df.loc[features[1]] = value2

                if self._classification is True:
                    predictions = self._model.predict_proba(copy_df)[:, 1] * 100.
                else:
                    predictions = self._model.predict(copy_df)

                pdp_values[i, k] = np.mean(predictions)

        return pdp_values, var1_range, var2_range

    def calculate_first_order_ale(self, examples, feature=None, xdata=None):

        """
            Computes first-order ALE function on single continuous feature data.

            Parameters
            ----------
            feature : string
                The name of the feature to consider.
            xdata : array
                Quantiles of feature.
        """

        # TODO: incorporate the monte carlo aspect into these routines in a clean way...
        nbins = 15
        # make sure feature is set
        if feature is None:
            raise Exception("Specify a feature.")

        # convert xdata to array if list
        if isinstance(xdata, list):
            xdata = np.array(xdata)

        if xdata is None:
            # Find the ranges to calculate the local effects over
            # Using xdata ensures each bin gets the same number of examples
            xdata = np.percentile(
                examples[feature].values, np.arange(2.5, 97.5 + 5, 5)
            )

        # define ALE function
        ale = np.zeros(len(xdata) - 1)

        # loop over all ranges
        for i in range(1, len(xdata)):

            # get subset of data
            df_subset = self._examples[
                (examples[feature] >= xdata[i - 1])
                & (examples[feature] < xdata[i])
            ]

            # Without any observation, local effect on splitted area is null
            if len(df_subset) != 0:
                lower_bound = df_subset.copy()
                upper_bound = df_subset.copy()

                # The main ALE idea that compute prediction difference between same data except feature's one
                lower_bound[feature] = xdata[i - 1]
                upper_bound[feature] = xdata[i]

                # The main ALE idea that compute prediction difference between same data except feature's one
                lower_bound[feature] = xdata[i - 1]
                upper_bound[feature] = xdata[i]


                upper_bound = upper_bound.values
                lower_bound = lower_bound.values


                if self._classification:
                    effect = 100.0 * (
                        self._model.predict_proba(upper_bound)[:, 1]
                        - self._model.predict_proba(lower_bound)[:, 1]
                    )
                else:
                    effect = self._model.predict(upper_bound) - self._model.predict(
                        lower_bound
                    )

                ale[i - 1] = np.mean(effect)

        # The accumulated effect
        ale = ale.cumsum()
        mean_ale = ale.mean()

        # Now we have to center ALE function in order to obtain null expectation for ALE function
        ale -= mean_ale

        return ale, xdata

    def compute_first_order_interpretation_curve(self, feature, compute_func, xdata):
        """
        Computes first-order ALE function for a feature with bootstrap 
        resampling for confidence intervals. Additional functionality for
        bootstrap resampling to plot confidence intervals.
    
        Args:
        --------------
            feature : str
                the feature name (in the pandas.DataFrame) to
                compute the interpretation curve for.
            compute_func : callable
                Either the ALE or PDP computation functions 
            xdata : array shape (N,)
                The x values at which to compute the interpretation curves.
                If None, the values are the percentile values from 2.5-97.5 every 5% 
            subsample : float (0,1]
                subsampling portion. Can be useful to reduce computational expensive
                Examples are bootstrap resampled with replacement
            nbootstrap : int [1,inf]
                Number of bootstrapp resampling. Used to provided confidence intervals 
                on the interpretation curves. 

        Returns:
        -----------------
            ydata : array, shape (nboostrap, N,)
                Values of the interpretation curve
            xdata : array, shape (N)
                Values of where the interpretation curves was calculated.
        """
        n_examples = len(self._examples)
        bootstrap_replicates = np.asarray(
            [
                [
                    np.random.choice(range(n_examples))
                    for _ in range(int(self.subsample * n_examples))
                ]
                for _ in range(self.nbootstrap)
            ]
        )

        if self.nbootstrap > 1:
            xdata = np.percentile(
                self._examples[feature].values, np.arange(2.5, 97.5 + 5, 5)
            )
            ydata_set = []
            for _, idx in enumerate(bootstrap_replicates):
                examples_temp = self._examples.iloc[idx, :]
                ydata, xdata = compute_func(
                    examples=examples_temp, feature=feature, xdata=xdata
                )
                ydata_set.append(ydata)

            return ydata_set, xdata

        else:
            ydata, xdata = compute_func(examples=self._examples, feature=feature, xdata=xdata)

        return ydata, xdata


    def calculate_second_order_ale(self, features=None, xdata=None):
        """
            Computes second-order ALE function on two continuous features data.

            Parameters
            ----------
            feature : string
                The name of the feature to consider.
            xdata : array
                Quantiles of feature.
        """
        # make sure feature is set
        if features is None:
            raise Exception("Specify two features!")

        # convert xdata to array if list
        if isinstance(xdata, list):
            xdata = np.array(xdata)

        if xdata is None:
            xdata = np.array(
                [
                    np.percentile(self._examples[f].values, np.arange(2.5, 97.5 + 5, 5))
                    for f in features
                ]
            )

        # define ALE function
        ale = np.zeros((xdata.shape[1] - 1, xdata.shape[1] - 1))

        for i in range(1, len(xdata[0])):
            for j in range(1, len(xdata[1])):
                # Select subset of training data that falls within subset
                df_subset = self._examples[
                    (self._examples[features[0]] >= xdata[0, i - 1])
                    & (self._examples[features[0]] < xdata[0, i])
                    & (self._examples[features[1]] >= xdata[1, j - 1])
                    & (self._examples[features[1]] < xdata[1, j])
                ]
                # Without any observation, local effect on splitted area is null
                if len(df_subset) != 0:
                    # get lower and upper bounds on accumulated grid
                    z_low = [df_subset.copy() for _ in range(2)]
                    z_up = [df_subset.copy() for _ in range(2)]

                    # The main ALE idea that compute prediction difference between
                    # same data except feature's one
                    z_low[0][features[0]] = xdata[0, i - 1]
                    z_low[0][features[1]] = xdata[1, j - 1]
                    z_low[1][features[0]] = xdata[0, i]
                    z_low[1][features[1]] = xdata[1, j - 1]
                    z_up[0][features[0]] = xdata[0, i - 1]
                    z_up[0][features[1]] = xdata[1, j]
                    z_up[1][features[0]] = xdata[0, i]
                    z_up[1][features[1]] = xdata[1, j]

                    if self._classification is True:
                        effect = 100.0 * (
                            self._model.predict_proba(z_up[1])[:, 1]
                            - self._model.predict_proba(z_up[0])[:, 1]
                            - (
                                self._model.predict_proba(z_low[1])[:, 1]
                                - self._model.predict_proba(z_low[0])[:, 1]
                            )
                        )
                    else:
                        effect = (
                            self._model.predict(z_up[1])
                            - self._model.predict(z_up[0])
                            - (
                                self._model.predict(z_low[1])
                                - self._model.predict(z_low[0])
                            )
                        )

                    ale[i - 1, j - 1] = np.mean(effect)

        # The accumulated effect
        ale = np.cumsum(ale, axis=0)

        # Now we have to center ALE function in order to obtain null expectation for ALE function
        ale -= ale.mean()

        return ale, xdata

    def permutation_importance(
        self,
        n_multipass_vars,
        evaluation_fn="auprc",
        subsample=1.0,
        njobs=1,
        nbootstrap=1,
    ):
        """
        Perform single or multipass permutation importance using Eli's code.

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
        if evaluation_fn.lower() == "auc":
            evaluation_fn = roc_auc_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == "auprc":
            evaluation_fn = average_precision_score
            scoring_strategy = "argmin_of_mean"

        print(evaluation_fn)
        targets =pd.DataFrame(data=self._targets, columns=['Test'])
        result = sklearn_permutation_importance(
            model=self._model,
            scoring_data=(self._examples, targets),
            evaluation_fn=evaluation_fn,
            variable_names=self._feature_names,
            scoring_strategy=scoring_strategy,
            subsample=subsample,
            nimportant_vars=n_multipass_vars,
            njobs=njobs,
            nbootstrap=nbootstrap,
        )
        return result

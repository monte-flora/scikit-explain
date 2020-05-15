import numpy as np
import pandas as pd

from partial_dependence import PartialDependence
from accumulated_local_effects import AccumulatedLocalEffects
from tree_interpreter import TreeInterpreter
from interpretability_plotting import InterpretabilityPlotting

from PermutationImportance.permutation_importance import sklearn_permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

list_of_acceptable_tree_models = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
]


class ModelClarify:

    """
<<<<<<< HEAD
    Class for running various ML model interpretations.

    Args:
        model : a trained single scikit-learn model, or list of scikit-learn models, or
            dictionary of models where the key is a generic name and the value
            is a train model.
=======
    ModelClarify is composed of various ML model interpretion methods. 
    It includes permutation importance, partial dependence plots,
    accumulated local effects, and random forest feature contributions. 


    Attributes:
        model : a scikit-learn model or dict of models 
>>>>>>> master
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

<<<<<<< HEAD
        # if model is of type list or single objection, convert to dictionary
        if not isinstance(model, dict):
            if isinstance(model, list):
                self._models = {type(m).__name__ : m for m in model}
            else:
                self._models = {type(model).__name__ : model}
        # user provided a dict
        else:
            self._models = model

=======
        if not isinstance(model, dict):
            self.model_set  = {type(model).__name__ : model}
        else:
            self.model_set = model
        
>>>>>>> master
        self._examples = examples

        # check that targets are assigned correctly
        if isinstance(targets, list):
            self._targets = np.array(targets)
        elif isinstance(targets, np.ndarray):
            self._targets = targets
        else:
            raise TypeError('Target variable must numpy array.')

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

<<<<<<< HEAD
        # initialize a PD object
        self._pdp_object = PartialDependence(model=model, examples=examples,
                                targets=targets, classification=classification,
                                feature_names=feature_names)

        # initialize a ALE object
        self._ale_object = AccumulatedLocalEffects(model=model, examples=examples,
                                targets=targets, classification=classification,
                                feature_names=feature_names)

        # initialize a plotting object
        self._clarify_plot_obj = InterpretabilityPlotting()

    def __str__(self):

        return '{}'.format(self._models)

    def run_pd(self, features=None, **kwargs):
=======
    def calc_ale(self, feature, model=None, xdata=None, subsample=1.0, nbootstrap=1):
>>>>>>> master
        """
            Runs the partial dependence calculation and populates a dictionary with all
            necessary inputs for plotting.

            feature: List of strings for first-order partial dependence, or list of tuples
                     for second-order
            subsample: a float (between 0-1) for fraction of examples used in bootstrap
            nbootstrap: number of bootstrap iterations to perform. Defaults to 1 (no
                        bootstrapping).

            Return:
                dictionary of PD values for each model and feature set specified. Will be
                used for plotting.
        """
<<<<<<< HEAD

        # compute pd
        self._pdp_object.run_pd(features=features, **kwargs)

        # get the final dictionary object used for plotting
        self.pd_dict = self._pdp_object.get_final_dict()

        return self.pd_dict

    def plot_pd(self, **kwargs):
=======
        if model is None: 
            model = self.model_set.items()

        self.subsample = subsample
        self.nbootstrap = nbootstrap
        compute_func = self.calculate_first_order_ale 
        ale, xdata = self.compute_first_order_interpretation_curve(feature, compute_func, model=model, xdata=xdata)
        
        return ale, xdata

    def calc_pdp(self, feature, model=None, xdata=None, subsample=1.0, nbootstrap=1):
>>>>>>> master
        """
            Plots the PD. If the first instance is a tuple, then a 2-D plot is
            assumed, else 1-D.
        """
<<<<<<< HEAD
=======
        if model is None:
            model = self.model_set.items()

        self.subsample = subsample
        self.nbootstrap = nbootstrap
        compute_func = self.compute_1d_partial_dependence                       
        pdp, xdata = self.compute_first_order_interpretation_curve(feature, compute_func, model=model, xdata=xdata)
        
        return pdp, xdata
>>>>>>> master

        # plot the PD data. Use first feature key to see if 1D (str) or 2D (tuple)
        if isinstance(list(self.pd_dict.keys())[0], tuple):
            return self._clarify_plot_obj.plot_2d_field(self.pd_dict, **kwargs)
        else:
            return self._clarify_plot_obj.plot_1d_curve(self.pd_dict, **kwargs)

    def run_ale(self, features=None, **kwargs):
        """
            Runs the accumulated local effects calculation and populates a dictionary
            with all necessary inputs for plotting.

            feature: List of strings for first-order partial dependence, or list of tuples
                     for second-order
            subsample: a float (between 0-1) for fraction of examples used in bootstrap
            nbootstrap: number of bootstrap iterations to perform. Defaults to 1 (no
                        bootstrapping).

            Return:
                dictionary of ALE values for each model and feature set specified. Will be
                used for plotting.
        """

        # compute ale
        self._ale_object.run_ale(features=features, **kwargs)

        # get the final dictionary object used for plotting
        self.ale_dict = self._ale_object.get_final_dict()

        return self.ale_dict


    def plot_ale(self, **kwargs):
        """
            Plots the ALE. If the first instance is a tuple, then a 2-D plot is
            assumed, else 1-D.
        """

        # plot the PD data. Use first feature key to see if 1D (str) or 2D (tuple)
        if isinstance(list(self.ale_dict.keys())[0], tuple):
            #return self._clarify_plot_obj.plot_2d_ale(self.pd_dict, **kwargs)
            return print("No 2D ALE plotting functionality yet... sorry!")
        else:
            return self._clarify_plot_obj.plot_1d_curve(self.ale_dict, **kwargs)


    def get_indices_based_on_performance(self, model, n_examples=None):
        """
        Determines the best hits, worst false alarms, worst misses, and best
        correct negatives using the data provided during initialization.

        Args:
        ------------------
            model : The model to process
            n_examples: number of "best/worst" examples to return. If None,
                the routine uses the whole dataset

        Return:
            a dictionary containing the indices of each of the 4 categories
            listed above
        """

        #default is to use all examples
        if (n_examples is None):
            n_examples = self._examples.shape[0]

        #make sure user didn't goof the input
        if (n_examples <= 0):
            print("n_examples less than or equals 0. Defaulting back to all")
            n_examples = self._examples.shape[0]

        #get indices for each binary class
        positive_idx = np.where(self._targets > 0)[0]   #77
        negative_idx = np.where(self._targets < 1)[0]   #173

        #get targets for each binary class
        positive_class = self._targets[positive_idx]
        negative_class = self._targets[negative_idx]    
 
        #compute forecast probabilities for each binary class
        forecast_probs_pos_class = model.predict_proba(self._examples.iloc[positive_idx])[:,1]
        forecast_probs_neg_class = model.predict_proba(self._examples.iloc[negative_idx])[:,1]        
    
        #compute the absolute difference
<<<<<<< HEAD
        diff_from_pos = abs(positive_class - forecast_probs_pos_class)
        diff_from_neg = abs(negative_class - forecast_probs_neg_class)
=======
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
        adict = { 
                'hits': [ sorted_diff_for_hits[i][1] for i in range(min(n_examples, sorted_diff_for_hits.shape[0])) ],
                'false_alarms': [ sorted_diff_for_false_alarms[i][1] for i in range(min(n_examples, sorted_diff_for_false_alarms.shape[0])) ],
                'misses': [ sorted_diff_for_misses[i][1] for i in range(min(n_examples, sorted_diff_for_misses.shape[0])) ],
                "corr_negs": [sorted_diff_for_corr_negs[i][1] for i in range(min(n_examples, sorted_diff_for_corr_negs.shape[0])) ],
                } 

        for key in list(adict.keys()):
            adict[key] = np.array(adict[key]).astype(int)

        return adict  

>>>>>>> master

        #sort based on forecast probabilities (ascending order assumed by argsort)
        sorted_hits = np.argsort(diff_from_pos) #best hits
        sorted_miss = np.argsort(diff_from_pos)[::-1] #worst misses
        sorted_fa   = np.argsort(diff_from_neg)[::-1] #worst false alarms
        sorted_cn   = np.argsort(diff_from_neg) #best corr negs

<<<<<<< HEAD
        sorted_dict = {
                        'hits':         positive_idx[sorted_hits[:n_examples]].astype(int),
                        'misses':       positive_idx[sorted_miss[:n_examples]].astype(int),
                        'false_alarms': negative_idx[sorted_fa[:n_examples]].astype(int),
                        'corr_negs':    negative_idx[sorted_cn[:n_examples]].astype(int)
                      }
=======
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
            sorted_df = series.reindex(series.abs().sort_values(ascending=False).index)
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
>>>>>>> master

        return sorted_dict

<<<<<<< HEAD





        # # compute forecast probabilities
        # forecast_prob = model.predict_proba(self._examples)[:,1]

        # #get indices of hits, misses, false alrams, and correct negs
        # ihit         = np.where((self._targets > 0) & (forecast_prob > ModelClarify.default_binary_threshold))[0]
        # imiss        = np.where((self._targets > 0) & (forecast_prob < ModelClarify.default_binary_threshold))[0]
        # ifalse_alarm = np.where((self._targets < 1) & (forecast_prob > ModelClarify.default_binary_threshold))[0]
        # icorr_neg    = np.where((self._targets < 1) & (forecast_prob < ModelClarify.default_binary_threshold))[0]

        # #sort based on forecast_prob
        # sorted_hits = np.argsort(forecast_prob[ihit])[::-1]  #best hits
        # sorted_miss = np.argsort(forecast_prob[imiss]) #worst misses
        # sorted_fa   = np.argsort(forecast_prob[ifalse_alarm])[::-1] #worst false alarms
        # sorted_cn   = np.argsort(forecast_prob[icorr_neg]) #best corr negs

        # sorted_dict = {
        #                 'hits': ihit[sorted_hits[:n_examples]].astype(int),
        #                 'false_alarms': ifalse_alarm[sorted_fa[:n_examples]].astype(int),
        #                 'misses': imiss[sorted_miss[:n_examples]].astype(int),
        #                 'corr_negs': icorr_neg[sorted_cn[:n_examples]].astype(int)
        #               }

        # return sorted_dict

    def avg_and_sort_contributions(self, the_dict, performance_dict=None):
        """
        Get the mean value (of data for a predictory) and contribution from
        each predictor and sort"

        Args:
        -----------
            the_dict: dictionary to process
            performance_dict: if using performance based apporach, this should be
                the dictionary with corresponding indices

        Return:

            a dictionary of mean values and contributions
        """

        return_dict = {}

        for key in list(the_dict.keys()):

            df        = the_dict[key]
            series    = df.mean(axis=0)
            sorted_df = series.reindex(series.abs().sort_values(ascending=False).index)

            if (performance_dict is None):
                idxs = self._examples.index.to_list()
            else:
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
                        "Mean Value": np.mean(self._examples.loc[idxs,var].values),
                        "Mean Contribution": series[var],
                    }

            return_dict[key] = top_vars

        return return_dict

    def tree_interpreter(self, model, indices=None):

        """
        Method for intrepreting tree based ML models using treeInterpreter.
        ADD REFERENCE HERE SOMEWHERE
=======
    def tree_interpreter(self, indices=None):

        """
        Method for intrepreting tree based ML models using treeInterpreter. 
        ADD REFERENCE HERE SOMEWHERE

        Args:
            indices: list of indices to perform interpretation over. If None, all indices
                are used

        Return:
    
            contributions_dataframe: Pandas DataFrame 
>>>>>>> master

        Args:

            model: the tree based model to use for computation
            indices: list of indices to perform interpretation over. If None, all indices
                are used

        Return:

            contributions_dataframe: Pandas DataFrame

        """

        # if indices to use is None, implies use all data
<<<<<<< HEAD
        if (indices is None):
            indices  = self._examples.index.to_list()
            examples = self._examples
        else:
            examples = self._examples.iloc[indices]
=======
        if (indices is None): indices = self._examples.index.to_list()
>>>>>>> master

        # number of examples
        n_examples = len(indices)

<<<<<<< HEAD
        # check that we have data to process
        if (n_examples == 0):
            print(f"No examples, returning empty dataframe")
            return pd.DataFrame()

        # create an instance of the TreeInterpreter class
        ti = TreeInterpreter(model, examples)

        print(f"Interpreting {n_examples} examples...")
        prediction, bias, contributions = ti.predict()

=======
        # get examples for key
        examples = self._examples.loc[indices, :]

        print(f"Interpreting {n_examples} examples...")
        prediction, bias, contributions = ti.predict(self._model, examples)

        forecast_probabilities = self._model.predict_proba(examples)[:, 1] * 100.0
>>>>>>> master
        positive_class_contributions = contributions[:, :, 1]
        positive_class_bias = bias[0,1]   #bias is all the same values for first index

        tmp_data = []

        # loop over each case appending each feature and value to a dictionary
        for i in range(n_examples):

            key_list = []
            var_list = []

            for c, feature in zip( positive_class_contributions[i, :], self._feature_names):

                key_list.append(feature)
                var_list.append(round(100.0 * c, 2))

            key_list.append('Bias')
            var_list.append(round(100. * positive_class_bias, 2))

            tmp_data.append(dict(zip(key_list, var_list)))

        # return a pandas DataFrame to do analysis on
        contributions_dataframe = pd.DataFrame(data=tmp_data)

        return contributions_dataframe

    def run_tree_interpreter(self, performance_based=False, n_examples=10):
<<<<<<< HEAD

        """
        Method for running tree interpreter. This function is called by end user

        Args:
            performance_based: string of whether to use indices based on hits, misses,
                false alarms, or correct negatives. Default is False (uses all examples
                provided during constructor call)
            n_examples : int representing the number of examples to get if
                performance_based is True. Default is 10 examples.
=======

        """
        Method for running tree interpreter. This function is called by end user

        Args:
            performance_based: string of whether to use indices based on hits, misses,
                false alarms, or correct negatives. Default is False (uses all examples 
                provided during constructor call)
            n_examples : int representing the number of examples to get if 
                performance_based is True. Default is 10 examples.

        Return:
    
            dict_of_dfs: dictionary of pandas DataFrames, one for each key
        """

        # check to make sure model is of type Tree
        if type(self._model).__name__ not in list_of_acceptable_tree_models:
            raise Exception(f"{model_name} model is not accepted for this method.")

        # will be returned; a list of pandas dataframes, one for each performance dict key
        dict_of_dfs = {}

        # if performance based interpretor, run tree_interpretor for each metric
        if performance_based is True:
            performance_dict = self.get_indices_based_on_performance(n_examples=n_examples)

            for key, values in zip(performance_dict.keys(), performance_dict.values()):
                
                print(f"Processing {key}...")

                cont_df = self.tree_interpreter(indices=values)

                dict_of_dfs[key] = cont_df
        else:
            cont_df = self.tree_interpreter()
            dict_of_dfs['all_data'] = cont_df

        return dict_of_dfs

    def compute_1d_partial_dependence(self, examples, model=None, feature=None, xdata=None, **kwargs):

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
        if model is None:
            model = self.model_set.items()

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
                predictions = model.predict_proba(copy_df)[:, 1] * 100.
            else:
                predictions = model.predict(copy_df)

            pdp_values[i] = np.mean(predictions)

        return pdp_values, xdata
>>>>>>> master

        Return:

            dict_of_dfs: dictionary of pandas DataFrames, one for each key
        """

        # will be returned; a list of pandas dataframes, one for each performance dict key
        self.ti_dict = {}

        # if performance based interpretor, run tree_interpretor for each metric
        if performance_based is True:

<<<<<<< HEAD
            # loop over each model
            for model_name, model in self._models.items():

                # check to make sure model is of type Tree
                if type(model).__name__ not in list_of_acceptable_tree_models:
                    print(f"{model_name} is not accepted for this method. Passing...")
                    continue

                # create entry for current model
                self.ti_dict[model_name] = {}
=======
    def calculate_first_order_ale(self, examples, model=None, feature=None, xdata=None):

        """
            Computes first-order ALE function on single continuous feature data.

            Parameters
            ----------
            feature : string
                The name of the feature to consider.
            xdata : array
                Quantiles of feature.
        """
        if model is None:
            model = self.model_set.items()

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
>>>>>>> master

                # get the dictionary of hits, misses, correct neg, false alarms indices
                performance_dict = self.get_indices_based_on_performance(model, n_examples=n_examples)

                for key, values in performance_dict.items():

<<<<<<< HEAD
                    print(f"Processing {key}...")
=======
            # get subset of data
            df_subset = examples[(examples[feature] >= xdata[i - 1]) & 
                                 (examples[feature] < xdata[i])]
>>>>>>> master

                    # run the tree interpreter
                    cont_dict = self.tree_interpreter(model, indices=values)

                    self.ti_dict[model_name][key] = cont_dict

                # average out the contributions and sort based on contribution
                some_dict = self.avg_and_sort_contributions(self.ti_dict[model_name],
                                                    performance_dict=performance_dict)

                self.ti_dict[model_name] = some_dict
        else:

            # loop over each model
            for model_name, model in self._models.items():

                # check to make sure model is of type Tree
                if type(model).__name__ not in list_of_acceptable_tree_models:
                    print(f"{model_name} is not accepted for this method. Passing...")
                    continue

<<<<<<< HEAD
                # create entry for current model
                self.ti_dict[model_name] = {}
=======
                if self._classification:
                    effect = 100.0 * (
                        model.predict_proba(upper_bound)[:, 1]
                        - model.predict_proba(lower_bound)[:, 1]
                    )
                else:
                    effect = model.predict(upper_bound) - model.predict(
                        lower_bound
                    )
>>>>>>> master

                # run the tree interpreter
                cont_dict = self.tree_interpreter(model)

                self.ti_dict[model_name]['all_data'] = cont_dict

                # average out the contributions and sort based on contribution
                some_dict = self.avg_and_sort_contributions(self.ti_dict[model_name])

                self.ti_dict[model_name] = some_dict

<<<<<<< HEAD
        return self.ti_dict
=======
    def compute_first_order_interpretation_curve(self, feature, compute_func, model, xdata):
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
        
        # get bootstrap indices
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
                ydata, _ = compute_func(
                    examples=examples_temp, feature=feature, xdata=xdata, model=model
                )
                ydata_set.append(ydata)
>>>>>>> master


<<<<<<< HEAD
    def plot_tree_interpreter(self, **kwargs):
=======
        else:
            ydata, xdata = compute_func(examples=self._examples, feature=feature, xdata=xdata, model=model)
>>>>>>> master

        return self._clarify_plot_obj.plot_treeinterpret(self.ti_dict, **kwargs)

    def permutation_importance(self, n_multipass_vars=5, evaluation_fn="auprc",
            subsample=1.0, njobs=1, nbootstrap=1):

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

        if evaluation_fn.lower() == "auc":
            evaluation_fn = roc_auc_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == "auprc":
            evaluation_fn = average_precision_score
            scoring_strategy = "argmin_of_mean"

<<<<<<< HEAD
=======
        print(evaluation_fn)

>>>>>>> master
        self.nbootstrap = nbootstrap

        targets = pd.DataFrame(data=self._targets, columns=['Test'])

<<<<<<< HEAD
        self.pi_dict = {}

        # loop over each model
        for model_name, model in self._models.items():

            print(f"Processing {model_name}...")

            pi_result = sklearn_permutation_importance(
                model            = model,
                scoring_data     = (self._examples, targets),
                evaluation_fn    = evaluation_fn,
                variable_names   = self._feature_names,
                scoring_strategy = scoring_strategy,
                subsample        = subsample,
                nimportant_vars  = n_multipass_vars,
                njobs            = njobs,
                nbootstrap       = self.nbootstrap,
            )

            self.pi_dict[model_name] = pi_result

        return self.pi_dict

    def plot_importance(self, **kwargs):

        return self._clarify_plot_obj.plot_variable_importance(self.pi_dict, **kwargs)
=======
        result = sklearn_permutation_importance(
            model            = self._model,
            scoring_data     = (self._examples, targets),
            evaluation_fn    = evaluation_fn,
            variable_names   = self._feature_names,
            scoring_strategy = scoring_strategy,
            subsample        = subsample,
            nimportant_vars  = n_multipass_vars,
            njobs            = njobs,
            nbootstrap       = self.nbootstrap,
        )
    
        return result
>>>>>>> master

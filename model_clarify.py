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
    Class for running various ML model interpretations.

    Args:
        model : a trained single scikit-learn model, or list of scikit-learn models, or
            dictionary of models where the key is a generic name and the value
            is a train model.
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

        # if model is of type list or single objection, convert to dictionary
        if not isinstance(model, dict):
            if isinstance(model, list):
                self._models = {type(m).__name__ : m for m in model}
            else:
                self._models = {type(model).__name__ : model}
        # user provided a dict
        else:
            self._models = model

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

        # compute pd
        self._pdp_object.run_pd(features=features, **kwargs)

        # get the final dictionary object used for plotting
        self.pd_dict = self._pdp_object.get_final_dict()

        return self.pd_dict

    def plot_pd(self, **kwargs):
        """
            Plots the PD. If the first instance is a tuple, then a 2-D plot is
            assumed, else 1-D.
        """

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
        diff_from_pos = abs(positive_class - forecast_probs_pos_class)
        diff_from_neg = abs(negative_class - forecast_probs_neg_class)

        #sort based on forecast probabilities (ascending order assumed by argsort)
        sorted_hits = np.argsort(diff_from_pos) #best hits
        sorted_miss = np.argsort(diff_from_pos)[::-1] #worst misses
        sorted_fa   = np.argsort(diff_from_neg)[::-1] #worst false alarms
        sorted_cn   = np.argsort(diff_from_neg) #best corr negs

        sorted_dict = {
                        'hits':         positive_idx[sorted_hits[:n_examples]].astype(int),
                        'misses':       positive_idx[sorted_miss[:n_examples]].astype(int),
                        'false_alarms': negative_idx[sorted_fa[:n_examples]].astype(int),
                        'corr_negs':    negative_idx[sorted_cn[:n_examples]].astype(int)
                      }

        return sorted_dict






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

        Args:

            model: the tree based model to use for computation
            indices: list of indices to perform interpretation over. If None, all indices
                are used

        Return:

            contributions_dataframe: Pandas DataFrame

        """

        # if indices to use is None, implies use all data
        if (indices is None):
            indices  = self._examples.index.to_list()
            examples = self._examples
        else:
            examples = self._examples.iloc[indices]

        # number of examples
        n_examples = len(indices)

        # check that we have data to process
        if (n_examples == 0):
            print(f"No examples, returning empty dataframe")
            return pd.DataFrame()

        # create an instance of the TreeInterpreter class
        ti = TreeInterpreter(model, examples)

        print(f"Interpreting {n_examples} examples...")
        prediction, bias, contributions = ti.predict()

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

        # will be returned; a list of pandas dataframes, one for each performance dict key
        self.ti_dict = {}

        # if performance based interpretor, run tree_interpretor for each metric
        if performance_based is True:

            # loop over each model
            for model_name, model in self._models.items():

                # check to make sure model is of type Tree
                if type(model).__name__ not in list_of_acceptable_tree_models:
                    print(f"{model_name} is not accepted for this method. Passing...")
                    continue

                # create entry for current model
                self.ti_dict[model_name] = {}

                # get the dictionary of hits, misses, correct neg, false alarms indices
                performance_dict = self.get_indices_based_on_performance(model, n_examples=n_examples)

                for key, values in performance_dict.items():

                    print(f"Processing {key}...")

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

                # create entry for current model
                self.ti_dict[model_name] = {}

                # run the tree interpreter
                cont_dict = self.tree_interpreter(model)

                self.ti_dict[model_name]['all_data'] = cont_dict

                # average out the contributions and sort based on contribution
                some_dict = self.avg_and_sort_contributions(self.ti_dict[model_name])

                self.ti_dict[model_name] = some_dict

        return self.ti_dict


    def plot_tree_interpreter(self, **kwargs):

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

        self.nbootstrap = nbootstrap

        targets = pd.DataFrame(data=self._targets, columns=['Test'])

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

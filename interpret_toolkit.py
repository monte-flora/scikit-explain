import numpy as np
import pandas as pd

from .partial_dependence import PartialDependence
from .accumulated_local_effects import AccumulatedLocalEffects
from .tree_interpreter import TreeInterpreter
from .plot import InterpretabilityPlotting
from .utils import get_indices_based_on_performance, avg_and_sort_contributions, retrieve_important_vars

from .PermutationImportance.permutation_importance import sklearn_permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score


list_of_acceptable_tree_models = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
]

class InterpretToolkit:

    """
    InterpretToolkit contains computations for various machine learning model 
    interpretations and plotting subroutines for producing publication-quality 
    figures. InterpretToolkit initialize companion classes for the computation 
    and plotting. 

    Attributes:
        model : object, list, or dict
            a trained single scikit-learn model, or list of scikit-learn models, or
            dictionary of models where the key-value pairs are the model name as 
            a string and prefit model object.
            
        examples : pandas.DataFrame or ndnumpy.array; shape = (n_examples, n_features)
            training or validation examples to evaluate.
            If ndnumpy array, make sure to specify the feature names
            
        targets: list or numpy.array 
            Target values.  
            
        classification: defaults to True for classification problems.
            Set to false otherwise.
            
        feature_names : defaults to None. Should only be set if examples is a
            nd.numpy array. Make sure it's a list
    """

    def __init__(self, model=None, examples=None, targets=None, classification=True,
            feature_names=None):

        self.check_model_attribute(model)
        self.check_target_attribute(targets)
        self.check_examples_attribute(examples)
        
        if len(np.unique(targets)) == 2 and targets is not None:
            self._classification = True
        else:
            self._classification = classification

        # initialize a PD object
        self._pdp_object = PartialDependence(model=model, examples=examples,
                                classification=classification,
                                feature_names=feature_names)

        # initialize a ALE object
        self._ale_object = AccumulatedLocalEffects(model=model, examples=examples,
                                classification=classification,
                                feature_names=feature_names)

        # initialize a plotting object
        self._clarify_plot_obj = InterpretabilityPlotting()

    def check_model_attribute(self, model):
        """
        Checks the type of the model attribute. 
        If a list or not a dict, then the model argument
        is converted to a dict for processing. 
        
        Args:
        ----------
            model : object, list, or dict 
        """
         # if model is of type list or single objection, convert to dictionary
        if not isinstance(model, dict):
            if isinstance(model, list):
                self._models = {type(m).__name__ : m for m in model}
            else:
                self._models = {type(model).__name__ : model}
        # user provided a dict
        else:
            self._models = model
    
    def check_target_attribute(self, targets):
        """
        Checks the type of the targets attribute. 
        """
         # check that targets are assigned correctly
        if isinstance(targets, list):
            self._targets = np.array(targets)
        elif isinstance(targets, np.ndarray):
            self._targets = targets
        elif isinstance(targets, (pd.DataFrame, pd.Series)):
            self._targets = targets.values
        else:
            if targets is not None:
                raise TypeError('Target variable must be numpy array or pandas.DataFrame.')
            
    def check_examples_attribute(self, examples):
        """
        Check the type of the examples attribute.
        """
        # make sure data is the form of a pandas dataframe regardless of input type
        if isinstance(examples, np.ndarray):
            if (feature_names is None):
                raise Exception('Feature names must be specified if using NumPy array.')
            else:
                self._feature_names = feature_names
                self._examples      = pd.DataFrame(data=examples, columns=feature_names)
        else:
            self._examples = examples
        
        if examples is not None:
            self._feature_names  = examples.columns.to_list()
        
    def __str__(self):

        return '{}'.format(self._models)
    
    def get_important_vars(self, results, multipass=True):
        """
        Returns the top predictors for each model from an ImportanceResults object
        """
        return retrieve_important_vars(results, multipass=True)
    
    def set_results(self, results, option):
        """ Set result dict from PermutationImportance as 
            attribute
        """
        available_options = {'permutation_importance' : 'pi_dict',
                             'pdp' : 'pd_dict',
                             'ale' : 'ale_dict',
                             'tree_interpret' : 'ti_dict'
                            }
        if option not in list(available_options.keys()):
            raise ValueError(f'{option} is not a possible option!')
        
        setattr(self, available_options[option], results)
    
    def save_figure(self, fig, 
                    fname, bbox_inches='tight', 
                    dpi=300, aformat='png'):
        """ Saves a figure """
        self._clarify_plot_obj.save_figure(fig, 
                                           fname, 
                                           bbox_inches=bbox_inches, 
                                           dpi=dpi, 
                                           aformat=aformat)
        
    def run_pd(self, features, **kwargs):
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

    def plot_pd(self, readable_feature_names={}, feature_units={}, **kwargs):
        """
            Plots the PD. If the first instance is a tuple, then a 2-D plot is
            assumed, else 1-D.
        """
        kwargs['left_yaxis_label'] = 'Mean Probability (%)'
        kwargs['wspace'] = 0.6
        kwargs['plot_type'] ='pdp'
        # plot the PD data. Use first feature key to see if 1D (str) or 2D (tuple)
        if isinstance(list(self.pd_dict.keys())[0], tuple):
            return self._clarify_plot_obj.plot_contours(self.pd_dict, 
                                                        readable_feature_names=readable_feature_names, 
                                                        feature_units=feature_units, 
                                                        **kwargs)
        else:
            return self._clarify_plot_obj.plot_1d_curve(self.pd_dict, 
                                                        readable_feature_names=readable_feature_names, 
                                                        feature_units=feature_units, 
                                                        **kwargs)

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


    def plot_ale(self, readable_feature_names={}, feature_units={}, **kwargs):
        """
            Plots the ALE. If the first instance is a tuple, then a 2-D plot is
            assumed, else 1-D.
        """
        kwargs['left_yaxis_label'] = 'Accumulated Local Effect (%)'
        kwargs['wspace'] = 0.6
        kwargs['add_zero_line'] = True
        kwargs['plot_type'] ='ale'
        # plot the PD data. Use first feature key to see if 1D (str) or 2D (tuple)
        if isinstance(list(self.ale_dict.keys())[0], tuple):
            return self._clarify_plot_obj.plot_contours(self.ale_dict, 
                                                        readable_feature_names=readable_feature_names, 
                                                        feature_units=feature_units, 
                                                        **kwargs)
        else:
            return self._clarify_plot_obj.plot_1d_curve(self.ale_dict, 
                                                        readable_feature_names=readable_feature_names, 
                                                        feature_units=feature_units, 
                                                        **kwargs)


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
                performance_dict = get_indices_based_on_performance(model, 
                                                                    examples=self._examples, 
                                                                    targets=self._targets, 
                                                                    n_examples=n_examples)

                for key, values in performance_dict.items():

                    print(f"Processing {key}...")

                    # run the tree interpreter
                    cont_dict = self.tree_interpreter(model, indices=values)

                    self.ti_dict[model_name][key] = cont_dict

                # average out the contributions and sort based on contribution
                some_dict = avg_and_sort_contributions(self.ti_dict[model_name],
                                                       examples = self._examples,
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


    def plot_tree_interpreter(self, to_only_varname=None, 
                              readable_feature_names={}, **kwargs):

        return self._clarify_plot_obj.plot_treeinterpret(self.ti_dict, 
                                                         to_only_varname=to_only_varname,
                                                         readable_feature_names=readable_feature_names, 
                                                         **kwargs)

    def permutation_importance(self, n_vars=5, evaluation_fn="auprc",
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
        elif evaluation_fn.lower() == 'bss':
            evaluation_fn = None 
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
                nimportant_vars  = n_vars,
                njobs            = njobs,
                nbootstrap       = self.nbootstrap,
            )

            self.pi_dict[model_name] = pi_result

        return self.pi_dict

    def plot_importance(self, result_dict=None, **kwargs):
        if hasattr(self, 'pi_dict'):
            result = self.pi_dict
        else:
            result = result_dict

        return self._clarify_plot_obj.plot_variable_importance(result, **kwargs)

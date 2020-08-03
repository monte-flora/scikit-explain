import numpy as np
import pandas as pd

from .attributes import Attributes
from .partial_dependence import PartialDependence
from .accumulated_local_effects import AccumulatedLocalEffects
from .tree_interpreter import TreeInterpreter
from .plot import InterpretabilityPlotting
from .utils import (
       get_indices_based_on_performance, 
     avg_and_sort_contributions, 
     retrieve_important_vars)

from .PermutationImportance.permutation_importance import sklearn_permutation_importance
from sklearn.metrics import (roc_auc_score, 
                             roc_curve, 
                             average_precision_score, 
                             brier_score_loss)


list_of_acceptable_tree_models = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
]

def brier_skill_score(target_values, forecast_probabilities):
    climo = np.mean((target_values - np.mean(target_values))**2)
    return 1.0 - brier_score_loss(target_values, forecast_probabilities) / climo

class InterpretToolkit(Attributes):

    """
    InterpretToolkit contains computations for various machine learning model 
    interpretations and plotting subroutines for producing publication-quality 
    figures. InterpretToolkit initialize companion classes for the computation 
    and plotting. 
    
    PartialDependence, AccumulatedLocalEffects are abstract base classes 
    (Abstract base classes exist to be inherited, but never instantiated).
    

    Attributes:
        model : object, list, or dict
            a trained single scikit-learn model, or list of scikit-learn models, or
            dictionary of models where the key-value pairs are the model name as 
            a string identifier and prefit model object.
            
        examples : pandas.DataFrame or ndnumpy.array; shape = (n_examples, n_features)
            training or validation examples to evaluate.
            If ndnumpy array, make sure to specify the feature names
            
        targets: list or numpy.array 
            Target values.  
            
        model_output : "predict" or "predict_proba"
            What output of the model should be evaluated. 
            
        feature_names : defaults to None. Should only be set if examples is a
            nd.numpy array. Make sure it's a list
    """

    def __init__(self, model=None, examples=None, targets=None, 
                 model_output='probability',
                 feature_names=None):

        self.set_model_attribute(model)
        self.set_target_attribute(targets)
        self.set_examples_attribute(examples, feature_names)
        
        # TODO: Check that all the models given have the requested model_output
        self.model_output = model_output 
       
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
        # initialize a plotting object
        plot_obj = InterpretabilityPlotting()
        plot_obj.save_figure(fig, 
                             fname, 
                             bbox_inches=bbox_inches, 
                             dpi=dpi, 
                             aformat=aformat)
        
    def calc_pd(self, features, nbins=25, njobs=1, subsample=1.0, nbootstrap=1):
        """
            Runs the partial dependence calculation and populates a dictionary with all
            necessary inputs for plotting.

            feature: List of strings for first-order partial dependence, or list of tuples
                     for second-order
            nbins : int
            njobs : int or float
            subsample: a float (between 0-1) for fraction of examples used in bootstrap
            nbootstrap: number of bootstrap iterations to perform. Defaults to 1 (no
                        bootstrapping).

            Return:
                dictionary of PD values for each model and feature set specified. Will be
                used for plotting.
        """
        pd_object = PartialDependence(model=self.models, 
                                      examples=self.examples,  
                                      model_output=self.model_output)
        
        results = pd_object.run_pd(features=features, 
                                   nbins=nbins, 
                                   njobs=njobs, 
                                   subsample=subsample, 
                                   nbootstrap=nbootstrap)
        self.pd_dict = results
        
        return results

    def plot_pd(self, readable_feature_names={}, feature_units={}, **kwargs):
        """
            Plots the PD. If the first instance is a tuple, then a 2-D plot is
            assumed, else 1-D.
        """
        # Check if calc_pd has been ran
        if not hasattr(self, 'pd_dict'):
            raise AttributeError('No results! Run calc_pd first!') 
        
        # initialize a plotting object
        plot_obj = InterpretabilityPlotting()
        
        kwargs['left_yaxis_label'] = 'Centered PD (%)'
        kwargs['wspace'] = 0.6
        kwargs['plot_type'] ='pdp'
        
        # plot the PD data. Use first feature key to see if 1D (str) or 2D (tuple)
        if len(list(self.pd_dict.keys())[0])> 1:
            return plot_obj.plot_contours(self.pd_dict, 
                                          readable_feature_names=readable_feature_names, 
                                          feature_units=feature_units, 
                                          **kwargs)
        else:
            return plot_obj.plot_1d_curve(self.pd_dict, 
                                         readable_feature_names=readable_feature_names, 
                                         feature_units=feature_units, 
                                         **kwargs)

    def calc_ale(self, features, nbins=25, njobs=1, subsample=1.0, nbootstrap=1):
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
        # initialize a ALE object
        ale_object = AccumulatedLocalEffects(model=self.models, 
                                      examples=self.examples,  
                                      model_output=self.model_output)
        
        results = ale_object.run_ale(features=features, 
                                     nbins=nbins, 
                                     njobs=njobs, 
                                     subsample=subsample, 
                                     nbootstrap=nbootstrap)
        self.ale_dict = results
        
        return results
        
    def plot_ale(self, readable_feature_names={}, feature_units={}, **kwargs):
        """
            Plots the ALE. If the first instance is a tuple, then a 2-D plot is
            assumed, else 1-D.
        """
        # Check if calc_pd has been ran
        if not hasattr(self, 'ale_dict'):
            raise AttributeError('No results! Run calc_ale first!')
        
        # initialize a plotting object
        plot_obj = InterpretabilityPlotting()
        
        # initialize a plotting object
        plot_obj = InterpretabilityPlotting()
        kwargs['left_yaxis_label'] = 'Accumulated Local Effect (%)'
        kwargs['wspace'] = 0.6
        kwargs['add_zero_line'] = True
        kwargs['plot_type'] ='ale'
        
        # plot the ALE data. Use first feature key to see if 1D (str) or 2D (tuple)
        if len(list(self.pd_dict.keys())[0])> 1:
            return plot_obj.plot_contours(self.ale_dict, 
                                          readable_feature_names=readable_feature_names, 
                                          feature_units=feature_units, 
                                          **kwargs)
        else:
            return plot_obj.plot_1d_curve(self.ale_dict, 
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
        available_scores = ['auc', 'aupdc', 'bss']
        
        if evaluation_fn.lower not in available_scores and scoring_strategy is None:
            raise ValueError(""" scoring_strategy is None! If you are using a user-defined
                                 evaluation_fn then scoring_strategy must be set! If a metric is positively-oriented (a higher 
                                 value is better), then set scoring_strategy = "argmin_of_mean" and if is negatively-oriented-
                                 oriented (a lower value is better), then set scoring_strategy = "argmax_of_mean"
                             """)
            
      
        
        if evaluation_fn.lower() == "auc":
            evaluation_fn = roc_auc_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == "auprc":
            evaluation_fn = average_precision_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == 'bss':
            evaluation_fn = brier_skill_score
            scoring_strategy = "argmin_of_mean"
               
        targets = pd.DataFrame(data=self.targets, columns=['Test'])

        self.pi_dict = {}

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

            self.pi_dict[model_name] = pi_result
              
        return self.pi_dict

    def plot_importance(self, result_dict=None, **kwargs):
        """
        Method for plotting the permutation importance results
        
        Args:
            result_dict : dict 
            kwargs : keyword arguments 
        """
        # initialize a plotting object
        plot_obj = InterpretabilityPlotting()
        
        if hasattr(self, 'pi_dict'):
            result = self.pi_dict
        else:
            result = result_dict
        
        if result is None:
            raise ValueError('result_dict is None! Either set it or run the .permutation_importance method first!')

        return plot_obj.plot_variable_importance(result, **kwargs)

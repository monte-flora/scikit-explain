import numpy as np
import pandas as pd

from .attributes import Attributes
from .partial_dependence import PartialDependence
from .accumulated_local_effects import AccumulatedLocalEffects
from .local_prediction import ExplainLocalPrediction
from .plot import InterpretabilityPlotting
from .utils import (
    get_indices_based_on_performance,
     retrieve_important_vars,
     brier_skill_score)

from .PermutationImportance.permutation_importance import sklearn_permutation_importance
from sklearn.metrics import (roc_auc_score, 
                             roc_curve, 
                             average_precision_score
                            )

class InterpretToolkit(Attributes):

    """
    InterpretToolkit contains computations for various machine learning model 
    interpretations and plotting subroutines for producing publication-quality 
    figures. InterpretToolkit initialize companion classes for the computation 
    and plotting. 
    
    PartialDependence, AccumulatedLocalEffects are abstract base classes 
    (Abstract base classes exist to be inherited, but never instantiated).
    

    Attributes:
        model : object, list
            a trained single scikit-learn model, or list of scikit-learn models
            
        model_names : str, list
            Names of the models (for internal and plotting purposes) 
            
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

    def __init__(self, model=None, model_names=None, 
                 examples=None, targets=None, 
                 model_output='probability',
                 feature_names=None):

        self.set_model_attribute(model, model_names)
        self.set_target_attribute(targets)
        self.set_examples_attribute(examples, feature_names)
        
        # TODO: Check that all the models given have the requested model_output
        self.model_output = model_output 
        
        self.checked_attributes = True
       
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
                             'contributions' : 'contributions_dict'
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

            feature: list of strs, or list of 2-tuples of strs
                if list of strs, computes the first-order PD for the given features
                if list of 2-tuples of strs, computes the second-order PD for the pairs of features.
                     
            nbins : int
                Number of evenly-spaced bins to compute PD
                
            njobs : int or float
            subsample: float (between 0-1) 
                fraction of examples used in bootstrap
            nbootstrap: int 
                    number of bootstrap iterations to perform. Defaults to 1 (no
                        bootstrapping).

            Return:
                dictionary of PD values for each model and feature set specified. Will be
                used for plotting.
        """
        pd_object = PartialDependence(model=self.models, 
                                      model_names=self.model_names,
                                      examples=self.examples,  
                                      model_output=self.model_output, 
                                     checked_attributes=self.checked_attributes)
        
        results = pd_object.run_pd(features=features, 
                                   nbins=nbins, 
                                   njobs=njobs, 
                                   subsample=subsample, 
                                   nbootstrap=nbootstrap)
        self.pd_dict = results
        self.features_used = features
        
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
        if isinstance( list( self.pd_dict.keys() )[0] , tuple):
            return plot_obj.plot_contours(self.pd_dict, 
                                          model_names=self.model_names,
                                          features=self.features_used,
                                          readable_feature_names=readable_feature_names, 
                                          feature_units=feature_units, 
                                          **kwargs)
        else:
            return plot_obj.plot_1d_curve(self.pd_dict, 
                                          model_names=self.model_names,
                                          features=self.features_used,
                                          readable_feature_names=readable_feature_names, 
                                          feature_units=feature_units, 
                                          **kwargs)
        
    def calc_feature_interactions(self, model_name, features, nbins=15):
        """
            Runs the Friedman's H-statistic for computing feature interactions
        """
        pd_object = PartialDependence(model=self.models, 
                                      model_names=self.model_names,
                                      examples=self.examples,  
                                      model_output=self.model_output, 
                                      checked_attributes=self.checked_attributes)
        
        results = pd_object.friedman_h_statistic(model_name, 
                                                 feature_tuple=features, 
                                                 nbins=nbins
                                                )
        return results 

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
                                             model_names=self.model_names,
                                             examples=self.examples,  
                                             model_output=self.model_output,
                                             checked_attributes=self.checked_attributes
                                            )
        
        results = ale_object.run_ale(features=features, 
                                     nbins=nbins, 
                                     njobs=njobs, 
                                     subsample=subsample, 
                                     nbootstrap=nbootstrap
                                    )
        self.ale_dict = results
        self.features_used = features
        
        return results
    
    
    def plot_ale(self, readable_feature_names={}, feature_units={}, add_shap=False, **kwargs):
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
        if isinstance( list( self.ale_dict.keys() )[0] , tuple):
            return plot_obj.plot_contours(self.ale_dict, 
                                          model_names=self.model_names,
                                          features=self.features_used,
                                          readable_feature_names=readable_feature_names, 
                                          feature_units=feature_units, 
                                          **kwargs)
        else:
            return plot_obj.plot_1d_curve(self.ale_dict, 
                                          model_names=self.model_names,
                                          features=self.features_used,
                                         readable_feature_names=readable_feature_names, 
                                         feature_units=feature_units, 
                                         **kwargs)

        
    def calc_contributions(self, method, data_for_shap=None, performance_based=False, 
                           n_examples=100, shap_sample_size=1000): 
        """
        """
        elp = ExplainLocalPrediction(model=self.models,
                            model_names=self.model_names,
                            examples=self.examples,
                            targets=self.targets,
                            model_output=self.model_output,
                            checked_attributes=self.checked_attributes         
                            )
        
        results = elp._get_local_prediction(method=method, 
                                            data_for_shap=data_for_shap,
                                            performance_based=performance_based, 
                                            n_examples=n_examples,
                                            shap_sample_size=shap_sample_size)
        
        self.contributions_dict = results
        
        return results                  
           
    def plot_contributions(self, to_only_varname=None, 
                              readable_feature_names={}, **kwargs):
        """
        
        """
        # Check if calc_pd has been ran
        if not hasattr(self, 'contributions_dict'):
            raise AttributeError('No results! Run calc_contributions first!')

        # initialize a plotting object
        plot_obj = InterpretabilityPlotting()
        
        return plot_obj.plot_contributions(self.contributions_dict, 
                                           to_only_varname=to_only_varname,
                                           readable_feature_names=readable_feature_names, 
                                           **kwargs)
        
        
    def plot_shap(self, features=None, display_feature_names=None, 
                  plot_type='summary', data_for_shap=None, subsample_size=1000, 
                  performance_based=False, n_examples=100):
        """
        """
        elp = ExplainLocalPrediction(model=self.models,
                            model_names=self.model_names,
                            examples=self.examples,
                            targets=self.targets,
                            model_output=self.model_output,
                            checked_attributes=self.checked_attributes         
                            )
        
        model = list(self.models.items())[0][1]
        
        elp.data_for_shap = data_for_shap
        if performance_based:
            performance_dict = get_indices_based_on_performance(model, 
                                                                examples=self.examples, 
                                                                targets=self.targets, 
                                                                n_examples=n_examples)
            indices = performance_dict['hits']
            examples = self.examples.iloc[indices,:]
        else:
            examples=self.examples
            
        shap_values, bias = elp._get_shap_values(model=model, 
                                                 examples=examples,
                                                 subsample_size=subsample_size)
                  
        # initialize a plotting object
        plot_obj = InterpretabilityPlotting()
        plot_obj.plot_shap(shap_values=shap_values, 
                           examples=examples, 
                           features=features, 
                           plot_type=plot_type,
                           display_feature_names=display_feature_names
                          )

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
        available_scores = ['auc', 'auprc', 'bss']
        
        if not isinstance(evaluation_fn.lower(),str) and scoring_strategy is None:
            raise ValueError(
                ''' 
                The scoring_strategy argument is None! If you are using a user-define evaluation_fn 
                then scoring_strategy must be set! If a metric is positively-oriented (a higher value is better), 
                then set scoring_strategy = "argmin_of_mean" and if is negatively-oriented-
                (a lower value is better), then set scoring_strategy = "argmax_of_mean"
                ''') 
            
        if evaluation_fn.lower() == "auc":
            evaluation_fn = roc_auc_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == "auprc":
            evaluation_fn = average_precision_score
            scoring_strategy = "argmin_of_mean"
        elif evaluation_fn.lower() == 'bss':
            evaluation_fn = brier_skill_score
            scoring_strategy = "argmin_of_mean"
        else:
            raise ValueError("evaluation_fn is not set! Available options are 'auc', 'aupdc', or 'bss'") 
                
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

        return plot_obj.plot_variable_importance(result, 
                                                 model_names=self.model_names, 
                                                 **kwargs)

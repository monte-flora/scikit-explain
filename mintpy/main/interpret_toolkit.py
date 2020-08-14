import numpy as np
import pandas as pd

# Computation imports
from ..common.attributes import Attributes
from .local_interpret import LocalInterpret
from .global_interpret import GlobalInterpret

# Plotting imports 
from ..plot.plot_interpret_curves import PlotInterpretCurves
from ..plot.plot_permutation_importance import PlotImportance
from ..plot.plot_feature_contributions import PlotFeatureContributions

from ..common.utils import (
    get_indices_based_on_performance,
    retrieve_important_vars,
    load_pickle,
    save_pickle,
    combine_top_features)

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
    
    def calc_permutation_importance(self, n_vars=5, evaluation_fn="auprc",
            subsample=1.0, njobs=1, nbootstrap=1, scoring_strategy=None ):
        """
        Performs single-pass and/or multi-pass permutation importance 

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
        global_obj = GlobalInterpret(model=self.models, 
                                      model_names=self.model_names,
                                      examples=self.examples,  
                                      targets = self.targets,
                                      model_output=self.model_output, 
                                     checked_attributes=self.checked_attributes)
        
        results = global_obj.permutation_importance(n_vars=n_vars, 
                                                    evaluation_fn=evaluation_fn,
                                                    subsample=subsample, 
                                                    njobs=njobs, 
                                                    nbootstrap=nbootstrap, 
                                                    scoring_strategy=scoring_strategy
                                                   )
        
        self.pi_dict = results
        return results                                            
                                                                      
    def _calc_interpret_curve(self, method, 
                             features, 
                             nbins=25, 
                             njobs=1, 
                             subsample=1.0, 
                             nbootstrap=1):
        """
            Runs the partial dependence or accumulated local effect calculation and 
            populates a dictionary with all necessary inputs for plotting.

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
                dictionary of PD/ALE values for each model and feature set specified. Will be
                used for plotting.
        """
        global_obj = GlobalInterpret(model=self.models, 
                                      model_names=self.model_names,
                                      examples=self.examples,  
                                      targets =self.targets,
                                      model_output=self.model_output, 
                                     checked_attributes=self.checked_attributes)
        
        results = global_obj._run_interpret_curves(method=method,
                                   features=features, 
                                   nbins=nbins, 
                                   njobs=njobs, 
                                   subsample=subsample, 
                                   nbootstrap=nbootstrap)

        return results 
        
    def calc_pd(self, features, nbins=25, njobs=1, subsample=1.0, nbootstrap=1):
        """ Alias function for user-friendly API. Runs the partial dependence calcuations.
            See _calc_interpret_curve for details. 
        """
        results = self._calc_interpret_curve(method='pd', features=features, 
                                        nbins=nbins, njobs=njobs,subsample=subsample, 
                                             nbootstrap=nbootstrap
                                        )
        self.pd_dict = results
        self.features_used = features
        
        return results
    
    def calc_ale(self, features, nbins=25, njobs=1, subsample=1.0, nbootstrap=1):
        """ Alias function for user-friendly API. Runs the accumulated local effects calcuations.
            See _calc_interpret_curve for details. 
        """
        results = self._calc_interpret_curve(method='ale', features=features, 
                                        nbins=nbins, njobs=njobs,subsample=subsample, 
                                             nbootstrap=nbootstrap
                                            )
        self.ale_dict = results
        self.features_used = features
        
        return results
    
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
    
    
    def _plot_interpret_curves(self, data, readable_feature_names={}, feature_units={}, **kwargs):
        """
        Handles 1D or 2D PD/ALE plots. 
        """
        # initialize a plotting object
        plot_obj = PlotInterpretCurves()
        
        # plot the data. Use first feature key to see if 1D (str) or 2D (tuple)
        if isinstance( list( data.keys() )[0] , tuple):
            return plot_obj.plot_contours(data, 
                                          model_names=self.model_names,
                                          features=self.features_used,
                                          readable_feature_names=readable_feature_names, 
                                          feature_units=feature_units, 
                                          **kwargs)
        else:
            return plot_obj.plot_1d_curve(data, 
                                          model_names=self.model_names,
                                          features=self.features_used,
                                          readable_feature_names=readable_feature_names, 
                                          feature_units=feature_units, 
                                          **kwargs)
        
    
    
    def plot_pd(self, readable_feature_names={}, feature_units={}, **kwargs):
        """ Alias function for user-friendly API. Runs the partial dependence plotting.
            See _plot_interpret_curves for details. 
        """
        # Check if calc_pd has been ran
        if not hasattr(self, 'pd_dict'):
            raise AttributeError('No results! Run calc_pd first!') 
        else:
            data = self.pd_dict
        
        kwargs['left_yaxis_label'] = 'Centered PD (%)'
        kwargs['wspace'] = 0.6
        
        return self._plot_interpret_curves(data, 
                               readable_feature_names=readable_feature_names, 
                               feature_units=feature_units,
                               **kwargs)

    def plot_ale(self, readable_feature_names={}, feature_units={}, add_shap=False, **kwargs):
        """ Alias function for user-friendly API. Runs the accumulated local effects plotting.
            See _plot_interpret_curves for details. 
        """
        # Check if calc_pd has been ran
        if not hasattr(self, 'ale_dict'):
            raise AttributeError('No results! Run calc_ale first!')
        else:
            data = self.ale_dict
        
        kwargs['left_yaxis_label'] = 'Accumulated Local Effect (%)'
        kwargs['wspace'] = 0.6
        
        return self._plot_interpret_curves(data, 
                               readable_feature_names=readable_feature_names, 
                               feature_units=feature_units,
                               **kwargs)

        
    def calc_contributions(self, method, data_for_shap=None, performance_based=False, 
                           n_examples=100, shap_sample_size=1000): 
        """
        Computes the individual feature contributions to a predicted outcome for 
        a series of examples. 
        
        Args:
        ------------------
            method : 'shap' or 'tree_interpreter'
                Can use SHAP or treeinterpreter to compute the feature contributions. 
                SHAP is model-agnostic while treeinterpreter can only be used on 
                select decision-tree based models in scikit-learn.
            
            data_for_shap : array (m,n) 
                Data to used to train the models. Used for the background dataset 
                to compute the expected values for the SHAP calculations. 
            
            performance_based : True or False
                If True, will average feature contributions over the best and worst 
                performing of the given examples. The number of examples to average over 
                is given by n_examples
            
            n_examples : interger
                Number of examples to compute average over if performance_based = True
            
            shap_sample_size : interger 
                Number of random samples to use for the background dataset for SHAP. 
        
        """
        local_obj = LocalInterpret(model=self.models,
                            model_names=self.model_names,
                            examples=self.examples,
                            targets=self.targets,
                            model_output=self.model_output,
                            checked_attributes=self.checked_attributes         
                            )
        
        results = local_obj._get_local_prediction(method=method, 
                                            data_for_shap=data_for_shap,
                                            performance_based=performance_based, 
                                            n_examples=n_examples,
                                            shap_sample_size=shap_sample_size)
        
        self.contributions_dict = results
        
        return results                  
           
    def plot_contributions(self, to_only_varname=None, 
                              readable_feature_names={}, **kwargs):
        """
        Plots the feature contributions. 
        """
        # Check if calc_pd has been ran
        if not hasattr(self, 'contributions_dict'):
            raise AttributeError('No results! Run calc_contributions first!')

        # initialize a plotting object
        plot_obj = PlotFeatureContributions()
        
        return plot_obj.plot_contributions(self.contributions_dict, 
                                           to_only_varname=to_only_varname,
                                           readable_feature_names=readable_feature_names, 
                                           **kwargs)
        
        
    def plot_shap(self, features=None, display_feature_names=None, 
                  plot_type='summary', data_for_shap=None, subsample_size=1000, 
                  performance_based=False, n_examples=100, **kwargs):
        """
        Plot the SHAP summary plot or dependence plots for various features.  
        """
        local_obj = LocalInterpret(model=self.models,
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
            
        shap_values, bias = local_obj._get_shap_values(model=model, 
                                                 examples=examples,
                                                 subsample_size=subsample_size)
                  
        if self.model_output == 'probability':
            shap_values *= 100.    
        
        # initialize a plotting object
        plot_obj = PlotFeatureContributions()
        plot_obj.plot_shap(shap_values=shap_values, 
                           examples=examples, 
                           features=features, 
                           plot_type=plot_type,
                           display_feature_names=display_feature_names,
                           **kwargs
                          )

    def plot_importance(self, result_dict=None, **kwargs):
        """
        Method for plotting the permutation importance results
        
        Args:
            result_dict : dict 
            kwargs : keyword arguments 
        """
        # initialize a plotting object
        plot_obj = PlotImportance()
        
        if hasattr(self, 'pi_dict'):
            result = self.pi_dict
        else:
            result = result_dict
        
        if result is None:
            raise ValueError('result_dict is None! Either set it or run the .permutation_importance method first!')

        return plot_obj.plot_variable_importance(result, 
                                                 model_names=self.model_names, 
                                                 **kwargs)

    def get_important_vars(self, results, multipass=True, combine=True, nvars=None):
        """
        Returns the top predictors for each model from an ImportanceResults object 
        as a dict with each model as the key (combine=False) or a list of important
        features as a combined list (combine=True) with duplicate top features removed.
        if combine=True, nvars can be set such that you only include a certain amount of
        top features from each model. E.g., nvars=5 and combine=True means to combine 
        the top 5 features from each model into a single list.
        """
        results = retrieve_important_vars(results, multipass=True)
        
        if not combine:
            return results
        else:
            return combine_top_features(results, nvars=nvars)
    
    def load_results(self, fnames, option, model_names):
        """ Load results of a computation (permutation importance, calc_ale, calc_pd, etc).
            and sets the data as class attribute, which is used for plotting. 
        """
        results = load_pickle(fnames=fnames)
        self.set_results(results=results, 
                          option=option
                         )
        self.model_names = model_names
        
        return results
 
    def save_results(self, fname, data):
        """ 
        Save results of a computation (permutation importance, calc_ale, calc_pd, etc)
        
        Args:
        ----------------
            fname : str
                filename to store the results in (including path)
            data : InterpretToolkit results
                the results of a InterpretToolkit calculation 
        """
        save_pickle(fname=fname,data=data)
    
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
            raise ValueError(f"""{option} is not a possible option! 
                             Possible options are {list(available_options.keys())}
                             """
                            )
        
        setattr(self, available_options[option], results)
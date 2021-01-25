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
from ..plot.plot_2D import PlotInterpret2D

from ..common.utils import (
    to_xarray,
    get_indices_based_on_performance,
    retrieve_important_vars,
    load_netcdf,
    save_netcdf,
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

    def __init__(self, models=None, model_names=None,
                 examples=pd.DataFrame(np.array([])), 
                 targets=np.array([]),
                 model_output=None,
                 feature_names=None):

        self.set_model_attribute(models, model_names)
        self.set_target_attribute(targets)
        self.set_examples_attribute(examples, feature_names)
        self.set_model_output(model_output, models)
        self.checked_attributes = True

        # Initialize a global interpret object
        self.global_obj = GlobalInterpret(models=self.models,
                                      model_names=self.model_names,
                                      examples=self.examples,
                                      targets =self.targets,
                                      model_output=self.model_output,
                                     checked_attributes=self.checked_attributes)

        # Initialize a local interpret object
        self.local_obj = LocalInterpret(models=self.models,
                            model_names=self.model_names,
                            examples=self.examples,
                            targets=self.targets,
                            model_output=self.model_output,
                            checked_attributes=self.checked_attributes
                            )
        
        self.attrs_dict = {
                      'model_output' : self.model_output,
                      'models used' : self.model_names
                     }

    def __repr__(self):
        return 'InterpretToolkit(model=%s \n \
                                 model_names=%s \n \
                                 examples=%s length:%d \n \
                                 targets=%s length:%d \n \
                                 model_output=%s \n \
                                 feature_names=%s length %d)' % \
                                 (self.models,
                                 self.model_names,
                                 type(self.examples), len(self.examples),
                                 type(self.targets), len(self.targets),
                                 self.model_output,
                                 type(self.feature_names), len(self.feature_names))

    def _append_attributes(self,ds):
        """ Append attributes to a xarray.Dataset.
        """
        for key in self.attrs_dict.keys():
            ds.attrs[key] = self.attrs_dict[key]
            
        return ds
    
    def calc_permutation_importance(self, n_vars, evaluation_fn="auprc",
            subsample=1.0, n_jobs=1, n_bootstrap=1, scoring_strategy=None, verbose=False ):
        """
        Performs single-pass and/or multi-pass permutation importance using the PermutationImportance 
        package. 

            Args:
            --------------------------------------------------------
            n_vars : integer
                number of variables to calculate the multipass permutation importance for. If n_vars=1, then 
                only the single-pass permutation importance is computed. If n_vars>1, both the single-pass 
                and multiple-pass are computed. 
            
            evaluation_fn : string or callable 
                evaluation/scoring function for evaluating the loss of skill once a feature is permuted.
                
                evaluation_fn can be set to one of the following:
                    'auc', Area under the Curve
                    'auprc', Area under the Precision-Recall Curve
                    'bss', Brier Skill Score
                    'mse', Mean Square Error
                    'norm_aupdc' Normalized Area under the Performance Diagram (Precision-Recall) Curve
                    
                 Otherwise, evaluation_fn can be any function of the following type: 
                 evaluation_fn(targets, predictions), which returns a scalar value
                 
                 However, when using your own function, then you must also set the scoring strategy (see below).
            
            subsample: float or integer (default=1.0 for no subsampling)
                if value between 0-1 interpreted as fraction of total examples to use 
                if value > 1, interpreted as the number of examples to randomly sample 
                    from the original dataset.
            
            n_jobs : interger or float (default=1; no multiprocessing)
                if integer, interpreted as the number of processors to use for multiprocessing
                if float, interpreted as the fraction of proceesors to use for multiprocessing
            
            n_bootstrap: integer (default=1 for no bootstrapping)
                number of bootstrap resamples for computing confidence intervals on the feature rankings. 
                
            scoring_strategy : string (default=None)
                If the evaluation_fn is based on a positively-oriented (a higher value is better), 
                then set scoring_strategy = "argmin_of_mean" and if is negatively-oriented-
                (a lower value is better), then set scoring_strategy = "argmax_of_mean"
                
                This argument is only required if using a non-default evaluation_fn (see above) 
            
            verbose : boolean
                True for print statements on the progress
     
            
            Returns: 
            --------------------------------------------------------
                results : dict 
                    Keys are the user-provided model names and items
                    are PermutationImportance result objects
                
        """
        results_ds = self.global_obj.permutation_importance(n_vars=n_vars,
                                                    evaluation_fn=evaluation_fn,
                                                    subsample=subsample,
                                                    n_jobs=n_jobs,
                                                    n_bootstrap=n_bootstrap,
                                                    scoring_strategy=scoring_strategy,
                                                    verbose=verbose
                                                   )
        
        self.attrs_dict['n_multipass_vars'] = n_vars
        self.attrs_dict['method'] = 'permutation_importance'
        self.attrs_dict['evaluation_fn'] = evaluation_fn
        results_ds = self._append_attributes(results_ds)
    
        self.perm_imp_ds = results_ds
        
        return results_ds

    def calc_ice(self, features, n_bins=30, n_jobs=1, subsample=1.0, n_bootstrap=1):
        """
        Compute the indiviudal conditional expectations (ICE).

            Args:
            --------------------------------------------------------
                features : string or list of strings
                    Features to compute the ICE for.  
                    
                n_bins : integer (default=30)
                    Number of bins used to compute the ICE for. 
                    
                n_jobs : float or integer (default=1)
                    if integer, interpreted as the number of processors to use for multiprocessing
                    if float, interpreted as the fraction of proceesors to use for multiprocessing
                    
                subsample : float or integer
                    if value between 0-1 interpreted as fraction of total examples to use 
                    if value > 1, interpreted as the number of examples to randomly sample 
                        from the original dataset.
                        
                n_bootstrap : integer 
                    number of bootstrap resamples for computing confidence intervals on the ICE curves.
                    
            Returns:
            --------------------------------------------------------
                results : nested dict
                    Main keys are the user-provided model names while the sub-keys 
                    are the features computed for. The items are data for the ICE curves. Also, 
                    contains X data (feature values where the ICE curves were computed) for plotting. 
        """
        results_ds = self.global_obj._run_interpret_curves(method="ice",
                            features=features,
                            n_bins=n_bins,
                            n_jobs=n_jobs,
                            subsample=subsample,
                            n_bootstrap=n_bootstrap)
        
        dimension = '2D' if isinstance( list(features)[0] , tuple) else '1D'
        self.attrs_dict['method'] = 'ice'
        self.attrs_dict['dimension'] = dimension
        
        results_ds = self._append_attributes(results_ds)
        
        self.ice_dict = results_ds
        self.feature_used=features

        return results_ds

    def calc_pd(self, features, n_bins=25, n_jobs=1, subsample=1.0, n_bootstrap=1):
        """ Alias function for user-friendly API. Runs the partial dependence calcuations.
            See _run_interpret_curves in global_interpret.py for details.
        """
        results_ds = self.global_obj._run_interpret_curves(method="pd",
                            features=features,
                            n_bins=n_bins,
                            n_jobs=n_jobs,
                            subsample=subsample,
                            n_bootstrap=n_bootstrap)
        dimension = '2D' if isinstance( list(features)[0] , tuple) else '1D'
        self.attrs_dict['method'] = 'pd'
        self.attrs_dict['dimension'] = dimension
        
        results_ds = self._append_attributes(results_ds)
        
        self.pd_ds = results_ds
        self.features_used = features
        
        return results_ds

    def calc_ale(self, features, n_bins=30, n_jobs=1, subsample=1.0, n_bootstrap=1):
        """ Alias function for user-friendly API. Runs the accumulated local effects calcuations.
            See run_interpret_curves in global_interpret.py  for details.
        """
        results_ds = self.global_obj._run_interpret_curves(method="ale",
                            features=features,
                            n_bins=n_bins,
                            n_jobs=n_jobs,
                            subsample=subsample,
                            n_bootstrap=n_bootstrap)
        
        dimension = '2D' if isinstance( list(features)[0] , tuple) else '1D'
        self.attrs_dict['method'] = 'ale'
        self.attrs_dict['dimension'] = dimension

        results_ds = self._append_attributes(results_ds)
        
        self.ale_ds = results_ds
        self.features_used = features

        return results_ds

    def calc_friedman_h_stat(self, model_name, features, n_bins=30, subsample=1.0):
        """
            Runs the Friedman's H-statistic for computing feature interactions. 
            See https://christophm.github.io/interpretable-ml-book/interaction.html 
            for details. 
            
            Only computes the interaction strength between two features. 
            Future version of MintPy will include the first-order H-statistics
            that measures the interaction between a single feature and the 
            remaining set of features. 
            
             Args:
             --------------------------------------------------------
                
                model_name : string
                    Model name, which was provided to the InterpretToolKit
             
                features : 2-tuple 
                    The two features to compute the feature interaction between
                
                n_bins : integer 
                    Number of evenly-spaced bins to compute the 
                      partial dependence functions over. 
                
              Return:
              --------------------------------------------------------
                 The second-order Friedman H-statistic (float)  
        """
        return self.global_obj.friedman_h_statistic(model_name,
                                                 feature_tuple=features,
                                                 n_bins=n_bins,
                                                 subsample=subsample
                                                )
    
    def calc_interaction_strength(self, model_name, n_bins=30, 
                                  subsample=1.0, n_jobs=1, n_bootstrap=1, 
                                  **kwargs):
        """
        
        Compute the Interaction Strength statistic from Molnar et al. (2019).
        
        
        Molnar, C., G. Casalicchio, and B. Bischl, 2019: Quantifying Model Complexity via Functional 
        Decomposition for Better Post-Hoc Interpretability. arXiv. 
        
            Args:
             --------------------------------------------------------
                
                model_name : string
                    Model name, which was provided to the InterpretToolKit
             
                features : 2-tuple 
                    The two features to compute the feature interaction between
                
                n_bins : integer 
                    Number of evenly-spaced bins to compute the 
                      partial dependence functions over. 
                      
                n_jobs : float or integer (default=1)
                    if integer, interpreted as the number of processors to use for multiprocessing
                    if float, interpreted as the fraction of proceesors to use for multiprocessing
                    
                subsample : float or integer
                    if value between 0-1 interpreted as fraction of total examples to use 
                    if value > 1, interpreted as the number of examples to randomly sample 
                        from the original dataset.
                        
                n_bootstrap : integer 
                    number of bootstrap resamples for computing confidence intervals on the ICE curves.
                
              Return:
              --------------------------------------------------------
                  The scalar interaction strength statistic (float)   
        """
        return self.global_obj.interaction_strength(model_name, 
                                                    n_bins=n_bins, 
                                                    subsample=subsample, 
                                                    n_jobs=n_jobs, 
                                                    n_bootstrap=n_bootstrap, 
                                                    **kwargs
                                                   ) 
    

    def _plot_interpret_curves(self, method, data, display_feature_names={}, display_units={}, 
                               to_probability=False, **kwargs):
        """
        FOR INTERNAL USE ONLY. 
        
        Handles 1D or 2D PD/ALE plots.
        """
        if data.attrs['dimension'] == '2D':
            plot_obj = PlotInterpret2D()
            return plot_obj.plot_contours(method=method,
                                          data=data,
                                          model_names=self.model_names,
                                          features=self.features_used,
                                          display_feature_names=display_feature_names,
                                          display_units=display_units,
                                          to_probability = to_probability,
                                          **kwargs)
        else:
            plot_obj = PlotInterpretCurves()
            return plot_obj.plot_1d_curve(method=method,
                                          data=data,
                                          model_names=self.model_names,
                                          features=self.features_used,
                                          display_feature_names=display_feature_names,
                                          display_units=display_units,
                                          to_probability = to_probability,
                                          **kwargs)

    def plot_pd(self, display_feature_names={}, display_units={}, 
                line_colors=None, to_probability=False, **kwargs):
        """ Alias function for user-friendly API. Runs the partial dependence plotting.
            
            Args:
            --------------------------------------------------------
                display_feature_names : dict 
                    For plotting purposes. Dictionary that maps the feature names 
                    in the pandas.DataFrame to display-friendly versions.
                    E.g., display_feature_names = { 'dwpt2m' : '$T_{d}$', }
                    
                    The plotting code can handle latex-style formatting. 
                
                display_units : dict 
                    For plotting purposes. Dictionary that maps the feature names
                    to their units. 
                    E.g., display_units = { 'dwpt2m' : '$^\circ$C', }
                
                to_probability : boolean 
                    If True, the values are multipled by 100. 
                
                Keyword arguments include arguments typically used for matplotlib. 

             Returns:
             --------------------------------------------------------
                 the figure 

        """
        # Check if calc_pd has been ran
        if not hasattr(self, 'pd_ds'):
            raise AttributeError('No results! Run calc_pd first!')
        else:
            data = self.pd_ds

        if data.attrs['model_output'] == 'probability':
            to_probability=True
            
        if to_probability:
            kwargs['left_yaxis_label'] = 'Centered PD (%)'
        else:
            kwargs['left_yaxis_label'] = 'Centered PD'
            
        return self._plot_interpret_curves(
                               method='pd',
                               data=data,
                               display_feature_names=display_feature_names,
                               display_units=display_units,
                               to_probability=to_probability,
                               line_colors=line_colors,            
                               **kwargs)

    def plot_ale(self, display_feature_names={}, display_units={}, 
                 line_colors=None, to_probability=False, **kwargs):
        """ Alias function for user-friendly API. Runs the accumulated local effects plotting.
            See plot_pd for details.
        """
        # Check if calc_pd has been ran
        if not hasattr(self, 'ale_ds'):
            raise AttributeError('No results! Run calc_ale first!')
        else:
            data = self.ale_ds

        if data.attrs['model_output'] == 'probability':
            to_probability=True
        
        if to_probability:
            kwargs['left_yaxis_label'] = 'Centered ALE (%)'
        else:
            kwargs['left_yaxis_label'] = 'Centered ALE'
            
        return self._plot_interpret_curves(
                               method = 'ale',
                               data=data,
                               display_feature_names=display_feature_names,
                               display_units=display_units,
                               to_probability=to_probability,
                               line_colors=line_colors,
                               **kwargs)

    def calc_contributions(self, 
                           method='shap', 
                           background_dataset=None, 
                           performance_based=False,
                           n_examples=100, ):
        """
        Computes the individual feature contributions to a predicted outcome for
        a series of examples either based on tree interpreter or 
        Shapley Additive Explanations. 

        Args:
        ------------------
            method : 'shap' or 'tree_interpreter'
                Can use SHAP or treeinterpreter to compute the feature contributions.
                SHAP is model-agnostic while treeinterpreter can only be used on
                select decision-tree based models in scikit-learn.

            background_dataset : array (m,n)
                 A representative (often a K-means or random sample) subset of the 
                 data used to train the ML model. Used for the background dataset
                 to compute the expected values for the SHAP calculations. 
                    
                 Only required for non-tree based methods. 

            performance_based : True or False
                If True, will average feature contributions over the best and worst
                performing of the given examples. The number of examples to average over
                is given by n_examples

            n_examples : interger
                Number of examples to compute average over if performance_based = True

            shap_sample_size : interger
                Number of random samples to use for the background dataset for SHAP.

        """
        results = self.local_obj._get_local_prediction(method=method,
                                            background_dataset=background_dataset,
                                            performance_based=performance_based,
                                            n_examples=n_examples,)

        self.contrib_ds = results

        return results

    def plot_contributions(self, to_only_varname=None,
                              display_feature_names={}, **kwargs):
        """
        Plots the feature contributions.
        
            Args:
            ------------------
            
            to_only_varname : callable
            
            display_feature_names : dict 
                    For plotting purposes. Dictionary that maps the feature names 
                    in the pandas.DataFrame to display-friendly versions.
                    E.g., display_feature_names = { 'dwpt2m' : '$T_{d}$', }
                    
                    The plotting code can handle latex-style formatting. 
                    
            Keyword arguments 
        
        """
        # Check if calc_pd has been ran
        if not hasattr(self, 'contrib_ds'):
            raise AttributeError('No results! Run calc_contributions first!')

        # initialize a plotting object
        plot_obj = PlotFeatureContributions()

        return plot_obj.plot_contributions(contrib_dict = self.contrib_ds[0],
                                           feature_values = self.contrib_ds[1],
                                           model_names = self.model_names,
                                           to_only_varname=to_only_varname,
                                           display_feature_names=display_feature_names,
                                           model_output=self.model_output,
                                           **kwargs)

    def calc_shap(self, background_dataset=None):
        """
        Compute the SHapley Additive Explanations (SHAP) values. The calculations starts
        with the Tree-based explainer and then defaults to the Kernel-based explainer for
        non-tree based models. If using a non-tree based models, then you must provide a 
        background dataset 
        
            Args:
            ------------------ 
                background_dataset : array (m,n)
                    A representative (often a K-means or random sample) subset of the 
                    data used to train the ML model. Used for the background dataset
                    to compute the expected values for the SHAP calculations. 
                    
                    Only required for non-tree based methods. 
                    
            Return:
            -------------------
                 shap_values : numpy.array (n_examples, n_features)
                 bias : numpy.array (1, n_features)
        """

        self.local_obj.background_dataset = background_dataset
        results = {}
        
        for model_name, model in self.models.items():
            shap_values, bias = self.local_obj._get_shap_values(model=model,
                                                 examples=self.examples,)
            results[model_name] = (shap_values, bias)
        
        return results


    def plot_shap(self, 
                  plot_type='summary',
                  shap_values=None,
                  features=None, 
                  display_feature_names=None,
                  display_units =None,
                  to_probability=False, 
                  **kwargs):
        """
        Plot the SHapley Additive Explanations (SHAP) summary plot or dependence 
        plots for various features.
        
            Args:
            ------------------ 
                
                shap_values : array (n_examples, n_features) 
                
                plot_type : 'summary' or 'dependence' 
                    if 'summary'
                    if 'dependence'
                
                features : string or list of strings (default=None)
                    features to plots if plot_type == 'dependence'.
                
                display_feature_names : dict 
                    For plotting purposes. Dictionary that maps the feature names 
                    in the pandas.DataFrame to display-friendly versions.
                    E.g., display_feature_names = { 'dwpt2m' : '$T_{d}$', }
                    
                    The plotting code can handle latex-style formatting. 
                
                display_units : dict 
                    For plotting purposes. Dictionary that maps the feature names
                    to their units. 
                    E.g., display_units = { 'dwpt2m' : '$^\circ$C', }
                    
                plot_type : 'summary' or 'dependence' 
                
                data_for_shap : array (m,n)
                    Data to used to train the models. Used for the background dataset
                    to compute the expected values for the SHAP calculations. 
                    
                feature_values : 
                
                to_probability : boolean
                    if True, values are multiplied by 100. 

        """
        examples=self.examples

        if to_probability:
            shap_values *= 100.
            
        # initialize a plotting object
        plot_obj = PlotFeatureContributions()
        plot_obj.feature_names = self.feature_names
        plot_obj.plot_shap(shap_values=shap_values,
                           examples=examples,
                           features=features,
                           plot_type=plot_type,
                           display_feature_names=display_feature_names,
                           display_units=display_units,
                           **kwargs
                          )

    def plot_importance(self, metrics_used=None, data=None, **kwargs):
        """
        Method for plotting the permutation importance results

            Args:
            ---------------------------
                data : xarray.Dataset or list of xarray.Datasets
                
                    A permutation importance result dataset or list 
                    of permutation importance result datasets 
                
                kwargs : keyword arguments
        """
        # initialize a plotting object
        plot_obj = PlotImportance()

        if hasattr(self, 'perm_imp_ds') and data is None:
            data = self.perm_imp_ds
        elif data is None:
            raise ValueError('data is None! Either set it or run the .calc_permutation_importance method first!')

        if metrics_used is not None:
            if not isinstance(metrics_used, list):
                metrics_used = [metrics_used]
            
        return plot_obj.plot_variable_importance(data,
                                                 model_output=self.model_output,
                                                 model_names=self.model_names,
                                                 metrics_used = metrics_used,
                                                 **kwargs)

    def get_important_vars(self, results, multipass=True, combine=True, n_vars=None):
        """
        Returns the top predictors for each model from an ImportanceResults object
        as a dict with each model as the key (combine=False) or a list of important
        features as a combined list (combine=True) with duplicate top features removed.
        if combine=True, nvars can be set such that you only include a certain amount of
        top features from each model. E.g., nvars=5 and combine=True means to combine
        the top 5 features from each model into a single list.
        """
        results = retrieve_important_vars(results, 
                                          model_names=self.model_names, 
                                          multipass=True)

        if not combine:
            return results
        else:
            return combine_top_features(results, n_vars=n_vars)

    def load_results(self, fnames,):
        """ Load results of a computation (permutation importance, calc_ale, calc_pd, etc).
            and sets the data as class attribute, which is used for plotting.
        """
        ### print(f'Loading results from {fnames}...')
        results = load_netcdf(fnames=fnames)
        
        self.model_output = results.attrs['model_output']
        self.model_names = results.attrs['models used']
        option = results.attrs['method']
        
        self.set_results(results=results,
                          option=option
                         )

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
        ### print(f'Saving data to {fname}...')
        save_netcdf(fname=fname,ds=data)

    def set_results(self, results, option):
        """ Set result dict from PermutationImportance as
            attribute
        """
        available_options = {'permutation_importance' : 'perm_imp_ds',
                             'pd' : 'pd_ds',
                             'ale' : 'ale_ds',
                             'contributions' : 'contrib_ds'
                            }
        if option not in list(available_options.keys()):
            raise ValueError(f"""{option} is not a possible option!
                             Possible options are {list(available_options.keys())}
                             """
                            )

        setattr(self, available_options[option], results)

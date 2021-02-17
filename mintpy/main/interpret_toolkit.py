import numpy as np
import xarray as xr
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
    combine_top_features,
    determine_feature_dtype,
    is_str
    )

class InterpretToolkit(Attributes):

    """
    InterpretToolkit contains computations for various machine learning model
    interpretations and plotting subroutines for producing publication-quality
    figures.

    Attributes:
    ------------------------------
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
        """
        Append attributes to a xarray.Dataset.
        Internal function
        """

        for key in self.attrs_dict.keys():
            ds.attrs[key] = self.attrs_dict[key]
            
        return ds
    
    def calc_permutation_importance(self, n_vars, evaluation_fn="auprc", perm_method='marginal',
            subsample=1.0, n_jobs=1, n_bootstrap=None, scoring_strategy=None, verbose=False, random_state=None ):
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
        
        n_bootstrap: integer (default=None for no bootstrapping)
            number of bootstrap resamples for computing confidence intervals on the feature rankings. 
            
        scoring_strategy : string (default=None)
            If the evaluation_fn is based on a positively-oriented (a higher value is better), 
            then set scoring_strategy = "argmin_of_mean" and if is negatively-oriented-
            (a lower value is better), then set scoring_strategy = "argmax_of_mean"
            
            This argument is only required if using a non-default evaluation_fn (see above) 
        
        random_state : int, RandomState instance, default=None
            Pseudo-random number generator to control the permutations of each
            feature.
            Pass an int to get reproducible results across function calls.
        
        perm_method : 'marginal' or 'conditional'
            Whether to use marginal- or conditional-based permutations. 
        
        verbose : boolean
            True for print statements on the progress
    
        
        Returns: 
        --------------------------------------------------------
        results : xarray.DataSet 
            Keys are the user-provided model names and items
            are PermutationImportance result objects
        """

        results_ds = self.global_obj.permutation_importance(n_vars=n_vars,
                                                    evaluation_fn=evaluation_fn,
                                                    subsample=subsample,
                                                    n_jobs=n_jobs,
                                                    n_bootstrap=n_bootstrap,
                                                    scoring_strategy=scoring_strategy,
                                                    verbose=verbose,
                                                    perm_method=perm_method,
                                                    random_state=random_state
                                                   )
        
        self.attrs_dict['n_multipass_vars'] = n_vars
        self.attrs_dict['method'] = 'permutation_importance'
        self.attrs_dict['perm_method'] = perm_method 
        self.attrs_dict['evaluation_fn'] = evaluation_fn
        results_ds = self._append_attributes(results_ds)
    
        self.perm_imp_ds = results_ds
        
        return results_ds

    def calc_ale_variance(self, ale_data=None, model_names=None, n_bins=30, n_jobs=1, subsample=1.0, n_bootstrap=1, **kwargs):
        """
        Compute the standard deviation of the ALE values for each 
        features in a dataset and then rank by the magnitude. Features 
        will a higher std(ALE) have a greater expected contribution to
        a model's prediction. 

        Args:
        -------------------------------
            
        Returns:
        --------------------------------
        """
        if model_names is None:
            model_names = self.model_names
        
        if is_str(model_names):
            model_names = [model_names]
        
        if hasattr(self, 'ale_ds') and ale_data is None:
                ale_data = self.ale_ds
        elif ale_data is not None:
            # Check that ale_data is an xarray.Dataset
            if not isinstance(ale_data, xr.core.dataset.Dataset):
                raise ValueError("""
                                 ale_data must be an xarray.Dataset, 
                                 perferably generated by mintpy.InterpretToolkit.calc_ale to be formatted correctly
                                 """)
            else:
                any_missing = all([m in ale_data.attrs['models used'] for m in model_names])
                if not any_missing:
                    raise ValueError(f'ale_data does not contain data for all the model names given!')
        else:
            raise ValueError('Must provide ale_data or compute ale for each feature using mintpy.InterpretToolkit.calc_ale')

          
        results_ds = self.global_obj.compute_ale_variance(
                            data=ale_data, model_names=model_names, 
                            n_bins=n_bins, subsample=subsample, n_bootstrap=n_bootstrap, **kwargs)
            
            
            
        self.attrs_dict['method'] = 'ale_variance'
        self.attrs_dict['models used'] = model_names
        self.attrs_dict['model output'] = 'probability' 
        self.ale_var_ds = results_ds

        results_ds = self._append_attributes(results_ds)
        self.ale_var_ds = results_ds
        
        return results_ds


    def calc_ice(self, features, n_bins=30, n_jobs=1, subsample=1.0, n_bootstrap=1):
        """
        Compute the indiviudal conditional expectations (ICE).

        Args:
        --------------------------------------------------------
        features : string or list of strings or 'all'
            Features to compute the ICE for.  if 'all', the method will compute 
            the ICE for all features. 
            
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
        results : xarray.DataSet
            Main keys are the user-provided model names while the sub-keys 
            are the features computed for. The items are data for the ICE curves. Also, 
            contains X data (feature values where the ICE curves were computed) for plotting. 
        """
        if features == 'all':
            features = self.feature_names
        results_ds = self.global_obj._run_interpret_curves(method="ice",
                            features=features,
                            n_bins=n_bins,
                            n_jobs=n_jobs,
                            subsample=subsample,
                            n_bootstrap=n_bootstrap)
        
        dimension = '2D' if isinstance(list(features)[0], tuple) else '1D'
        self.attrs_dict['method'] = 'ice'
        self.attrs_dict['dimension'] = dimension
        
        results_ds = self._append_attributes(results_ds)
        
        self.ice_dict = results_ds
        self.feature_used=features

        return results_ds

    def calc_pd(self, features, n_bins=25, n_jobs=1, subsample=1.0, n_bootstrap=1):
        """
        Runs the partial dependence (PD) calculations.

        Args:
        --------------------------------------------------------
        features : string or list of strings, or 'all
            Features to compute the PD for.  if 'all', the method will compute 
            the PD for all features. 
            
        n_bins : integer (default=25)
            Number of bins used to compute the PD for. 
            
        n_jobs : float or integer (default=1)
            if integer, interpreted as the number of processors to use for multiprocessing
            if float, interpreted as the fraction of proceesors to use for multiprocessing
            
        subsample : float or integer 
            if value between 0-1 interpreted as fraction of total examples to use 
            if value > 1, interpreted as the number of examples to randomly sample 
                from the original dataset.
                
        n_bootstrap : integer 
            number of bootstrap resamples for computing confidence intervals on the PD curves.
                
        Returns:
        --------------------------------------------------------
        results : xarray.DataSet
        """
        if features == 'all':
            features = self.feature_names
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

    def calc_ale(self, features=None, cat_features=None, n_bins=30, n_jobs=1, subsample=1.0, n_bootstrap=1):
        """
        Runs the accumulated local effects (ALE) calculations.

        Args:
        --------------------------------------------------------
        features : string or list of strings or 'all'
            Features to compute the ALE for.  if 'all', the method will compute 
            the ALE for all features. 
            
        n_bins : integer (default=30)
            Number of bins used to compute the ALE for. 
            
        n_jobs : float or integer (default=1)
            if integer, interpreted as the number of processors to use for multiprocessing
            if float, interpreted as the fraction of proceesors to use for multiprocessing
            
        subsample : float or integer
            if value between 0-1 interpreted as fraction of total examples to use 
            if value > 1, interpreted as the number of examples to randomly sample 
                from the original dataset.
                
        n_bootstrap : integer 
            number of bootstrap resamples for computing confidence intervals on the ALE curves.
                
        Returns:
        --------------------------------------------------------
        results : xarray.DataSet
        """
        if features == 'all':
            features = self.feature_names
            
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
        
        features : 2-tuple of string
            The two features to compute the feature interaction between
        
        n_bins : integer (default=30)
            Number of evenly-spaced bins to compute the partial dependence 
            functions over. 

        subsample : float or integer (default 1.0)
            if value between 0-1 interpreted as fraction of total examples to use 
            if value > 1, interpreted as the number of examples to randomly sample 
                from the original dataset.

        Return:
        --------------------------------------------------------
        The second-order Friedman H-statistic (float)  
        """

        return self.global_obj.friedman_h_statistic(model_name,
                                                 feature_tuple=features,
                                                 n_bins=n_bins,
                                                 subsample=subsample
                                                )
    
    def calc_interaction_strength(self, ale_data=None, model_names=None, n_bins=30, 
                                  n_jobs=1, subsample=1.0, n_bootstrap=1, 
                                  **kwargs):
        """
        Compute the InterAction Strength (IAS) statistic from Molnar et al. (2019).

        Molnar, C., G. Casalicchio, and B. Bischl, 2019: Quantifying Model Complexity via Functional 
        Decomposition for Better Post-Hoc Interpretability. arXiv. 
        
        Args:
        --------------------------------------------------------
        
        ale_data : xarray.Dataset (default=None)
            pre-computed ALE curves for each feature; can be computed with calc_ale (see above)
            if None, the ALE curves are calculated internally for each feature. 
            
        model_names : string or list of strings
            Model name, which was provided to the InterpretToolKit
            The models for which to compute the IAS.  

        n_bins : integer (default=30)
            Number of evenly-spaced bins to compute the partial dependence functions over. 
                
        n_jobs : float or integer (default=1)
            if integer, interpreted as the number of processors to use for multiprocessing
            if float, interpreted as the fraction of proceesors to use for multiprocessing
            
        subsample : float or integer (default=1.0)
            if value between 0-1 interpreted as fraction of total examples to use 
            if value > 1, interpreted as the number of examples to randomly sample 
                from the original dataset.
                
        n_bootstrap : integer (default 1)
            number of bootstrap resamples for computing confidence intervals on the ICE curves.
                
        Return:
        --------------------------------------------------------
        The scalar interaction strength statistic (float)   
        """
        if model_names is None:
            model_names = self.model_names
        else:
            if is_str(model_names):
                model_names = [model_names]

        if hasattr(self, 'ale_ds') and ale_data is None:
                ale_data = self.ale_ds
        elif ale_data is not None:
            # Check that ale_data is an xarray.Dataset
            if not isinstance(ale_data, xr.core.dataset.Dataset):
                raise ValueError('ale_data must be an xarray.Dataset, perferably generated by mintpy.InterpretToolkit.calc_ale to be formatted correctly')
            else:
                # Check that ale_data actually has data for all the requested model names 
                if collections.Counter(model_names) != collections.Counter(ale_data.attrs['models used']):
                    raise ValueError(f'ale_data does not contain data for all the model names ({model_names}) given!')
        else:
            raise ValueError(f'Must provide ale_data or compute ale for each feature using mintpy.InterpretToolkit.calc_ale') 

        return self.global_obj.compute_scalar_interaction_stats(
                                                    method = 'ias',
                                                    data=ale_data,
                                                    model_names=model_names, 
                                                    n_bins=n_bins, 
                                                    subsample=subsample, 
                                                    n_jobs=n_jobs, 
                                                    n_bootstrap=n_bootstrap, 
                                                    **kwargs
                                                   ) 
    

    def _plot_interpret_curves(self, method, data, features=None, display_feature_names={}, display_units={}, 
                               to_probability=False, **kwargs):
        """
        FOR INTERNAL USE ONLY. 
        
        Handles 1D or 2D PD/ALE plots.
        """
        if features is None:
            try:
                features = self.features_used
            except:
                raise ValueError('No features were provided to plot!')
        else:
            if is_str(features):
                features=[features]
                
        if data.attrs['dimension'] == '2D':
            plot_obj = PlotInterpret2D()
            return plot_obj.plot_contours(method=method,
                                          data=data,
                                          model_names=self.model_names,
                                          features=features,
                                          display_feature_names=display_feature_names,
                                          display_units=display_units,
                                          to_probability = to_probability,
                                          **kwargs)
        else:
            plot_obj = PlotInterpretCurves()
            return plot_obj.plot_1d_curve(method=method,
                                          data=data,
                                          model_names=self.model_names,
                                          features=features,
                                          display_feature_names=display_feature_names,
                                          display_units=display_units,
                                          to_probability = to_probability,
                                          **kwargs)

    def plot_pd(self, data=None, features=None, display_feature_names={}, display_units={}, 
                line_colors=None, to_probability=False, **kwargs):
        """
        Runs the partial dependence plotting.
        
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
        
        line_colors : str or list of strs equal to number of models
            User-defined colors for curve plotting.

        to_probability : boolean 
            If True, the values are multipled by 100. 
        
        Keyword arguments include arguments typically used for matplotlib. 

        Returns:
        --------------------------------------------------------
        fig: matplotlib figure instance
        """

        # Check if calc_pd has been ran
        if not hasattr(self, 'pd_ds') and data is None:
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
                               features=features,
                               display_feature_names=display_feature_names,
                               display_units=display_units,
                               to_probability=to_probability,
                               line_colors=line_colors,            
                               **kwargs)

    def plot_ale(self, data=None, features=None, display_feature_names={}, display_units={}, 
                 line_colors=None, to_probability=False, **kwargs):
        """
        Runs the accumulated local effects plotting.
        
        Args:
        ------------------------
        display_feature_names : dict 
            For plotting purposes. Dictionary that maps the feature names 
            in the pandas.DataFrame to display-friendly versions.
            E.g., display_feature_names = { 'dwpt2m' : '$T_{d}$', }
            
            The plotting code can handle latex-style formatting. 
        
        display_units : dict 
            For plotting purposes. Dictionary that maps the feature names
            to their units. 
            E.g., display_units = { 'dwpt2m' : '$^\circ$C', }

        line_colors : str or list of strs equal to number of models
            User-defined colors for curve plotting.

        to_probability : boolean 
            If True, the values are multipled by 100. 
        
        Keyword arguments include arguments typically used for matplotlib.

        Returns:
        -----------------------
        fig: matplotlib figure instance
        """

        # Check if calc_pd has been ran
        if not hasattr(self, 'ale_ds') and data is None:
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
                               features=features,
                               display_feature_names=display_feature_names,
                               display_units=display_units,
                               to_probability=to_probability,
                               line_colors=line_colors,
                               **kwargs)

    def calc_contributions(self, 
                           method='shap', 
                           background_dataset=None, 
                           performance_based=False,
                           n_examples=100 ):
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

        performance_based : boolean (default=False)
            If True, will average feature contributions over the best and worst
            performing of the given examples. The number of examples to average over
            is given by n_examples

        n_examples : interger (default=100)
            Number of examples to compute average over if performance_based = True

        Return:
        -------------------
        results : nested dictionary for plotting purposes
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

        Keyword arguments include arguments typically used for matplotlib

        Returns:
        -----------------------
        fig: matplotlib figure instance
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
        results : dict
            Dictionary where the keys represent model names, and the
            values represent a tuple of shap values and bias.
            shap_values is of type numpy.array (n_examples, n_features)
            bias is of type numpy.array (1, n_features)
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

        plot_type : 'summary' or 'dependence' 
            if 'summary'
            if 'dependence'

        shap_values : array (n_examples, n_features) 
        
        features : string or list of strings (default=None)
            features to plots if plot_type is 'dependence'.
        
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
            if True, values are multiplied by 100. 

        Returns:
        -----------------------
        fig: matplotlib figure instance
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

    def plot_importance(self, method='multipass', 
                        xlabels=None, ylabels=None, metrics_used=None, 
                        data=None, model_names=None, plot_correlated_features=False,
                        **kwargs):
        """
        Method for plotting the permutation importance results

        Args:
        ---------------------------
        method: 'ale_variance', 'multipass' or 'singlepass'
            Method used to compute the feature rankings. 
                
        data : xarray.Dataset or list of xarray.Datasets
            A permutation importance result dataset or list 
            of permutation importance result datasets 
        
        kwargs : keyword arguments
            multipass : boolen (defaults to False)
                If True, will plot multi-pass (i.e., Lakshmanan) results.
                Otherwise Briemann
            num_vars_to_plot : integer
                Number of features to plot from permutation importance calculation.

        Returns:
        -----------------------
        fig: matplotlib figure instance
        """
        if method == 'ale_variance':
            metrics_used=['$\sigma$(ALE)']
        
        model_output = kwargs.get('model_output', self.model_output)
        kwargs.pop('model_output', None)

        # initialize a plotting object
        plot_obj = PlotImportance()

        if hasattr(self, 'perm_imp_ds') and data is None and method != 'ale_variance':
            data = self.perm_imp_ds
        elif method == 'ale_variance' and hasattr(self, 'ale_var_ds') and data is None:
            data = self.ale_var_ds
        elif data is None:
            raise ValueError("""
                             data is None! Either set it or run either the 
                             .calc_permutation_importance or .calc_ale_variance methods first!
                             """)

        if is_str(metrics_used):
            metrics_used=[metrics_used]
            
        if xlabels is None and metrics_used is None and ylabels is None:
            metrics_used = [data.attrs['evaluation_fn'].replace("_", "").upper()]
        else:
            xlabels = ['']
            
        if model_names is not None:
            if is_str(model_names):
                model_names = [model_names]            
        else:
            model_names=self.model_names
            
        if plot_correlated_features:
            kwargs['examples'] = self.examples
            
        
            
            
        return plot_obj.plot_variable_importance(data,
                                                method=method, 
                                                model_output=model_output,
                                                model_names=model_names,
                                                metrics_used = metrics_used,
                                                xlabels = xlabels,
                                                ylabels=ylabels,
                                                plot_correlated_features=plot_correlated_features,
                                                 **kwargs)

    def get_important_vars(self, results, multipass=True, n_vars=10, combine=False):
        """
        Retrieve the most important variables from permutation importance.

        Args:
        ----------------
        results : permutation importance results

        multipass : boolean (defaults to True)

        n_vars : integer (default=10)
            Number of variables to retrieve.

        combine : boolean  (default=False)
            If combine=True, nvars can be set such that you only include a certain amount of
            top features from each model. E.g., nvars=5 and combine=True means to combine
            the top 5 features from each model into a single list.

        Return:
        -------------------
        Returns the top predictors for each model from an ImportanceResults object
        as a dict with each model as the key (combine=False) or a list of important
        features as a combined list (combine=True) with duplicate top features removed.
        """
        results = retrieve_important_vars(results, 
                                          model_names=self.model_names, 
                                          multipass=True)

        if not combine:
            return results
        else:
            return combine_top_features(results, n_vars=n_vars)

    def load_results(self, fnames,):
        """
        Load results of a computation (permutation importance, calc_ale, calc_pd, etc)
            and sets the data as class attribute, which is used for plotting.

        Args:
        ----------------
        fnames : str
            File names for loading

        Return:
        -------------------
        results : xarray.DataSet
            results xarray dataset for plotting purposes
        """

        results = load_netcdf(fnames=fnames)
        
        for s in [self, self.global_obj, self.local_obj]:
            setattr(s, 'model_output', results.attrs['model_output'])
            model_names = [results.attrs['models used']]
            if not isinstance(model_names, list):
                model_names = [model_names]

            if (any(isinstance(i, list) for i in model_names)):
                model_names = model_names[0] 
            
            setattr(s, 'model_names', model_names)
            setattr(s, 'models used', model_names) 

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

        save_netcdf(fname=fname,ds=data)

    def set_results(self, results, option):
        """
        Set result dict from PermutationImportance as attribute

        Args:
        ----------------
        results : xarray.DataSet

        option : str. Valid options are 'permutation_importance', 'pd',
            'ale', 'contributions'
        """

        available_options = {'permutation_importance' : 'perm_imp_ds',
                             'pd' : 'pd_ds',
                             'ale' : 'ale_ds',
                             'contributions' : 'contrib_ds',
                             'ice' : 'ice_ds',
                             'ale_variance' : 'ale_var_ds'
                             }
        
        if option not in list(available_options.keys()):
            raise ValueError(f"""{option} is not a possible option!
                             Possible options are {list(available_options.keys())}
                             """
                            )

        setattr(self, available_options[option], results)

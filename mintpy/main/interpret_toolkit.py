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
                 model_output=None,
                 feature_names=None):

        self.set_model_attribute(model, model_names)
        self.set_target_attribute(targets)
        self.set_examples_attribute(examples, feature_names)

        self.set_model_output(model_output, model)

        self.checked_attributes = True

        # Initialize a global interpret object
        self.global_obj = GlobalInterpret(model=self.models,
                                      model_names=self.model_names,
                                      examples=self.examples,
                                      targets =self.targets,
                                      model_output=self.model_output,
                                     checked_attributes=self.checked_attributes)

        # Initialize a local interpret object
        self.local_obj = LocalInterpret(model=self.models,
                            model_names=self.model_names,
                            examples=self.examples,
                            targets=self.targets,
                            model_output=self.model_output,
                            checked_attributes=self.checked_attributes
                            )

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

        results = self.global_obj.permutation_importance(n_vars=n_vars,
                                                    evaluation_fn=evaluation_fn,
                                                    subsample=subsample,
                                                    njobs=njobs,
                                                    nbootstrap=nbootstrap,
                                                    scoring_strategy=scoring_strategy
                                                   )

        self.pi_dict = results
        return results

    def calc_ice(self, features, nbins=30, njobs=1, subsample=1.0, nbootstrap=1):
        """
        Compute the indiviudal conditional expectations.

        Args:
            feautres,
            nbins,
            njobs,
            subsample,
            nbootstrap,
            subsample_size=100

        """

        results = self.global_obj._run_interpret_curves(method="ice",
                            features=features,
                            nbins=nbins,
                            njobs=njobs,
                            subsample=subsample,
                            nbootstrap=nbootstrap)

        self.ice_dict = results
        self.feature_used=features

        return results

    def calc_pd(self, features, nbins=25, njobs=1, subsample=1.0, nbootstrap=1):
        """ Alias function for user-friendly API. Runs the partial dependence calcuations.
            See _calc_interpret_curve for details.
        """

        results = self.global_obj._run_interpret_curves(method="pd",
                            features=features,
                            nbins=nbins,
                            njobs=njobs,
                            subsample=subsample,
                            nbootstrap=nbootstrap)

        self.pd_dict = results
        self.features_used = features
        return results

    def calc_ale(self, features, nbins=25, njobs=1, subsample=1.0, nbootstrap=1):
        """ Alias function for user-friendly API. Runs the accumulated local effects calcuations.
            See _calc_interpret_curve for details.
        """

        results = self.global_obj._run_interpret_curves(method="ale",
                            features=features,
                            nbins=nbins,
                            njobs=njobs,
                            subsample=subsample,
                            nbootstrap=nbootstrap)

        self.ale_dict = results
        self.features_used = features

        return results

    def calc_feature_interactions(self, model_name, features, nbins=15):
        """
            Runs the Friedman's H-statistic for computing feature interactions
        """

        results = self.global_obj.friedman_h_statistic(model_name,
                                                 feature_tuple=features,
                                                 nbins=nbins
                                                )
        return results

    def _plot_interpret_curves(self, data, display_feature_names={}, display_units={}, **kwargs):
        """
        Handles 1D or 2D PD/ALE plots.
        """
        # plot the data. Use first feature key to see if 1D (str) or 2D (tuple)
        if isinstance( list( data.keys() )[0] , tuple):
            plot_obj = PlotInterpret2D()
            return plot_obj.plot_contours(data,
                                          model_names=self.model_names,
                                          features=self.features_used,
                                          display_feature_names=display_feature_names,
                                          display_units=display_units,
                                          **kwargs)
        else:
            plot_obj = PlotInterpretCurves()
            return plot_obj.plot_1d_curve(data,
                                          model_names=self.model_names,
                                          features=self.features_used,
                                          display_feature_names=display_feature_names,
                                          display_units=display_units,
                                          **kwargs)

    def plot_pd(self, display_feature_names={}, display_units={}, to_probability=False, **kwargs):
        """ Alias function for user-friendly API. Runs the partial dependence plotting.
            See _plot_interpret_curves for details.
        """
        # Check if calc_pd has been ran
        if not hasattr(self, 'pd_dict'):
            raise AttributeError('No results! Run calc_pd first!')
        else:
            data = self.pd_dict

        kwargs['left_yaxis_label'] = 'Centered PD (%)'

        return self._plot_interpret_curves(data,
                               display_feature_names=display_feature_names,
                               display_units=display_units,
                               to_probability=to_probability,
                               **kwargs)

    def plot_ale(self, display_feature_names={}, display_units={}, add_shap=False, to_probability=False, **kwargs):
        """ Alias function for user-friendly API. Runs the accumulated local effects plotting.
            See _plot_interpret_curves for details.
        """
        # Check if calc_pd has been ran
        if not hasattr(self, 'ale_dict'):
            raise AttributeError('No results! Run calc_ale first!')
        else:
            data = self.ale_dict

        kwargs['left_yaxis_label'] = 'Accumulated Local Effect (%)'

        return self._plot_interpret_curves(data,
                               display_feature_names=display_feature_names,
                               display_units=display_units,
                               to_probability=to_probability,
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

        results = self.local_obj._get_local_prediction(method=method,
                                            data_for_shap=data_for_shap,
                                            performance_based=performance_based,
                                            n_examples=n_examples,
                                            shap_sample_size=shap_sample_size)

        self.contributions_dict = results

        return results

    def plot_contributions(self, to_only_varname=None,
                              display_feature_names={}, **kwargs):
        """
        Plots the feature contributions.
        """
        # Check if calc_pd has been ran
        if not hasattr(self, 'contributions_dict'):
            raise AttributeError('No results! Run calc_contributions first!')

        # initialize a plotting object
        plot_obj = PlotFeatureContributions()

        return plot_obj.plot_contributions(self.contributions_dict,
                                           self.model_names,
                                           to_only_varname=to_only_varname,
                                           display_feature_names=display_feature_names,
                                           **kwargs)

    def calc_shap(self, data_for_shap=None, subsample_size=100, **kwargs):
        """
        Compute the SHAP values.
        """

        model = list(self.models.items())[0][1]
        local_obj.data_for_shap = data_for_shap
        shap_values, bias = self.local_obj._get_shap_values(model=model,
                                                 examples=self.examples,
                                                 subsample_size=subsample_size)
        return shap_values


    def plot_shap(self, shap_values=None,
                  features=None, display_feature_names=None,
                  plot_type='summary', data_for_shap=None, subsample_size=1000,
                  performance_based=False, n_examples=100, feature_values=None, **kwargs):
        """
        Plot the SHAP summary plot or dependence plots for various features.
        """
        if shap_values is None:

            model = list(self.models.items())[0][1]

            self.local_obj.data_for_shap = data_for_shap
            if performance_based:
                performance_dict = get_indices_based_on_performance(model,
                                                                examples=self.examples,
                                                                targets=self.targets,
                                                                n_examples=n_examples)
                indices = performance_dict['hits']
                examples = self.examples.iloc[indices,:]
            else:
                examples=self.examples

            shap_values, bias = self.local_obj._get_shap_values(model=model,
                                                 examples=examples,
                                                 subsample_size=subsample_size)

        else:
            examples=self.examples

        if self.model_output == 'probability':
            shap_values *= 100.
        # initialize a plotting object
        plot_obj = PlotFeatureContributions()
        plot_obj.feature_names = self.feature_names
        plot_obj.plot_shap(shap_values=shap_values,
                           examples=examples,
                           features=features,
                           plot_type=plot_type,
                           display_feature_names=display_feature_names,
                           feature_values=feature_values,
                           **kwargs
                          )

    def plot_importance(self, result_dict=None, **kwargs):
        """
        Method for plotting the permutation importance results

        Args:
            result_dict : dict or list of dicts
            kwargs : keyword arguments
        """
        # initialize a plotting object
        plot_obj = PlotImportance()

        if hasattr(self, 'pi_dict') and result_dict is None:
            result = self.pi_dict
        elif result_dict is None:
            raise ValueError('result_dict is None! Either set it or run the .calc_permutation_importance method first!')
        else:
            result = result_dict

        return plot_obj.plot_variable_importance(result,
                                                 model_output=self.model_output,
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
        print(f'Loading results from {fnames}...')
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
        print(f'Saving data to {fname}...')
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

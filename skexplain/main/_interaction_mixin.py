import warnings
import xarray as xr

from ..common.utils import check_all_features_for_ale
from ._validation import normalize_estimator_names


class InteractionMixin:
    """Mixin providing feature-interaction methods for ExplainToolkit."""

    def perm_based_interaction(
        self,
        features,
        evaluation_fn,
        estimator_names=None,
        n_jobs=1,
        subsample=1.0,
        n_bootstrap=1,
        verbose=False,
    ):
        """
        Compute the performance-based feature interactions from Oh (2019) [6]_.
        For a pair of features, the loss of skill is recorded for permuting
        each feature separately and permuting both. If there is no feature interaction
        and the covariance between the two features is close to zero, the sum of the
        individual losses will approximately equal the loss of skill from permuting
        both features. Otherwise, a non-zero difference indicates some interaction.
        The differences for different pairs of features can be used to rank the
        strength of any feature interactions.

        References
        -------------
        .. [6]  Oh, Sejong, 2019. Feature Interaction in Terms of Prediction Performance
            https://www.mdpi.com/2076-3417/9/23/5191


        Parameters
        -----------

        features : list of 2-tuple of strings
            Pairs of features to compute the interaction strength for.

        evaluation_fn : string or callable
            evaluation/scoring function for evaluating the loss of skill once a feature is permuted.
            evaluation_fn can be set to one of the following strings:

                - ``"auc"``, Area under the Curve
                - ``"auprc"``, Area under the Precision-Recall Curve
                - ``"bss"``, Brier Skill Score
                - ``"mse"``, Mean Square Error
                - ``"norm_aupdc"``,  Normalized Area under the Performance Diagram (Precision-Recall) Curve

            Otherwise, evaluation_fn can be any function of form,
            `evaluation_fn(targets, predictions)` and must return a scalar value

        estimator_names : string, list of strings

            If using multiple estimators, you can pass a single (or subset of) estimator name(s)
            to compute for.

        subsample: float or integer (default=1.0 for no subsampling)

            - if value is between 0-1, it is interpreted as fraction of total X to use
            - if value > 1, interpreted as the absolute number of random samples of X.

        n_jobs : interger or float (default=1; no multiprocessing)

            - if integer, interpreted as the number of processors to use for multiprocessing
            - if float between 0-1, interpreted as the fraction of proceesors to use for multiprocessing

        n_bootstrap: integer (default=None for no bootstrapping)
            Number of bootstrap resamples for computing confidence intervals on the feature pair rankings.

        Returns
        ---------

        results_ds : xarray.Dataset
            Permutation importance-based feature interaction strength results


        Examples
        ---------
        >>> import skexplain
        >>> # pre-fit estimators within skexplain
        >>> estimators = skexplain.load_models()
        >>> X, y = skexplain.load_data() # training data
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators,
        ...                             X=X,
        ...                             y=y,
        ...                            )
        >>> important_vars = ['sfc_temp', 'temp2m', 'sfcT_hrs_bl_frez', 'tmp2m_hrs_bl_frez',
        ...      'uplwav_flux']
        >>> important_vars_2d = list(itertools.combinations(important_vars, r=2))
        >>> perm_based_interact_ds = explainer.perm_based_interaction(
        ...                          important_vars_2d, evaluation_fn='norm_aupdc',
        ...                         )
        """
        estimator_names = normalize_estimator_names(estimator_names, self.estimator_names)

        results_ds = self.global_obj.compute_interaction_rankings_performance_based(
            estimator_names,
            features,
            evaluation_fn=evaluation_fn,
            estimator_output=self.estimator_output,
            subsample=subsample,
            n_bootstrap=n_bootstrap,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.attrs_dict["method"] = "perm_based"
        self.attrs_dict["estimators used"] = estimator_names
        self.attrs_dict["estimator output"] = self.estimator_output
        self.attrs_dict["evaluation_fn"] = "Interaction Importance"

        results_ds = self._append_attributes(results_ds)

        return results_ds

    def friedman_h_stat(
        self, dataset_1d=None, dataset_2d=None, features=None, estimator_names=None, **kwargs
    ):
        """
        Compute the second-order Friedman's H-statistic for computing feature interactions [11]_ [12]_.
        Based on equation (44) from Friedman and Popescu (2008) [12]_. Only computes the interaction strength
        between two features. In future versions of skexplain we hope to include the first-order H-statistics
        that measure the interaction between a single feature and the
        remaining set of features. This statistic can be computed from both the accumulated local effects
        and partial dependence.

        References
        -----------

        .. [11] https://christophm.github.io/interpretable-ml-book/interaction.html
        .. [12] Friedman, J. H., and B. E. Popescu, 2008: Predictive learning via rule ensembles.
                Ann Appl Statistics, 2, 916-954, https://doi.org/10.1214/07-aoas148.


        Parameters
        -----------

        dataset_1d : xarray.Dataset
            1D partial dependence or accumulated local effect dataset.
            Results of :func:`~ExplainToolkit.pd` or :func:`~ExplainToolkit.ale` for ``features``

        dataset_2d : xarray.Dataset
            2D partial dependence or accumulated local effects dataset.
            Results of :func:`~ExplainToolkit.pd` or :func:`~ExplainToolkit.ale`, but 2-tuple combinations
            of ``features``.

        features : list of 2-tuples of strings
            The pairs of features to compute the feature interaction between.

        estimator_names : string, list of strings (default is None)

            If using multiple estimators, you can pass a single (or subset of) estimator name(s)
            to compute the H-statistic for.

        Returns
        ----------

        results_ds : xarray.Dataset
            The second-order Friedman H-statistic for all estimators.

        Examples
        ---------
        >>> import skexplain
        >>> # pre-fit estimators within skexplain
        >>> estimators = skexplain.load_models()
        >>> X, y = skexplain.load_data() # training data
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators
        ...                             X=X,
        ...                             y=y,
        ...                            )
        >>> ale_1d = explainer.ale(features='all')
        >>> ale_2d = explainer.ale(features='all_2d')
        >>> hstat = explainer.friedman_h_stat(ale_1d, ale_2d,)
        """
        estimator_names = normalize_estimator_names(estimator_names, self.estimator_names)

        # Check if old arguments are provided
        old_arg_1d = kwargs.get("pd_1d", None)
        old_arg_2d = kwargs.get("pd_2d", None)

        if old_arg_1d is not None:
            warnings.warn(
                "'pd_1d' argument is deprecated and will be removed in future versions. Use 'dataset_1d' instead.",
                DeprecationWarning,
            )
            if dataset_1d is None:
                dataset_1d = old_arg_1d

        if old_arg_2d is not None:
            warnings.warn(
                "'pd_2d' argument is deprecated and will be removed in future versions. Use 'dataset_2d' instead.",
                DeprecationWarning,
            )
            if dataset_2d is None:
                dataset_2d = old_arg_2d

        # Check if the new arguments are provided
        if dataset_1d is None or dataset_2d is None or features is None:
            raise ValueError(
                "Please provide the necessary arguments: 'dataset_1d', 'dataset_2d', and 'features'."
            )

        results_ds = self.global_obj.compute_scalar_interaction_stats(
            method="hstat",
            data=dataset_1d,
            data_2d=dataset_2d,
            features=features,
            estimator_names=estimator_names,
            **kwargs,
        )

        results_ds = self._append_attributes(results_ds)

        return results_ds

    def interaction_strength(self, ale, estimator_names=None, **kwargs):
        """
        Compute the InterAction Strength (IAS) statistic from Molnar et al. (2019) [5]_.
        The IAS varies between 0-1 where values closer to 0 indicate no feature interaction
        strength.

        Parameters
        ------------

        ale : xarray.Dataset

            Results of :func:`~ExplainToolkit.ale`, but must be computed for all features

        estimator_names : string, list of strings (default is None)

            If using multiple estimators, you can pass a single (or subset of) estimator name(s)
            to compute the IAS for.

        kwargs : dict

            - subsample
            - n_bootstrap
            - estimator_output

        Returns
        ----------

        results_ds : xarray.Dataset
            Interaction strength result dataset

        Examples
        ---------
        >>> import skexplain
        >>> # pre-fit estimators within skexplain
        >>> estimators = skexplain.load_models()
        >>> X, y = skexplain.load_data() # training data
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators
        ...                             X=X,
        ...                             y=y,
        ...                            )
        >>> ale = explainer.ale(features='all')
        >>> ias = explainer.interaction_strength(ale)
        """
        estimator_names = normalize_estimator_names(estimator_names, self.estimator_names)

        check_all_features_for_ale(ale, estimator_names, self.feature_names)

        # Check that ale_data is an xarray.Dataset
        if not isinstance(ale, xr.core.dataset.Dataset):
            raise ValueError(
                """
                                 ale must be an xarray.Dataset,
                                 perferably generated by mintpy.ExplainToolkit.calc_ale to be formatted correctly
                                 """
            )
        else:
            any_missing = all([m in ale.attrs["estimators used"] for m in estimator_names])
            if not any_missing:
                raise ValueError(f"ale does not contain data for all the estimator names given!")

        kwargs["estimator_output"] = self.estimator_output

        results_ds = self.global_obj.compute_scalar_interaction_stats(
            method="ias",
            data=ale,
            estimator_names=estimator_names,
            **kwargs,
        )
        results_ds = self._append_attributes(results_ds)

        return results_ds

    def sobol_indices(self, n_bootstrap=5000, class_index=1):
        """
        Compute the 1st Order and Total order Sobol Indices. Useful for diagnosing feature
        interactions.


        Parameters
        ------------

        Returns
        ----------


        Examples
        ---------
        >>> import skexplain
        >>> # pre-fit estimators within skexplain
        >>> estimators = skexplain.load_models()
        >>> X, y = skexplain.load_data() # training data
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators
        ...                             X=X,
        ...                             y=y,
        ...                            )
        >>> ale = explainer.ale(features='all')
        >>> ias = explainer.interaction_strength(ale)
        """

        results_ds = self.global_obj.compute_sobol(n_bootstrap, class_idx=class_index)
        results_ds = self._append_attributes(results_ds)

        return results_ds

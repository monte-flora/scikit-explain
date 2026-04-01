import itertools
import numpy as np
import xarray as xr

from ..common.utils import is_str, to_xarray, check_all_features_for_ale
from ..common.importance_utils import (
    retrieve_important_vars, combine_top_features, compute_importance,
    to_skexplain_importance,
)
from ._validation import normalize_features, normalize_estimator_names, track_timing


class ImportanceMixin:
    """Mixin providing feature-importance methods for ExplainToolkit."""

    @track_timing
    def permutation_importance(
        self,
        n_vars,
        evaluation_fn,
        direction="backward",
        subsample=1.0,
        n_jobs=1,
        n_permute=1,
        scoring_strategy=None,
        verbose=False,
        return_iterations=False,
        random_seed=42,
        to_importance=False,
    ):
        """
        Performs single-pass and/or multi-pass permutation importance using a modified version of the
        PermutationImportance package (skexplain.PermutationImportance) [1]_. The single-pass approach was first
        developed in Brieman (2001) [2]_ and then improved upon in Lakshmanan et al. (2015) [3]_.

        .. attention ::
                The permutation importance rankings can be sensitive to the evaluation function used.
                Consider re-computing with multiple evaluation functions.

        .. attention ::
                The permutation importance rankings can be sensitive to the direction used.
                Consider re-computing with both forward- and backward-based methods.

        .. hint ::
            Since the permutation importance is a marginal-based method, you can often use
            subsample << 1.0 without substantially altering the feature rankings.
            Using a subsample << 1.0 can reduce the computation time for larger datasets (e.g., >100 K X),
            especially since 100-1000s of bootstrap iterations are often required for reliable rankings.

        Parameters
        ----------

        n_vars : integer
            number of variables to calculate the multipass permutation importance for. If ``n_vars=1``, then
            only the single-pass permutation importance is computed. If ``n_vars>1``, both the single-pass
            and multiple-pass are computed.

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

            When using a custom function, you must also set the scoring strategy (see below).

        scoring_strategy : 'maximize', 'minimize', or None (default=None)

            This argument is only required if you are using a non-default evaluation_fn (see above)

            If the evaluation_fn is positively-oriented (a higher value is better),
            then set ``scoring_strategy = "minimize"`` (i.e., a lower score after permutation
            indicates higher importance) and if it is negatively-oriented-
            (a lower value is better), then set ``scoring_strategy = "maximize"``

        direction : ``"forward"`` or ``"backward"``

            For the multi-pass method. For ``"backward"``, the top feature is left permuted before determining
            the second-most important feature (and so on). For ``"forward"``, all features are permuted
            and then the top features are progressively left unpermuted. For real-world datasets, the two
            methods often do not produce the same feature rankings and is worth exploring both.

        subsample: float or integer (default=1.0 for no subsampling)

            if value is between 0-1, it is interpreted as fraction of total X to use
            if value > 1, interpreted as the number of X to randomly sample
            from the original dataset.

        n_jobs : interger or float (default=1; no multiprocessing)

            if integer, interpreted as the number of processors to use for multiprocessing
            if float between 0-1, interpreted as the fraction of proceesors to use for multiprocessing

        n_permute: integer (default=1 for only one permutation per feature)
            Number of permutations for computing confidence intervals on the feature rankings.

        random_seed : int, RandomState instance, default=None

            Pseudo-random number generator to control the permutations of each
            feature. Pass an int to get reproducible results across function calls.

        verbose : boolean
            True for print statements on the progress

        Returns
        --------
        results : xarray.DataSet
            Permutation importance results. Includes the both multi-pass and single-pass
            feature rankings and the scores with the various features permuted.

        References
        -----------
        .. [1] https://github.com/gelijergensen/PermutationImportance

        .. [2] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

        .. [3] Lakshmanan, V., C. Karstens, J. Krause, K. Elmore, A. Ryzhkov, and S. Berkseth, 2015:
               Which Polarimetric Variables Are Important for Weather/No-Weather Discrimination?
               Journal of Atmospheric and Oceanic Technology, 32, 1209-1223,
               https://doi.org/10.1175/jtech-d-13-00205.1.

        Examples
        ----------
        >>> import skexplain
        >>> # pre-fit estimators within skexplain
        >>> estimators = skexplain.load_models()
        >>> X, y = skexplain.load_data() # training data
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators,
        ...                             X=X,
        ...                             y=y,
        ...                            )
        >>> perm_imp_results = explainer.permutation_importance(
        ...                       n_vars=10,
        ...                       evaluation_fn = 'norm_aupdc',
        ...                       subsample=0.5,
        ...                       n_permute=20,
        ...                       )
        >>> print(perm_imp_results)
        <xarray.Dataset>
            Dimensions:           (n_permute: 20, n_vars_multipass: 10, n_vars_singlepass: 30)
            Dimensions without coordinates: n_permute, n_vars_multipass, n_vars_singlepass
            Data variables:
                multipass_rankings__Random Forest   (n_vars_multipass) <U17 'sfc_te...
                multipass_scores__Random Forest     (n_vars_multipass, n_permute) float64 ...
                singlepass_rankings__Random Forest  (n_vars_singlepass) <U17 'sfc_t...
                singlepass_scores__Random Forest    (n_vars_singlepass, n_permute) float64 ...
                original_score__Random Forest       (n_permute) float64 0.9851 .....
            Attributes:
                estimator_output:  probability
                estimators used:   ['Random Forest']
                n_multipass_vars:  10
                method:            permutation_importance
                direction:         backward
                evaluation_fn:     norm_aupdc
        """
        results_ds, scoring_strategy = self.global_obj.calc_permutation_importance(
            n_vars=n_vars,
            evaluation_fn=evaluation_fn,
            subsample=subsample,
            n_jobs=n_jobs,
            n_permute=n_permute,
            scoring_strategy=scoring_strategy,
            verbose=verbose,
            direction=direction,
            return_iterations=return_iterations,
            random_seed=random_seed,
        )

        # Rename the results:
        for opt in ["multipass", "singlepass"]:
            pimp_vars = [v for v in results_ds.data_vars if opt in v]
            name_dict = {v: f"{direction}_{v}" for v in pimp_vars}
            results_ds = results_ds.rename(name_dict)

        if not is_str(evaluation_fn):
            evaluation_fn = evaluation_fn.__name__

        self.attrs_dict["n_multipass_vars"] = n_vars
        self.attrs_dict["method"] = "permutation_importance"
        self.attrs_dict["direction"] = direction
        self.attrs_dict["evaluation_fn"] = evaluation_fn
        results_ds = self._append_attributes(results_ds)

        # Convert the permutation scores to proper importance scores.
        if to_importance:
            results_ds = compute_importance(results_ds, scoring_strategy, direction)

        return results_ds

    @track_timing
    def grouped_permutation_importance(
        self,
        perm_method,
        evaluation_fn,
        scoring_strategy=None,
        n_permute=1,
        groups=None,
        sample_size=100,
        subsample=1.0,
        n_jobs=1,
        clustering_kwargs=None,
    ):
        """
        The group only permutation feature importance (GOPFI) from Au et al. 2021 [1]_
        (see their equations 10 and 11). This function has a built-in method for clustering
        features using the sklearn.cluster.FeatureAgglomeration. It also has the ability to
        compute the results over multiple permutations to improve the feature importance
        estimate (and provide uncertainty).

        Original score = Jointly permute all features
        Permuted score = Jointly permuting all features except the considered group

        Loss metrics := Original_score - Permuted Score
        Skill Score metrics := Permuted score - Original Score

        Parameters
        ----------

        perm_method : ``"grouped"`` or ``"grouped_only"``
            If ``"grouped"``, the features within a group are jointly permuted and other features
            are left unpermuted.

            If ``"grouped_only"``, only the features within a group are left unpermuted and
            other features are jointly permuted.

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

            When using a custom function, you must also set the scoring strategy (see below).

        scoring_strategy : string (default=None)
            This argument is only required if you are using a non-default evaluation_fn (see above)

            If the evaluation_fn is positively-oriented (a higher value is better),
            then set ``scoring_strategy = "minimize"`` (i.e., a lower score after permutation
            indicates higher importance) and if it is negatively-oriented-
            (a lower value is better), then set ``scoring_strategy = "maximize"``

        n_permute: integer (default=1 for only one permutation per feature)
            Number of permutations for computing confidence intervals on the feature rankings.

        groups : dict (default=None)
            Dictionary of group names and the feature names or feature column indices.
            If None, then the feature groupings are determined internally based on
            feature clusterings.

        sample_size : integer (default=100)
            Number of random samples to determine the correlation for the feature clusterings

        subsample: float or integer (default=1.0 for no subsampling)
            if value is between 0-1, it is interpreted as fraction of total X to use
            if value > 1, interpreted as the number of X to randomly sample
            from the original dataset.

        n_jobs : interger or float (default=1; no multiprocessing)
           if integer, interpreted as the number of processors to use for multiprocessing
           if float between 0-1, interpreted as the fraction of proceesors to use for multiprocessing

        clustering_kwargs : dict (default = {'n_clusters' : 10})
            See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html
            for details

        Returns
        -------

        results : xarray.DataSet
            Permutation importance results. Includes the both multi-pass and single-pass
            feature rankings and the scores with the various features permuted.

        groups : dict
            If groups is None, then it returns the groups that were
            automatically created in the feature clustering. Otherwise,
            only results is returned.

        References
        -----------
        .. [1] Au, Q., J. Herbinger, C. Stachl, B. Bischl, and G. Casalicchio, 2021:
        Grouped Feature Importance and Combined Features Effect Plot. Arxiv,.

        Examples
        ----------
        >>> import skexplain
        >>> # pre-fit estimators within skexplain
        >>> estimators = skexplain.load_models()
        >>> X, y = skexplain.load_data() # training data
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators,
        ...                             X=X,
        ...                             y=y,
        ...                            )
        >>> # Group only, the features within a group are the only one's left unpermuted
        >>> results, groups = explainer.grouped_permutation_importance(
        ...                                          perm_method = 'grouped_only',
        ...                                          evaluation_fn = 'norm_aupdc',)
        """
        if perm_method not in ["grouped", "grouped_only"]:
            raise ValueError(
                "Invalid perm_method! Available options are 'grouped' and 'grouped_only'"
            )

        if clustering_kwargs is None:
            clustering_kwargs = {"n_clusters": 10}

        results_ds, groups = self.global_obj.grouped_feature_importance(
            evaluation_fn=evaluation_fn,
            perm_method=perm_method,
            n_permute=n_permute,
            groups=groups,
            scoring_strategy=scoring_strategy,
            sample_size=sample_size,
            subsample=subsample,
            clustering_kwargs=clustering_kwargs,
            n_jobs=n_jobs,
        )

        for k, v in groups.items():
            self.attrs_dict[k] = list(v)

        if not is_str(evaluation_fn):
            evaluation_fn = evaluation_fn.__name__

        self.attrs_dict["method"] = "grouped_permutation_importance"
        self.attrs_dict["perm_method"] = perm_method
        self.attrs_dict["evaluation_fn"] = evaluation_fn
        self.attrs_dict["feature_groups"] = {k: list(v) for k, v in groups.items()}
        results_ds = self._append_attributes(results_ds)

        return results_ds

    @track_timing
    def ale_variance(
        self,
        ale,
        features=None,
        estimator_names=None,
        interaction=False,
        method="ale",
    ):
        """
        Compute the standard deviation (std) of the ALE values for each
        features in a dataset and then rank by the magnitude. A higher std(ALE) indicates a
        greater expected contribution to an estimator's prediction and is thus considered more important.
        If ``interaction=True``, then the method computes a similar method for the
        2D ALE to measure the feature interaction strength.

        This method is inspired by the feature importance and interaction
        methods developed in Greenwell et al. (2018) [4]_.

        Parameters
        ----------

        ale : xarray.Dataset

            Results of :func:`~ExplainToolkit.ale` for
            ``features``.

        features : 'all', string, list of strings, list of 2-tuples

            Features to compute the ALE variance for. If set to ``'all'``, it is
            computed for all features. If ``interaction=True``, then features
            must be a list of 2-tuples for computing the interaction between
            the set of feature combinations.

        estimator_names : string, list of strings

            If using multiple estimators, you can pass a single (or subset of) estimator name(s)
            to compute the ALE variance for.

        interaction : boolean

            - If True, it computes the feature interaction strength
            - If False, compute the feature importance

        Returns
        --------

        results_ds : xarray.Dataset
            ALE variance results. Includes both the rankings and scores.

        References
        -------------

        .. [4] Greenwell, B. M., B. C. Boehmke, and A. J. McCarthy, 2018:
               A Simple and Effective estimator-Based Variable Importance Measure. Arxiv,.


        Examples
        -----------
        >>> import skexplain
        >>> import itertools
        >>> # pre-fit estimators within skexplain
        >>> estimators = skexplain.load_models()
        >>> X, y = skexplain.load_data() # training data
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators,
        ...                             X=X,
        ...                             y=y,
        ...                            )
        >>> ale = explainer.ale(features='all', n_bins=10, subsample=1000, n_bootstrap=1)
        >>> # Compute 1D ALE variance
        >>> ale_var_results = explainer.ale_variance(ale)
        """
        if (features == "all" or features is None) and interaction:
            features = list(itertools.combinations(self.feature_names, r=2))
        elif features == "all" or features is None:
            # Assume all features.
            features = self.feature_names

        estimator_names = normalize_estimator_names(estimator_names, self.estimator_names)

        if interaction:
            if ale.attrs["dimension"] != "2D":
                raise Exception("ale must be second-order if interaction == True")

        # Check that ale_data is an xarray.Dataset
        if not isinstance(ale, xr.core.dataset.Dataset):
            raise ValueError(
                """
                                 ale must be an xarray.Dataset, 
                                 perferably generated by ExplainToolkit.ale 
                                 to be formatted correctly
                                 """
            )
        else:
            any_missing = all([m in ale.attrs["estimators used"] for m in estimator_names])
            if not any_missing:
                raise ValueError("ale does not contain values for all the estimator names given!")

        if interaction:
            func = self.global_obj.compute_interaction_rankings
        else:
            func = self.global_obj.compute_variance

        results_ds = func(
            method=method,
            data=ale,
            estimator_names=estimator_names,
            features=features,
        )

        self.attrs_dict["method"] = "ale_variance"
        self.attrs_dict["estimators used"] = estimator_names
        self.attrs_dict["estimator output"] = "probability"
        self.attrs_dict["interaction"] = str(interaction)
        if interaction:
            self.attrs_dict["evaluation_fn"] = "Interaction Importance"
        else:
            self.attrs_dict["evaluation_fn"] = "sigma_ale"  #'$\sigma$(ALE)'

        results_ds = self._append_attributes(results_ds)

        return results_ds

    def pd_variance(
        self,
        pd,
        features=None,
        estimator_names=None,
        interaction=False,
    ):
        """See ale_variance for documentation."""
        results_ds = self.ale_variance(
            pd,
            features=features,
            estimator_names=estimator_names,
            interaction=interaction,
            method="pd",
        )

        self.attrs_dict["method"] = "pd_variance"
        results_ds = self._append_attributes(results_ds)

        return results_ds

    def get_important_vars(self, perm_imp_data, multipass=True, n_vars=10, combine=False):
        """
        Retrieve the most important variables from permutation importance.
        Can combine rankings from different estimators and only keep those variables that
        occur in more than one estimator.

        Parameters
        ------------

        perm_imp_data : xarray.Dataset
            Permutation importance result dataset

        multipass : boolean (defaults to True)

            if True, return the multipass rankings else returns the singlepass rankings

        n_vars : integer (default=10)
            Number of variables to retrieve if multipass=True.

        combine : boolean  (default=False)
            If combine=True, n_vars can be set such that you only include a certain amount of
            top features from each estimator. E.g., n_vars=5 and combine=True means to combine
            the top 5 features from each estimator into a single list.

        Examples
        -------
            if combine=True
                results : list
                    List of top features from a different estimators.
            if combine=False
                results : dict
                    keys are the estimator names and items are the
                    top features.

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
        >>> perm_imp_data = explainer.permutation_importance(
        ...                       n_vars=10,
        ...                       evaluation_fn = 'norm_aupdc',
        ...                       direction = 'backward',
        ...                       subsample=0.5,
        ...                       n_bootstrap=20,
        ...                       )
        >>> important_vars = explainer.get_important_vars(perm_imp_data,
        ...        multipass=True, n_vars=5, combine=False)
        ...
        >>> # set combine=True
        >>> important_vars = explainer.get_important_vars(perm_imp_data,
        ...        multipass=True, n_vars=5, combine=True)
        """
        results = retrieve_important_vars(
            perm_imp_data, estimator_names=self.estimator_names, multipass=multipass
        )
        if not combine:
            return results
        else:
            return combine_top_features(results, n_vars=n_vars)

    @track_timing
    def sage(
        self,
        background=None,
        groups=None,
        n_background=50,
        n_jobs=1,
        random_state=42,
        loss=None,
        **sage_kws,
    ):
        """
        Compute SAGE (Shapley Additive Global importancE) values [16]_.

        SAGE measures each feature's global importance by estimating its
        contribution to model performance using Shapley values. Unlike
        permutation importance, SAGE properly accounts for feature interactions.

        Requires the optional ``sage-importance`` package::

            pip install sage-importance

        Parameters
        ----------
        background : array-like, optional
            Background dataset for the marginal imputer. If None, uses ``self.X``.

        groups : dict, optional
            Feature groups for grouped SAGE. Keys are group names, values are lists
            of feature names. When provided, uses ``sage.GroupedMarginalImputer``.
            E.g., ``{'temperature': ['temp2m', 'sfc_temp'], 'wind': ['wind10m', 'fric_vel']}``

        n_background : int, default=50
            Number of random samples from the background data to use for the imputer.

        n_jobs : int, default=1
            Number of parallel jobs for the SAGE estimator.

        random_state : int, default=42
            Random seed for reproducibility.

        loss : str, optional
            Loss function for the SAGE estimator. If None, auto-detected:
            ``'cross entropy'`` for classifiers, ``'mse'`` for regressors.

        **sage_kws
            Additional keyword arguments passed to ``sage.PermutationEstimator.__call__``.
            E.g., ``batch_size``, ``detect_convergence``, ``thresh``, ``n_permutations``.

        Returns
        -------
        results : xarray.Dataset
            Dataset with SAGE importance rankings and scores for each estimator.
            Variables: ``sage_rankings__{est_name}``, ``sage_scores__{est_name}``,
            ``sage_scores_std__{est_name}``.

            When ``groups`` is provided, uses method name ``grouped_sage``.

        References
        ----------
        .. [16] Covert, I., Lundberg, S., and Lee, S.-I., 2020:
                Understanding Global Feature Contributions With Additive
                Importance Measures. NeurIPS.

        Examples
        --------
        >>> import skexplain
        >>> estimators = skexplain.load_models()
        >>> X, y = skexplain.load_data()
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators, X=X, y=y)
        >>> sage_results = explainer.sage()
        >>> explainer.plot_importance(
        ...     data=sage_results,
        ...     panels=[('sage', 'Random Forest')],
        ... )
        """
        try:
            import sage
        except ImportError:
            raise ImportError(
                "The 'sage-importance' package is required for SAGE computation. "
                "Install it with: pip install sage-importance"
            )

        if background is None:
            background = self.X

        rs = np.random.RandomState(random_state)
        n_bg = min(n_background, len(background))
        random_inds = rs.choice(len(background), size=n_bg, replace=False)
        try:
            X_bg = background.values[random_inds, :]
        except AttributeError:
            X_bg = background[random_inds, :]

        method_name = "grouped_sage" if groups is not None else "sage"

        results_list = []
        for estimator_name, estimator in self.estimators.items():
            # Determine model function and loss
            if loss is not None:
                loss_ = loss
            elif hasattr(estimator, "predict_proba"):
                loss_ = "cross entropy"
            else:
                loss_ = "mse"

            model_fn = (
                estimator.predict_proba
                if hasattr(estimator, "predict_proba")
                else estimator.predict
            )

            # Set up the imputer
            if groups is not None:
                # Convert group names → list of index lists
                group_indices = [
                    [self.feature_names.index(f) for f in feats]
                    for feats in groups.values()
                ]
                imputer = sage.GroupedMarginalImputer(model_fn, X_bg, group_indices)
                feature_names = list(groups.keys())
            else:
                imputer = sage.MarginalImputer(model_fn, X_bg)
                feature_names = self.feature_names

            # Compute SAGE
            estimator_sage = sage.PermutationEstimator(
                imputer, loss_, n_jobs=n_jobs, random_state=rs,
            )

            try:
                X_vals = self.X.values
            except AttributeError:
                X_vals = self.X

            sage_values = estimator_sage(X_vals, self.y, **sage_kws)

            # Convert to skexplain format
            result_ds = to_skexplain_importance(
                sage_values,
                estimator_name=estimator_name,
                feature_names=feature_names,
                method=method_name,
                normalize=False,
            )
            results_list.append(result_ds)

        # Merge results from all estimators
        results_ds = xr.merge(results_list, combine_attrs="override")

        self.attrs_dict["method"] = method_name
        if groups is not None:
            self.attrs_dict["feature_groups"] = {k: list(v) for k, v in groups.items()}
        results_ds = self._append_attributes(results_ds)

        return results_ds

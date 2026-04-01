from ..common.utils import to_xarray, check_all_features_for_ale
from ._validation import normalize_features, normalize_estimator_names


class CurvesMixin:
    """Mixin providing ICE / PD / ALE curve methods and main-effect complexity."""

    def ice(
        self,
        features,
        n_bins=30,
        n_jobs=1,
        subsample=1.0,
        n_bootstrap=1,
        random_seed=42,
    ):
        """
        Compute the individual conditional expectations (ICE) [7]_.

        References
        ------------
        .. [7] https://christophm.github.io/interpretable-ml-book/ice.html


        Parameters
        -----------

        features : string or list of strings or 'all'
            Features to compute the ICE for.  if 'all', the method will compute
            the ICE for all features.

        n_bins : integer (default=30)
            Number of bins used to compute the ICE for. Bins are decided based
            on percentile intervals to ensure the same number of samples are in
            each bin.

        n_jobs : float or integer (default=1)

            - if integer, interpreted as the number of processors to use for multiprocessing
            - if float, interpreted as the fraction of proceesors to use for multiprocessing

        subsample : float or integer (default=1.0)

            - if value between 0-1 interpreted as fraction of total X to use
            - if value > 1, interpreted as the absolute number of random samples of X.

        n_bootstrap : integer (default=1; no bootstrapping)
            Number of bootstrap resamples for computing confidence intervals on the ICE curves.

        Returns
        ---------

        results : xarray.DataSet
            Main keys are the user-provided estimator names while the sub-key
            are the features computed for. The items are data for the ICE curves. Also,
            contains X data (feature values where the ICE curves were computed) for plotting.

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
        >>> ice_ds = explainer.ice(features='all', subsample=200)

        """
        features = normalize_features(features, self.feature_names)

        results_ds = self.global_obj._run_interpret_curves(
            method="ice",
            features=features,
            n_bins=n_bins,
            n_jobs=n_jobs,
            subsample=subsample,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )

        dimension = "2D" if isinstance(list(features)[0], tuple) else "1D"
        self.attrs_dict["method"] = "ice"
        self.attrs_dict["dimension"] = dimension
        self.attrs_dict["features used"] = features

        results_ds = self._append_attributes(results_ds)

        self.feature_used = features

        return results_ds

    def pd(
        self,
        features,
        n_bins=25,
        n_jobs=1,
        subsample=1.0,
        n_bootstrap=1,
        random_seed=42,
    ):
        """
        Computes the 1D or 2D centered partial dependence (PD) [8]_.

        References
        ------------

        .. [8] https://christophm.github.io/interpretable-ml-book/pdp.html

        Parameters
        ----------

        features : string or list of strings or 'all'
            Features to compute the PD for.  if 'all', the method will compute
            the PD for all features.

        n_bins : integer (default=30)
            Number of bins used to compute the PD for. Bins are decided based
            on percentile intervals to ensure the same number of samples are in
            each bin.

        n_jobs : float or integer (default=1)

            - if integer, interpreted as the number of processors to use for multiprocessing
            - if float, interpreted as the fraction of proceesors to use for multiprocessing

        subsample : float or integer (default=1.0)

            - if value between 0-1 interpreted as fraction of total X to use
            - if value > 1, interpreted as the absolute number of random samples of X.

        n_bootstrap : integer (default=1; no bootstrapping)
            Number of bootstrap resamples for computing confidence intervals on the PD curves.

        Returns
        --------

        results : xarray.DataSet
            Partial dependence result dataset

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
        >>> pd = explainer.pd(features='all')
        """
        features = normalize_features(features, self.feature_names, allow_2d=True)

        results_ds = self.global_obj._run_interpret_curves(
            method="pd",
            features=features,
            n_bins=n_bins,
            n_jobs=n_jobs,
            subsample=subsample,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )

        dimension = "2D" if isinstance(list(features)[0], tuple) else "1D"
        self.attrs_dict["method"] = "pd"
        self.attrs_dict["dimension"] = dimension
        self.attrs_dict["features used"] = features

        results_ds = self._append_attributes(results_ds)
        self.features_used = features

        return results_ds

    def ale(
        self,
        features=None,
        n_bins=30,
        n_jobs=1,
        subsample=1.0,
        n_bootstrap=1,
        random_seed=42,
        class_index=1,
    ):
        """
        Compute the 1D or 2D centered accumulated local effects (ALE) [9]_ [10]_.
        For categorical features, simply set the type of those features in the
        dataframe as ``category`` and the categorical ALE will be computed.

        References
        -----------

        .. [9] https://christophm.github.io/interpretable-ml-book/ale.html

        .. [10] Apley, D. W., and J. Zhu, 2016: Visualizing the Effects of Predictor Variables in
                Black Box Supervised Learning Models. ArXiv.


        Parameters
        ----------

        features : string or list of strings or 'all'
            Features to compute the PD for.  if 'all', the method will compute
            the ALE for all features.

        n_bins : integer (default=30)
            Number of bins used to compute the ALE for. Bins are decided based
            on percentile intervals to ensure the same number of samples are in
            each bin.

        n_jobs : float or integer (default=1)

            - if integer, interpreted as the number of processors to use for multiprocessing
            - if float, interpreted as the fraction of proceesors to use for multiprocessing

        subsample : float or integer (default=1.0)

            - if value between 0-1 interpreted as fraction of total X to use
            - if value > 1, interpreted as the absolute number of random samples of X.

        n_bootstrap : integer (default=1; no bootstrapping)
            Number of bootstrap resamples for computing confidence intervals on the ALE curves.

        Returns
        ----------

        results : xarray.DataSet
            ALE result dataset

        Raise
        ----------
        Exception
            Highly skewed data may not be divisable into n_bins given. In that case, calc_ale
            uses the max bins the data can be divided into. But a warning message is raised.

        Examples
        ---------
        >>> import skexplain
        >>> estimators = skexplain.load_models() # pre-fit estimators within skexplain
        >>> X, y = skexplain.load_data() # training data
        >>> # Set the type for categorical features and ExplainToolkit with compute the
        >>> # categorical ALE.
        >>> X = X.astype({'urban': 'category', 'rural':'category'})
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators
        ...                             X=X,
        ...                             y=y,
        ...                            )
        >>> ale = explainer.ale(features='all')
        """
        features = normalize_features(features, self.feature_names, allow_2d=True)

        results_ds = self.global_obj._run_interpret_curves(
            method="ale",
            features=features,
            n_bins=n_bins,
            n_jobs=n_jobs,
            subsample=subsample,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
            class_index=class_index,
        )

        dimension = "2D" if isinstance(list(features)[0], tuple) else "1D"
        self.attrs_dict["method"] = "ale"
        self.attrs_dict["dimension"] = dimension
        self.attrs_dict["features used"] = features

        results_ds = self._append_attributes(results_ds)
        self.features_used = features

        return results_ds

    def main_effect_complexity(self, ale, estimator_names=None, max_segments=10, approx_error=0.05):
        """
        Compute the Main Effect Complexity (MEC; Molnar et al. 2019) [5]_.
        MEC is the number of linear segements required to approximate
        the first-order ALE curves averaged over all features.
        The MEC is weighted-averged by the variance. Higher values indicate
        a more complex estimator (less interpretable).

        References
        -----------
        .. [5] Molnar, C., G. Casalicchio, and B. Bischl, 2019: Quantifying estimator Complexity via
            Functional Decomposition for Better Post-Hoc Interpretability. ArXiv.


        Parameters
        ----------------

        ale : xarray.Dataset

             Results of :func:`~ExplainToolkit.ale`. Must be computed for all features in X.

        estimator_names : string, list of strings

            If using multiple estimators, you can pass a single (or subset of) estimator name(s)
            to compute the MEC for.

        max_segments : integer; default=10

            Maximum number of linear segments used to approximate the main/first-order
            effect of a feature. default is 10. Used to limit the computational runtime.

        approx_error : float; default=0.05

            The accepted error of the R squared between the piece-wise linear function
            and the true ALE curve. If the R square is within the approx_error, then
            no additional segments are added.


        Returns
        ---------
            mec_dict : dictionary
                mec_dict = {estimator_name0 : mec0, estimator_name1 : mec2, ..., estimator_nameN : mecN,}


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
        >>> ale = explainer.ale(features='all', n_bins=20, subsample=0.5, n_bootstrap=20)
        >>> # Compute Main Effect Complexity (MEC)
        >>> mec_ds = explainer.main_effect_complexity(ale)
        >>> print(mes_ds)
        {'Random Forest': 2.6792782503392756,
         'Gradient Boosting': 2.692392706080586,
         'Logistic Regression': 1.6338281469152958}
        """
        estimator_names = normalize_estimator_names(estimator_names, self.estimator_names)

        check_all_features_for_ale(ale, estimator_names, self.feature_names)

        dataset = {}
        for estimator_name in estimator_names:
            mec, _ = self.global_obj.compute_main_effect_complexity(
                estimator_name=estimator_name,
                ale_ds=ale,
                features=self.feature_names,
                max_segments=max_segments,
                approx_error=approx_error,
            )

            dataset[f"mec__{estimator_name}"] = mec

        results_ds = to_xarray(dataset)
        self.attrs_dict["method"] = "main_effect_complexity"
        results_ds = self._append_attributes(results_ds)

        return results_ds

import numpy as np
import warnings

from ..common.utils import to_xarray, is_str, is_list, is_dataset


class AttributionMixin:
    """Mixin providing local-attribution methods for ExplainToolkit."""

    def local_contributions(
        self,
        method="shap",
        performance_based=False,
        n_samples=100,
        shap_kwargs=None,
        lime_kws=None,
    ):
        warnings.warn(
            "ExplainToolkit.local_contributions is deprecated. Use local_attributions in the future.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.local_attributions(
            method=method,
            shap_kws=shap_kwargs,
            lime_kws=lime_kws,
        )

    def local_attributions(self, method, shap_kws=None, lime_kws=None, n_jobs=1):
        """
        Compute the SHapley Additive Explanations (SHAP) values [13]_ [14]_ [15]_,
        Local Interpretable Model Explanations (LIME) or the Tree Interpreter local
        attributions for a set of examples.
        .
        By default, we set the SHAP algorithm = ``'auto'``, so that the best algorithm
        for a model is determined internally in the SHAP package.

        Parameters
        ------------------
        method : ``'shap'`` , ``'tree_interpreter'``, or ``'lime'`` or list
            Can use SHAP, treeinterpreter, or LIME to compute the feature attributions.
            SHAP and LIME are estimator-agnostic while treeinterpreter can only be used on
            select decision-tree based estimators in scikit-learn (e.g., random forests).

        shap_kws : dict (default is None)
            Arguments passed to the shap.Explainer object. See
            https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer
            for details. The main two arguments supported in skexplain is the masker and
            algorithm options. By default, the masker option uses
            masker = shap.maskers.Partition(X, max_samples=100, clustering="correlation") for
            hierarchical clustering by correlations. You can also provide a background dataset
            e.g., background_dataset = shap.sample(X, 100).reset_index(drop=True). The algorithm
            option is set to "auto" by default.

            - masker
            - algorithm

        lime_kws : dict (default is None)
            Arguments passed to the LimeTabularExplainer object. See https://github.com/marcotcr/lime
            for details. Generally, you'll pass the in the following:

            - training_data
            - categorical_names (scikit-explain will attempt to determine it internally,
                                 if it is not passed in)
            - random_state (for reproduciability)

        n_jobs : float or integer (default=1)

            - if integer, interpreted as the number of processors to use for multiprocessing
            - if float, interpreted as the fraction of proceesors to use for multiprocessing

            For treeinterpreter, parallelization is used to process the trees of a random forest
            in parallel. For LIME, each example is computed in parallel. We do not apply
            parallelization to SHAP as we found it is faster without it.

        Returns
        -------------------

        results : xarray.Dataset
            A dataset containing shap values [(n_samples, n_features)] for each estimator
            (e.g., 'shap_values__estimator_name'), the bias ('bias__estimator_name')
            of shape (n_examples, 1), and the X and y values the shap values were determined from.

        References
        ------------
        .. [13] https://christophm.github.io/interpretable-ml-book/shap.html
        .. [14] Lundberg, S. M., G. G. Erion, and S.-I. Lee, 2018: Consistent Individualized
                Feature Attribution for Tree Ensembles. Arxiv,.
        .. [15] Lundberg, S. M., and Coauthors, 2020: From local explanations to global understanding
                with explainable AI for trees. Nat Mach Intell, 2, 56-67, https://doi.org/10.1038/s42256-019-0138-9.


        Examples
        ---------
        >>> import skexplain
        >>> import shap
        >>> # pre-fit estimators within skexplain
        >>> estimators = skexplain.load_models()
        >>> X, _ = skexplain.load_data() # training data
        >>> X_subset = shap.sample(X, 50, random_state=22)
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators
        ...                             X=X_subset,)
        >>> results = explainer.local_attributions(shap_kws={'masker' :
        ...                          shap.maskers.Partition(X, max_samples=100, clustering="correlation"),
        ...                          'algorithm' : 'auto'})
        """
        if shap_kws is None:
            shap_kws = {}
        if lime_kws is None:
            lime_kws = {}

        dataset = {}
        include_ys = True
        if len(self.y) < 1:
            warnings.warn(
                """No y values were provided!
                          The y values are useful for color-coding in the shap dependence plots."""
            )
            include_ys = False

        if not is_list(method):
            methods = [method]
        else:
            methods = method

        correct_names = ["shap", "tree_interpreter", "lime"]
        r = [[m in correct_names][0] for m in methods]
        if not all(r):
            ind = r.index(False)
            raise ValueError(
                f"Invalid method ({methods[ind]})! Method must be one of the following: 'shap', 'tree_interpreter', 'lime'"
            )

        for estimator_name, estimator in self.estimators.items():
            for method in methods:

                df = self.local_obj._get_feature_contributions(
                    estimator=estimator,
                    X=self.X,
                    shap_kws=shap_kws,
                    lime_kws=lime_kws,
                    n_jobs=n_jobs,
                    method=method,
                    estimator_output=self.estimator_output,
                )

                values = df[self.feature_names]
                bias = df["Bias"]

                dataset[f"{method}_values__{estimator_name}"] = (
                    ["n_examples", "n_features"],
                    values,
                )
                dataset[f"{method}_bias__{estimator_name}"] = (
                    ["n_examples"],
                    bias.astype(np.float64),
                )

        dataset["X"] = (["n_examples", "n_features"], self.X.values)

        # Y may not be given. Need to check!
        if include_ys:
            dataset["y"] = (["n_examples"], self.y)

        results_ds = to_xarray(dataset)
        self.attrs_dict["features"] = self.feature_names
        self.attrs_dict["method"] = methods
        results_ds = self._append_attributes(results_ds)

        return results_ds

    def average_attributions(
        self,
        method=None,
        data=None,
        performance_based=False,
        n_samples=100,
        shap_kws=None,
        lime_kws=None,
        n_jobs=1,
    ):
        """
        Computes the individual feature contributions to a predicted outcome for
        a series of examples either based on tree interpreter (only Tree-based methods)
        , Shapley Additive Explanations, or Local Interpretable Model-Agnostic Explanations (LIME).

        The primary difference between average_attributions and local_attributions is the
        performance-based determiniation of examples to compute the local attributions from.
        average_attributions can start with the full dataset and determine the top n_samples
        to compute explanations for.

        Parameters
        -----------
        method : ``'shap'`` , ``'tree_interpreter'``, or ``'lime'`` (default is None)
            Can use SHAP, treeinterpreter, or LIME to compute the feature attributions.
            SHAP and LIME are estimator-agnostic while treeinterpreter can only be used on
            select decision-tree based estimators in scikit-learn (e.g., random forests).

        data : dataframe or a list of dataframes, shape (n_examples, n_features) (Default is None)
            Local attribution data for each estimator.
            Results from explainer.local_attributions. If None, then the local attributions are computed
            internally.

        performance_based : boolean (default=False)
            If True, will average feature contributions over the best and worst
            performing of the given X. The number of examples to average over
            is given by n_samples

        n_samples : interger (default=100)
            Number of samples to compute average over if performance_based = True

        Returns
        --------

        results_df : nested pandas.DataFrame
            For each example, contributions from each feature plus the bias
            The dataframe is nested by the estimator names and additional keys
            if performance_based=True.


        Examples
        ---------
        >>> import skexplain
        >>> import shap
        >>> estimators = skexplain.load_models()
        >>> X, y = skexplain.load_data()
        >>> single_example = X.iloc[[0]]
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators,
        ...                             X=single_example,
        ...                            )
        >>> contrib_ds = explainer.local_attributions(method='shap',
        ...     shap_kws={'masker': shap.sample(X, 100), 'algorithm': 'auto'})
        >>> avg_contrib = explainer.average_attributions(data=contrib_ds)

        >>> # For the performance-based contributions,
        >>> # provide the full set of X and y values.
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators,
        ...                             X=X,
        ...                             y=y,
        ...                            )
        >>> avg_contrib = explainer.average_attributions(method='shap',
        ...                   performance_based=True, n_samples=100)

        """
        if data is not None:
            if not is_dataset(data):
                raise ValueError(
                    "Data needs to be a xarray.Dataset from ExplainToolkit.local_attributions."
                )
            methods = data.attrs["method"]
        else:
            if method is None:
                raise ValueError("Set the method if not providing a Dataset.")

            if not is_list(method):
                methods = [method]
            else:
                methods = method

        results = {}

        for method in methods:
            results_df = self.local_obj._average_attributions(
                data=data,
                method=method,
                performance_based=performance_based,
                n_samples=n_samples,
                shap_kws=shap_kws,
                lime_kws=lime_kws,
                n_jobs=n_jobs,
            )

            # Add metadata
            self.attrs_dict["method"] = method
            self.attrs_dict["n_samples"] = n_samples
            self.attrs_dict["performance_based"] = str(performance_based)
            self.attrs_dict["features"] = self.feature_names
            results_df = self._append_attributes(results_df)

            results[method] = results_df

        return results

    def shap(self, shap_kws=None, shap_kwargs=None):
        """
        Compute the SHapley Additive Explanations (SHAP) values [13]_ [14]_ [15]_.
        By default, we set algorithm = ``'auto'``, so that the best algorithm
        for a model is determined internally in the SHAP package.

        Parameters
        ------------------
        shap_kws : dict
            Arguments passed to the shap.Explainer object. See
            https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer
            for details. The main two arguments supported in skexplain is the masker and
            algorithm options. By default, the masker option uses
            masker = shap.maskers.Partition(X, max_samples=100, clustering="correlation") for
            hierarchical clustering by correlations. You can also provide a background dataset
            e.g., background_dataset = shap.sample(X, 100).reset_index(drop=True). The algorithm
            option is set to "auto" by default.

            - masker
            - algorithm

        Returns
        -------------------

        results : xarray.Dataset
            A dataset containing shap values [(n_samples, n_features)] for each estimator
            (e.g., 'shap_values__estimator_name'), the bias ('bias__estimator_name')
            of shape (n_examples, 1), and the X and y values the shap values were determined from.

        References
        ------------
        .. [13] https://christophm.github.io/interpretable-ml-book/shap.html
        .. [14] Lundberg, S. M., G. G. Erion, and S.-I. Lee, 2018: Consistent Individualized
                Feature Attribution for Tree Ensembles. Arxiv,.
        .. [15] Lundberg, S. M., and Coauthors, 2020: From local explanations to global understanding
                with explainable AI for trees. Nat Mach Intell, 2, 56-67, https://doi.org/10.1038/s42256-019-0138-9.


        Examples
        ---------
        >>> import skexplain
        >>> import shap
        >>> # pre-fit estimators within skexplain
        >>> estimators = skexplain.load_models()
        >>> X, _ = skexplain.load_data() # training data
        >>> X_subset = shap.sample(X, 50, random_state=22)
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators
        ...                             X=X_subset,)
        >>> results = explainer.shap(shap_kws={'masker' :
        ...                          shap.maskers.Partition(X, max_samples=100, clustering="correlation"),
        ...                          'algorithm' : 'auto'})
        """
        warnings.warn(
            "explainer.shap is deprecated. Use explainer.local_attributions in the future",
            DeprecationWarning,
            stacklevel=2,
        )

        if shap_kws is None:
            shap_kws = {"masker": None, "algorithm": "auto"}

        shap_kwargs = shap_kws

        dataset = {}
        include_ys = True
        if len(self.y) < 1:
            warnings.warn(
                """No y values were provided!
                          The y values are useful for color-coding in the shap dependence plots."""
            )
            include_ys = False

        for estimator_name, estimator in self.estimators.items():
            shap_values, bias = self.local_obj._get_shap_values(
                estimator=estimator,
                X=self.X,
                shap_kws=shap_kws,
            )

            dataset[f"shap_values__{estimator_name}"] = (
                ["n_examples", "n_features"],
                shap_values,
            )
            dataset[f"bias__{estimator_name}"] = (
                ["n_examples"],
                bias.astype(np.float64),
            )

        dataset["X"] = (["n_examples", "n_features"], self.X.values)

        # Y may not be given. Need to check!
        if include_ys:
            dataset["y"] = (["n_examples"], self.y)

        results_ds = to_xarray(dataset)
        self.attrs_dict["features"] = self.feature_names
        results_ds = self._append_attributes(results_ds)

        return results_ds

import numpy as np
import pandas as pd

from ..common.utils import is_str, is_list, is_dataset
from ..plot.plot_interpret_curves import PlotInterpretCurves
from ..plot.plot_permutation_importance import PlotImportance
from ..plot.plot_feature_contributions import PlotFeatureContributions
from ..plot.plot_2D import PlotInterpret2D
from ..plot._box_and_whisker import box_and_whisker
from ..plot._kde_2d import PlotScatter
from ._validation import normalize_estimator_names


class PlottingMixin:
    """Mixin providing all plotting methods for ExplainToolkit."""

    def _plot_interpret_curves(
        self,
        method,
        data,
        estimator_names,
        add_hist,
        features=None,
        display_feature_names=None,
        display_units=None,
        to_probability=False,
        line_kws=None,
        cbar_kwargs=None,
        **kwargs,
    ):
        """
        FOR INTERNAL USE ONLY.

        Handles 1D or 2D PD/ALE plots.
        """
        # Merge with plot config -- per-call args override config
        if display_feature_names is None:
            display_feature_names = self._plot_config.display_feature_names or {}
        if display_units is None:
            display_units = self._plot_config.display_units or {}
        if line_kws is None:
            line_kws = {}

        # Inject config defaults into kwargs (per-call kwargs win)
        if self._plot_config.figsize is not None and "figsize" not in kwargs:
            kwargs["figsize"] = self._plot_config.figsize
        if self._plot_config.n_columns is not None and "n_columns" not in kwargs:
            kwargs["n_columns"] = self._plot_config.n_columns
        if self._plot_config.wspace is not None and "wspace" not in kwargs:
            kwargs["wspace"] = self._plot_config.wspace
        if self._plot_config.hspace is not None and "hspace" not in kwargs:
            kwargs["hspace"] = self._plot_config.hspace

        if features is None:
            try:
                features = self.features_used
            except:
                raise ValueError("No features were provided to plot!")
        else:
            if is_str(features):
                features = [features]

        if data.attrs["dimension"] == "2D":
            plot_obj = PlotInterpret2D()
            return plot_obj.plot_contours(
                method=method,
                data=data,
                estimator_names=estimator_names,
                features=features,
                display_feature_names=display_feature_names,
                display_units=display_units,
                to_probability=to_probability,
                cbar_kwargs=cbar_kwargs,
                **kwargs,
            )
        else:
            base_font_size = 12 if len(features) <= 6 else 16
            base_font_size = kwargs.get("base_font_size", base_font_size)
            plot_obj = PlotInterpretCurves(
                BASE_FONT_SIZE=base_font_size, seaborn_kws=self.seaborn_kws
            )
            return plot_obj.plot_1d_curve(
                method=method,
                data=data,
                add_hist=add_hist,
                estimator_names=estimator_names,
                features=features,
                display_feature_names=display_feature_names,
                display_units=display_units,
                to_probability=to_probability,
                line_kws=line_kws,
                **kwargs,
            )

    def plot_pd(
        self,
        pd=None,
        features=None,
        estimator_names=None,
        add_hist=True,
        display_feature_names=None,
        display_units=None,
        to_probability=None,
        line_kws=None,
        cbar_kwargs=None,
        **kwargs,
    ):
        """
        Runs the 1D and 2D partial dependence plotting.

        Parameters
        ----------

        pd : xarray.Dataset
            Results of :func:`~ExplainToolkit.pd` for
            ``features``.

        features : string, list of strings, list of 2-tuple of strings
            Features to plot the PD for.  To plot for 2D PD,
            pass a list of 2-tuples of features.

        estimator_names : string, list of strings (default is None)
            If using multiple estimators, you can pass a single (or subset of) estimator name(s)
            to plot for.

        add_hist : True/False (default=True)
            If True, adds the histogram of a feature's values behind the interpret curves.

        display_feature_names : dict
            For plotting purposes. Dictionary that maps the feature names
            in the pandas.DataFrame to display-friendly versions.
            E.g., ``display_feature_names = { 'dwpt2m' : '$T_{d}$', }``

            The plotting code can handle latex-style formatting.

        display_units : dict
            For plotting purposes. Dictionary that maps the feature names
            to their units.
            E.g., ``display_units = { 'dwpt2m' : '$^\\circ$C', }``

        line_colors : str or list of strs of len(estimators)
            User-defined colors for curve plotting.

        to_probability : boolean
            If True, the values are multipled by 100.

        Keyword arguments include arguments typically used for matplotlib.


        Returns
        --------

        fig, axes: matplotlib figure instance and the corresponding axes


        Examples
        ---------
        >>> import skexplain
        >>> estimators = skexplain.load_models() # pre-fit estimators within skexplain
        >>> X, y = skexplain.load_data() # training data
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators
        ...                             X=X,
        ...                             y=y,
        ...                            )
        >>> pd_ds = explainer.pd(features='all')
        >>> # Provide a small subset of features to plot
        >>> important_vars = ['sfc_temp', 'temp2m', 'sfcT_hrs_bl_frez',
        ...     'tmp2m_hrs_bl_frez','uplwav_flux']
        >>> explainer.plot_pd(pd_ds, features=important_vars)

        """
        estimator_names = normalize_estimator_names(estimator_names, self.estimator_names)

        if to_probability is None and pd.attrs["estimator_output"] == "probability":
            to_probability = True
        elif to_probability is None:
            to_probability = False

        if to_probability:
            kwargs["left_yaxis_label"] = "Centered PD (%)"
        else:
            kwargs["left_yaxis_label"] = "Centered PD"

        return self._plot_interpret_curves(
            method="pd",
            data=pd,
            features=features,
            add_hist=add_hist,
            estimator_names=estimator_names,
            display_feature_names=display_feature_names,
            display_units=display_units,
            to_probability=to_probability,
            line_kws=line_kws,
            cbar_kwargs=cbar_kwargs,
            **kwargs,
        )

    def plot_ale(
        self,
        ale=None,
        features=None,
        estimator_names=None,
        add_hist=True,
        display_feature_names=None,
        display_units=None,
        to_probability=None,
        line_kws=None,
        cbar_kwargs=None,
        **kwargs,
    ):
        """
        Runs the 1D and 2D accumulated local effects plotting.

        Parameters
        ----------

        ale : xarray.Dataset
             Results of :func:`~ExplainToolkit.ale` for
            ``features``.

        features : string, list of strings, list of 2-tuple of strings
            Features to plot the PD for.  To plot for 2D PD,
            pass a list of 2-tuples of features.

        estimator_names : string, list of strings (default is None)
            If using multiple estimators, you can pass a single (or subset of) estimator name(s)
            to plot for.

        add_hist : True/False (default=True)
            If True, adds the histogram of a feature's values behind the interpret curves.

        display_feature_names : dict
            For plotting purposes. Dictionary that maps the feature names
            in the pandas.DataFrame to display-friendly versions.
            E.g., ``display_feature_names = { 'dwpt2m' : '$T_{d}$', }``

            The plotting code can handle latex-style formatting.

        display_units : dict
            For plotting purposes. Dictionary that maps the feature names
            to their units.
            E.g., ``display_units = { 'dwpt2m' : '$^\\circ$C', }``

        line_colors : str or list of strs of len(estimators)
            User-defined colors for curve plotting.

        to_probability : boolean
            If True, the values are multipled by 100.

        Keyword arguments include arguments typically used for matplotlib.
            E.g.,
            figsize, hist_color,

        Returns
        --------

        fig, axes: matplotlib figure instance and the corresponding axes


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
        >>> # Provide a small subset of features to plot
        >>> important_vars = ['sfc_temp', 'temp2m', 'sfcT_hrs_bl_frez',
        ...     'tmp2m_hrs_bl_frez','uplwav_flux']
        >>> explainer.plot_ale(ale, features=important_vars)

        .. image :: /_static/images/ale_1d.png
        """
        estimator_names = normalize_estimator_names(estimator_names, self.estimator_names)

        if to_probability is None and ale.attrs["estimator_output"] == "probability":
            to_probability = True
        elif to_probability is None:
            to_probability = False

        if to_probability:
            kwargs["left_yaxis_label"] = "Centered ALE (%)"
        else:
            kwargs["left_yaxis_label"] = "Centered ALE"

        return self._plot_interpret_curves(
            method="ale",
            data=ale,
            add_hist=add_hist,
            features=features,
            estimator_names=estimator_names,
            display_feature_names=display_feature_names,
            display_units=display_units,
            to_probability=to_probability,
            line_kws=line_kws,
            cbar_kwargs=cbar_kwargs,
            **kwargs,
        )

    def plot_contributions(
        self,
        contrib=None,
        features=None,
        estimator_names=None,
        display_feature_names=None,
        **kwargs,
    ):
        """
        Plots the feature contributions.

        Parameters
        ------------
        contrib : Nested pandas.DataFrame or dict of Nested pandas.DataFrame
            Results of :func:`~ExplainToolkit.local_attributions` or :func:`~ExplainToolkit.average_attributions`
            :func:`~ExplainToolkit.local_attributions` returns an xarray.Dataset which can be valid for multiple examples.
            For plotting, :func:`~ExplainToolkit.average_attributions` is used to average attributions and their
            feature values.

        features : string or list of strings (default=None)

               Features to plot. If None, all features are eligible to be plotted.
               However, the default number of features to plot is 10. Can be set
               by n_vars (see keyword arguments).

        estimator_names : string, list of strings (default is None)

            If using multiple estimators, you can pass a single (or subset of) estimator name(s)
            to compute the IAS for.

        display_feature_names : dict
            For plotting purposes. Dictionary that maps the feature names
            in the pandas.DataFrame to display-friendly versions.
            E.g., display_feature_names = { 'dwpt2m' : 'T$_{d}$', }
            The plotting code can handle latex-style formatting.

        Keyword arguments include arguments typically used for matplotlib

        Returns
        ---------

        fig: matplotlib figure instance

        Examples
        ---------
        >>> import skexplain
        >>> import shap
        >>> estimators = skexplain.load_models() # pre-fit estimators within skexplain
        >>> X, y = skexplain.load_data() # training data
        >>> single_example = X.iloc[[0]]
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators,
        ...                             X=single_example,
        ...                            )
        >>> background_dataset = shap.sample(X, 100)
        >>> contrib_ds = explainer.local_attributions(method='shap',
        ...     shap_kws={'masker': background_dataset, 'algorithm': 'auto'})
        >>> explainer.plot_contributions(contrib_ds)

        .. image :: /_static/images/feature_contribution_single.png
        """
        if display_feature_names is None:
            display_feature_names = self._plot_config.display_feature_names or {}

        if is_dataset(contrib):
            contrib = self.average_attributions(data=contrib, performance_based=False)

        keys = list(contrib.keys())

        estimator_names = normalize_estimator_names(
            estimator_names, contrib[keys[0]].attrs["estimators used"]
        )

        estimator_output = contrib[keys[0]].attrs["estimator_output"]
        if features is None:
            features = contrib[keys[0]].attrs["features"]

        # initialize a plotting object
        only_one_panel = (
            contrib[keys[0]].index[0][0] == "non_performance"
            and len(estimator_names) == 1
            and len(keys) == 1
        )

        base_font_size = kwargs.get("base_font_size", 16 if only_one_panel else 11)
        plot_obj = PlotFeatureContributions(
            BASE_FONT_SIZE=base_font_size, seaborn_kws=self.seaborn_kws
        )
        kwargs["estimator_output"] = self.estimator_output

        return plot_obj.plot_contributions(
            data=contrib,
            estimator_names=estimator_names,
            features=features,
            display_feature_names=display_feature_names,
            **kwargs,
        )

    def scatter_plot(
        self,
        dataset,
        estimator_name,
        method=None,
        plot_type="summary",
        features=None,
        display_feature_names=None,
        display_units=None,
        **kwargs,
    ):
        """
        Plot the SHapley Additive Explanations (SHAP) [13]_ [14]_ [15]_ summary plot or dependence
        plots for various features.

        Parameters
        -----------

        plot_type : ``'summary'`` or ``'dependence'``
            if 'summary', plots a feature importance-style plot
            if 'dependence', plots a partial depedence style plot

        dataset : xarray.Dataset
            Results from :func:`~ExplainToolkit.local_attributions`.
            Dataset containing feature attribution values, their biases, and
            the input feature values.

        method : ``'shap'`` , ``'tree_interpreter'``, or ``'lime'`` (default is None)
            Can use SHAP, treeinterpreter, or LIME to compute the feature attributions.
            SHAP and LIME are estimator-agnostic while treeinterpreter can only be used on
            select decision-tree based estimators in scikit-learn (e.g., random forests).
            If None, method is determine from the values Dataset. Otherwise, an
            error is raised.

        features : string or list of strings (default=None)
            features to plots if plot_type is 'dependence'.

        display_feature_names : dict
            For plotting purposes. Dictionary that maps the feature names
            in the pandas.DataFrame to display-friendly versions.
            E.g., ``display_feature_names = { 'dwpt2m' : '$T_{d}$', }``
            The plotting code can handle latex-style formatting.

        display_units : dict
            For plotting purposes. Dictionary that maps the feature names
            to their units.
            E.g., ``display_units = { 'dwpt2m' : '$^\\circ$C', }``

        to_probability : boolean
            if True, values are multiplied by 100.

        Returns
        -----------------------
        fig: matplotlib figure instance

        Examples
        ---------
        >>> import skexplain
        >>> import shap
        >>> # pre-fit estimators within skexplain
        >>> estimators = skexplain.load_models()
        >>> X, y = skexplain.load_data() # training data
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators
        ...                             X=X,
        ...                             y=y,
        ...                            )
        >>> results = explainer.local_attributions(method='shap')
        >>> # Plot the SHAP-summary style plot
        >>> explainer.scatter_plot(results, estimator_name='Random Forest',
        ...     plot_type='summary')
        >>> # Plot the SHAP-dependence style plot
        >>> important_vars = ['sfc_temp', 'temp2m', 'sfcT_hrs_bl_frez',
        ...     'tmp2m_hrs_bl_frez', 'uplwav_flux']
        >>> explainer.scatter_plot(results, estimator_name='Random Forest',
        ...     plot_type='dependence', features=important_vars)

        .. image :: /_static/images/shap_dependence.png

        """
        if display_feature_names is None:
            display_feature_names = self._plot_config.display_feature_names or {}
        if display_units is None:
            display_units = self._plot_config.display_units or {}

        if method is not None:
            if is_list(method):
                methods = method
            else:
                methods = [method]
        else:
            methods = dataset.attrs["method"]

        X = pd.DataFrame(dataset["X"].values, columns=dataset.attrs["features"])

        if plot_type == "summary" and len(methods) > 1:
            raise ValueError("At the moment, summary plots can only handle one method")
        elif plot_type == "summary":
            dataset = dataset[f"{methods[0]}_values__{estimator_name}"].values

        if plot_type not in ["summary", "dependence"]:
            raise ValueError("Invalid plot_type! Must be 'summary' or 'dependence'")

        # initialize a plotting object
        if plot_type == "summary":
            fontsize = 12
        else:
            fontsize = 12 if len(features) <= 6 else 16

        base_font_size = kwargs.get("base_font_size", fontsize)
        plot_obj = PlotFeatureContributions(
            BASE_FONT_SIZE=base_font_size, seaborn_kws=self.seaborn_kws
        )
        plot_obj.feature_names = self.feature_names
        return plot_obj.scatter_plot(
            attr_values=dataset,
            X=X,
            features=features,
            plot_type=plot_type,
            display_feature_names=display_feature_names,
            display_units=display_units,
            estimator_name=estimator_name,
            methods=methods,
            **kwargs,
        )

    def plot_importance(self, data, panels, plot_correlated_features=False, **kwargs):
        """
        Method for plotting the permutation importance and other ranking-based results.

        Parameters
        -------------
        panels: List of 2-tuple of (method, estimator name) to determine the sub-panel
                matrixing for the plotting. E.g., If you wanted to compare multi-pass to
                single-pass permutation importance for a random forest:
               ``panels  = [('multipass', 'Random Forest'), ('singlepass', 'Random Forest')]``
                The available ranking methods in skexplain include 'multipass', 'singlepass',
                'perm_based', 'ale_variance', or 'ale_variance_interactions'.

        data :  list of xarray.Datasets
            Results from

            - :func:`~ExplainToolkit.permutation_importance`
            - :func:`~ExplainToolkit.ale_variance`
            - :func:`~ExplainToolkit.friedman_h_stat`
            - :func:`~ExplainToolkit.perm_based_interaction`

            For each element in panels, there needs to be a corresponding element in data.

        columns : list of strings
            What will be the columns of the plot? These can be x-axis label (default is
            the different estimator names)

        rows : list of strings
            Y-axis label or multiple labels for each row in a multi-panel plot. (default is None).

        plot_correlated_features : boolean
            If True, pairs of features with a linear correlation coefficient > 0.8
            are annotate/paired by bars or color-coding. This is useful for identifying
            spurious rankings due to the correlations.

        kwargs : keyword arguments

        num_vars_to_plot : integer
            Number of features to plot from permutation importance calculation.

        Returns
        --------
        fig: matplotlib figure instance


        Examples
        -------
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
        ...                       direction = 'backward',
        ...                       subsample=0.5,
        ...                       n_bootstrap=20,
        ...                       )
        >>> explainer.plot_importance(data=perm_imp_results,
        ...     panels=[('multipass', 'Random Forest')])


        >>> #If we want to annotate pairs of highly correlated feature pairs
        >>> explainer.plot_importance(data=perm_imp_results,
        ...     panels=[('multipass', 'Random Forest')],
        ...     plot_correlated_features=True)

        .. image :: /_static/images/multi_pass_perm_imp.png

        """
        if is_list(data):
            assert len(data) == len(panels), "Panels and Data must have the same number of elements"
        else:
            data = [data]

        if len(data) != len(panels):
            # Assuming that data contains multiple models.
            given_estimator_names = [m[1] for m in panels]
            available_estimators = [
                f.split("rankings__")[1] for f in list(data[0].data_vars) if "rank" in f
            ]
            missing = np.array(
                [True if f not in available_estimators else False for f in given_estimator_names]
            )
            missing_estimators = list(np.array(given_estimator_names)[missing])
            if any(missing):
                txt = ""
                for i in missing_estimators:
                    txt += i + ", "
                raise ValueError(
                    f"""Results for {txt} are not in the given dataset.
                      Check for possible spelling errors"""
                )

            data *= len(panels)

        for r, (method, estimator_name) in zip(data, panels):
            available_methods = [
                d.split("__")[0] for d in list(r.data_vars) if f"rankings__{estimator_name}" in d
            ]
            if f"{method}_rankings" not in available_methods:
                raise ValueError(
                    f"""{method} does not match the available methods for this item({available_methods}).
                         Ensure that the elements of data match up with those panels!
                         Also check for any possible spelling error.
                         """
                )

        estimator_output = kwargs.get("estimator_output", self.estimator_output)
        kwargs.pop("estimator_output", None)

        # initialize a plotting object
        base_font_size = kwargs.get("base_font_size", 12)
        plot_obj = PlotImportance(BASE_FONT_SIZE=base_font_size, seaborn_kws=self.seaborn_kws)

        if plot_correlated_features:
            kwargs["X"] = self.X

        return plot_obj.plot_variable_importance(
            data,
            panels=panels,
            plot_correlated_features=plot_correlated_features,
            estimator_output=estimator_output,
            **kwargs,
        )

    def plot_box_and_whisker(
        self,
        important_vars,
        example,
        display_feature_names=None,
        display_units=None,
        **kwargs,
    ):
        """
        Plot the training dataset distribution for a given set of important variables
        as a box-and-whisker plot. The user provides a single example, which is highlighted
        over those examples. Useful for real-time explainability.

        Parameters
        -------------

        important_vars : str or list of strings
            List of features to plot

        example : Pandas Series, shape = (important_vars,)
            Single row dataframe to be overlaid, must have columns equal to
            the given important_vars


        """
        if display_feature_names is None:
            display_feature_names = self._plot_config.display_feature_names or {}
        if display_units is None:
            display_units = self._plot_config.display_units or {}

        if not is_list(important_vars):
            important_vars = [important_vars]

        axis = "columns" if isinstance(example, pd.DataFrame) else "index"
        if set(getattr(example, axis)) != set(important_vars):
            raise ValueError("The example dataframe/series must have important_vars as columns!")

        f, axes = box_and_whisker(
            self.X,
            top_preds=important_vars,
            example=example,
            display_feature_names=display_feature_names,
            display_units=display_units,
            **kwargs,
        )
        return f, axes

    def plot_scatter(
        self,
        features,
        kde=True,
        subsample=1.0,
        display_feature_names=None,
        display_units=None,
        **kwargs,
    ):
        """
        2-D Scatter plot of ML model predictions. If kde=True, it will plot KDE contours
        overlays to show highest concentrations. If the model type is classification, then
        the code will plot KDE contours per class.
        """
        if display_feature_names is None:
            display_feature_names = self._plot_config.display_feature_names or {}
        if display_units is None:
            display_units = self._plot_config.display_units or {}

        # Are features in X?
        bad_features = [f for f in features if f not in self.feature_names]
        if len(bad_features) > 0:
            raise ValueError(
                f"{bad_features} is not a valid feature. Check for possible spelling errors!"
            )

        # initialize a plotting object
        base_font_size = kwargs.get("base_font_size", 12)
        plot_obj = PlotScatter(base_font_size, seaborn_kws=self.seaborn_kws)

        f, axes = plot_obj.plot_scatter(
            self.estimators,
            X=self.X,
            y=self.y,
            features=features,
            display_feature_names=display_feature_names,
            display_units=display_units,
            subsample=subsample,
            peak_val=None,
            kde=kde,
            **kwargs,
        )

        return f, axes

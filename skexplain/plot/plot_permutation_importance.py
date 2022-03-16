import numpy as np
import collections

from ..common.importance_utils import find_correlated_pairs_among_top_features
from ..common.utils import is_list, is_correlated
from .base_plotting import PlotStructure
import random


class PlotImportance(PlotStructure):
    """
    PlotImportance handles plotting feature ranking plotting. The class
    is designed to be generic enough to handle all possible ranking methods
    computed within PyMint.
    """

    SINGLE_VAR_METHODS = [
        "multipass",
        "singlepass",
        "ale_variance",
        "coefs",
        "shap_sum",
        "gini",
        "combined",
        "sage",
        "grouped",
        "grouped_only",
    ]

    DISPLAY_NAMES_DICT = {
        "multipass": "Multi-Pass Importance Scores",
        "singlepass": "Single-Pass Importance Scores",
        "perm_based": "Permutation-based Interactions Importance Scores",
        "ale_variance": "ALE-Based Importance Scores",
        "ale_variance_interactions": "ALE-Based Interactions Importance Scores",
        "coefs": "Coef. Importance Scores",
        "shap_sum": "SHAP Importance Scores",
        "hstat": "H-Stat Importance Scores",
        "gini": "Gini Importance Scores",
        "combined": "Method-Average Ranking",
        "sage": "SAGE Importance Scores",
        "grouped": "Grouped Importance Scores",
        "grouped_only": "Grouped Only Importance Scores",
    }

    def __init__(self, BASE_FONT_SIZE=12):
        super().__init__(BASE_FONT_SIZE=BASE_FONT_SIZE)

    def is_bootstrapped(self, scores):
        """Check if the permutation importance results are bootstrapped"""
        return np.ndim(scores) > 1

    def _get_axes(self, n_panels, **kwargs):
        """
        Determine how many axes are required.
        """
        if n_panels == 1:
            kwargs["figsize"] = kwargs.get("figsize", (3, 2.5))
        elif n_panels == 2 or n_panels == 3:
            kwargs["figsize"] = kwargs.get("figsize", (6, 2.5))
        else:
            figsize = kwargs.get("figsize", (8, 5))

        # create subplots, one for each feature
        fig, axes = self.create_subplots(n_panels=n_panels, **kwargs)

        return fig, axes

    def _check_for_estimators(self, data, estimator_names):
        """Check that each estimator is in data"""
        for ds in data:
            if not (
                collections.Counter(ds.attrs["estimators used"])
                == collections.Counter(estimator_names)
            ):
                raise AttributeError(
                    """
                 The estimator names given do not match the estimators used to create
                 given data 
                                     """
                )

    def plot_variable_importance(
        self,
        data,
        panels,
        display_feature_names={},
        feature_colors=None,
        num_vars_to_plot=10,
        estimator_output="raw",
        plot_correlated_features=False,
        **kwargs,
    ):

        """Plots any variable importance method for a particular estimator

        Parameters
        -----------------
            data : xarray.Dataset or list of xarray.Dataset
                Permutation importance dataset for one or more metrics
            panels: list of 2-tuples of estimator names and rank method
                E.g., panels = [('singlepass', 'Random Forest',
                                ('multipass', 'Random Forest') ]
                will plot the singlepass and multipass results for the
                random forest model.
                Possible methods include 'multipass', 'singlepass',
                'perm_based', 'ale_variance', or 'ale_variance_interactions'
            display_feature_names : dict
                A dict mapping feature names to readable, "pretty" feature names
            feature_colors : dict
                A dict mapping features to various colors. Helpful for color coding groups of features
            num_vars_to_plot : int
                Number of top variables to plot (defalut is None and will use number of multipass results)
                
            kwargs: 
                - xlabels 
                - ylabel 
                - xticks
                - p_values 
                - colinear_features 
                - rho_threshold 
                
        """
        xlabels = kwargs.get("xlabels", None)
        ylabels = kwargs.get("ylabels", None)
        xticks = kwargs.get("xticks", None)
        title = kwargs.get("title", "")
        p_values = kwargs.get("p_values", None)
        colinear_features = kwargs.get("colinear_features", None)
        rho_threshold = kwargs.get("rho_threshold", 0.8)

        only_one_method = all([m[0] == panels[0][0] for m in panels])
        only_one_estimator = all([m[1] == panels[0][1] for m in panels])

        if not only_one_method:
            kwargs["hspace"] = kwargs.get("hspace", 0.6)

        if plot_correlated_features:
            X = kwargs.get("X", None)
            if X is None or X.empty:
                raise ValueError(
                    "Must provide X to InterpretToolkit to compute the correlations!"
                )
            corr_matrix = X.corr().abs()

        data = [data] if not is_list(data) else data
        n_panels = len(panels)

        fig, axes = self._get_axes(n_panels, **kwargs)
        ax_iterator = self.axes_to_iterator(n_panels, axes)

        for i, (panel, ax) in enumerate(zip(panels, ax_iterator)):

            # Set the facecolor.
            #ax.set_facecolor(kwargs.get("facecolor", (0.95, 0.95, 0.95)))

            method, estimator_name = panel
            results = data[i]
            if xlabels is not None:
                ax.set_xlabel(xlabels[i], fontsize=self.FONT_SIZES["small"])
            else:
                if not only_one_method:
                    ax.set_xlabel(
                        self.DISPLAY_NAMES_DICT.get(method, method),
                        fontsize=self.FONT_SIZES["small"],
                    )
            if not only_one_estimator:
                ax.set_title(estimator_name)

            sorted_var_names = list(
                results[f"{method}_rankings__{estimator_name}"].values
            )

            if num_vars_to_plot is None:
                num_vars_to_plot == len(sorted_var_names)

            sorted_var_names = sorted_var_names[
                : min(num_vars_to_plot, len(sorted_var_names))
            ]

            sorted_var_names = sorted_var_names[::-1]
            scores = results[f"{method}_scores__{estimator_name}"].values

            scores = scores[: min(num_vars_to_plot, len(sorted_var_names))]

            # Reverse the order.
            scores = scores[::-1]

            # Set very small values to zero.
            scores = np.where(np.absolute(np.round(scores, 17)) < 1e-15, 0, scores)

            # Get the colors for the plot
            colors_to_plot = [
                self.variable_to_color(var, feature_colors) for var in sorted_var_names
            ]
            
            # Get the predictor names
            variable_names_to_plot = [
                f" {var}"
                for var in self.convert_vars_to_readable(
                    sorted_var_names,
                    display_feature_names,
                )
            ]

            if method == "combined":
                scores_to_plot = np.nanpercentile(scores, 50, axis=1)
                # Compute the confidence intervals (ci)
                ci = np.abs(
                    np.nanpercentile(scores, 50, axis=1)
                    - np.nanpercentile(scores, [25, 75], axis=1)
                )
            else:
                scores_to_plot = np.nanmean(scores, axis=1)
                ci = np.abs(
                    np.nanpercentile(scores, 50, axis=1)
                    - np.nanpercentile(scores, [2.5, 97.5], axis=1)
                )

            # Despine
            self.despine_plt(ax)

            elinewidth = 0.9 if n_panels <= 3 else 0.5

            ax.barh(
                np.arange(len(scores_to_plot)),
                scores_to_plot,
                linewidth=1.75,
                edgecolor="white",
                alpha=0.5,
                color=colors_to_plot,
                xerr=ci,
                capsize=3.0,
                ecolor="k",
                error_kw=dict(
                    alpha=0.2,
                    elinewidth=elinewidth,
                ),
                zorder=2,
            )

            
            if plot_correlated_features:
                self._add_correlated_brackets(
                    ax, np.arange(len(scores_to_plot)), 
                    scores_to_plot,
                    corr_matrix, sorted_var_names, rho_threshold
                )

            if num_vars_to_plot >= 20:
                size = kwargs.get("fontsize", self.FONT_SIZES["teensie"] - 3)
            elif num_vars_to_plot > 10:
                size = kwargs.get("fontsize", self.FONT_SIZES["teensie"] - 2)
            else:
                size = kwargs.get("fontsize", self.FONT_SIZES["teensie"] - 1)


            # Put the variable names _into_ the plot
            if method not in self.SINGLE_VAR_METHODS and plot_correlated_features:
                results_dict = is_correlated(
                    corr_matrix, sorted_var_names, rho_threshold=rho_threshold
                )

            if colinear_features is None:
                fontweight = ["light"] * len(variable_names_to_plot)
                colors = ["k"] * len(variable_names_to_plot)
            else:
                # Bold text if the VIF > threshold (indicates a multicolinear predictor)
                fontweight =  [
                    "bold" if v in colinear_features else "light" for v in sorted_var_names
                ]  
                
                # Bold text if value is insignificant.
                colors =  ["xkcd:medium blue" if v in colinear_features else "k" for v in sorted_var_names]
                

            ax.set_yticks(range(len(variable_names_to_plot)))
            ax.set_yticklabels(variable_names_to_plot)
            labels = ax.get_yticklabels()
            
            # Bold var names 
            ##[label.set_fontweight(opt) for opt, label in zip(fontweight, labels)]
            
            [label.set_color(c) for c, label in zip(colors, labels)]
            
            ax.tick_params(axis="both", which="both", length=0)

            if xticks is not None:
                ax.set_xticks(xticks)
            else:
                self.set_n_ticks(ax, option="x")
                
        xlabel = (
            self.DISPLAY_NAMES_DICT.get(method, method)
            if (only_one_method and xlabels is None)
            else ""
        )
        
        major_ax = self.set_major_axis_labels(
            fig,
            xlabel=xlabel,
            ylabel_left="",
            ylabel_right="",
            title=title,
            fontsize=self.FONT_SIZES["small"],
            **kwargs,
        )
        
        if ylabels is not None:
            self.set_row_labels(
                labels=ylabels, axes=axes, pos=-1, pad=1.15, rotation=270, **kwargs
            )

        self.add_alphabet_label(
            n_panels, axes, pos=kwargs.get("alphabet_pos", (0.9, 0.09)), 
            alphabet_fontsize = kwargs.get("alphabet_fontsize", 10)
        )

        # Necessary to make sure that the tick labels for the feature names
        # do overlap another ax. 
        fig.tight_layout()
        
        return fig, axes

    def _add_correlated_brackets(self, ax, y, width, corr_matrix, top_features, rho_threshold):
        """
        Add bracket connecting features above a given correlation threshold.
        
        Parameters
        ------------------
        ax : matplotlib.ax.Axes object 
        y : 
        width : 
        corr_matrix: 
        top_features:
        rho_threshold:
        """
        get_colors = lambda n: list(
            map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF), range(n))
        )

        _, pair_indices = find_correlated_pairs_among_top_features(
            corr_matrix,
            top_features,
            rho_threshold=rho_threshold,
        )
        
        colors = get_colors(len(pair_indices))

        top_indices, bottom_indices = [], []
        for p, color in zip(pair_indices, colors):
            delta=0
            if p[0] > p[1]:
                bottom_idx = p[1]
                top_idx = p[0]
            else:
                bottom_idx = p[0]
                top_idx = p[1]
            
            # If a feature has already shown up in a correlated pair,
            # then we want to shift the brackets slightly for ease of 
            # interpretation. 
            if bottom_idx in bottom_indices or bottom_idx in top_indices:
                delta += 0.1
            if top_idx in top_indices or top_idx in bottom_indices:
                delta += 0.1
                
            top_indices.append(top_idx)
            bottom_indices.append(bottom_idx)
            
            self.annotate_bars(ax, bottom_idx, top_idx, y=y, width=width, delta=delta)

    # You can fill this in by using a dictionary with {var_name: legible_name}
    def convert_vars_to_readable(self, variables_list, VARIABLE_NAMES_DICT):
        """Substitutes out variable names for human-readable ones
        :param variables_list: a list of variable names
        :returns: a copy of the list with human-readable names
        """
        human_readable_list = list()
        for var in variables_list:
            if var in VARIABLE_NAMES_DICT:
                human_readable_list.append(VARIABLE_NAMES_DICT[var])
            else:
                human_readable_list.append(var)
        return human_readable_list

    # This could easily be expanded with a dictionary
    def variable_to_color(self, var, VARIABLES_COLOR_DICT):
        """
        Returns the color for each variable.
        """
        if var == "No Permutations":
            return "xkcd:pastel red"
        else:
            if VARIABLES_COLOR_DICT is None:
                return "xkcd:powder blue"
            elif not isinstance(VARIABLES_COLOR_DICT, dict) and isinstance(
                VARIABLES_COLOR_DICT, str
            ):
                return VARIABLES_COLOR_DICT
            else:
                return VARIABLES_COLOR_DICT[var]

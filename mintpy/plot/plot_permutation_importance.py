import numpy as np
import collections

from ..common.utils import find_correlated_pairs_among_top_features, is_correlated

from .base_plotting import PlotStructure


class PlotImportance(PlotStructure):
    def is_bootstrapped(self, original_score):
        """Check if the permutation importance results are bootstrapped"""
        try:
            len(original_score)
        except:
            bootstrapped = False
        else:
            bootstrapped = True

        if bootstrapped:
            original_score_mean = np.mean(original_score)
        else:
            original_score_mean = original_score

        return bootstrapped, original_score_mean

    def _get_axes(self, model_names, metrics_used, **kwargs):
        """
        Determine how many axes are required.
        """
        ylabels = kwargs.get("ylabels", "")
        xlabels = kwargs.get("xlabels", "")
        hspace = kwargs.get("hspace", 0.5)
        wspace = kwargs.get("wspace", 0.2)
        n_columns = kwargs.get("n_columns", 3)

        if len(model_names) == 1:
            # Only one model, but one or more metrics
            only_one_model = True
            xlabels = metrics_used
            n_columns = min(len(xlabels), 3)
            n_panels = len(xlabels)
        else:
            # More than one model, and one or more metrics
            only_one_model = False
            n_columns = min(len(model_names), 3)
            if metrics_used is not None:
                if len(metrics_used) == 1:
                    xlabels = metrics_used
                else:
                    ylabels = metrics_used

        if ylabels is not None:
            n_panels = n_columns * len(ylabels)
        else:
            n_panels = n_columns

        if n_panels == 1:
            figsize = (3, 2.5)
        elif n_panels == 2:
            figsize = (6, 2.5)
        elif n_panels == 3:
            figsize = kwargs.get("figsize", (6, 2.5))
        else:
            figsize = kwargs.get("figsize", (8, 5))
            hspace = 0.2

        # create subplots, one for each feature
        fig, axes = self.create_subplots(
            n_panels=n_panels,
            n_columns=n_columns,
            hspace=hspace,
            wspace=wspace,
            figsize=figsize,
        )

        if n_panels == 1:
            axes = [axes]

        return fig, axes, xlabels, ylabels, only_one_model, n_panels

    def _check_for_models(self, data, model_names):
        """Check that each model is in data"""
        for ds in data:
            if not (
                collections.Counter(ds.attrs["models used"])
                == collections.Counter(model_names)
            ):
                raise AttributeError(
                    """
                 The model names given do not match the models used to create
                 given data 
                                     """
                )

    def plot_variable_importance(
        self,
        data,
        method="multipass",
        metrics_used=None,
        display_feature_names={},
        feature_colors=None,
        num_vars_to_plot=10,
        model_output="raw",
        model_names=None,
        plot_correlated_features=False,
        **kwargs,
    ):

        """Plots any variable importance method for a particular estimator

        Args:
            data : xarray.Dataset or list of xarray.Dataset
                Permutation importance dataset for one or more metrics
            method: 'multipass', 'singlepass', or 'ale_variance'
                Method used to produce the feature ranking.
            display_feature_names : dict
                A dict mapping feature names to readable, "pretty" feature names
            feature_colors : dict
                A dict mapping features to various colors. Helpful for color coding groups of features
            num_vars_to_plot : int
                Number of top variables to plot (defalut is None and will use number of multipass results)
            xaxis_label : str
                Metric used to compute the predictor importance, which will display as the X-axis label.
        """
        xticks = kwargs.get("xticks", None)
        title = kwargs.get("title", "")

        if plot_correlated_features:
            examples = kwargs.get("examples", None)
            if examples is None or examples.empty:
                raise ValueError("Must provide examples to compute the correlations!")
            rho_threshold = 0.8
            corr_matrix = examples.corr().abs()

        if not isinstance(data, list):
            data = [data]

        if any(isinstance(i, list) for i in model_names):
            model_names = model_names[0]

        fig, axes, xlabels, ylabels, only_one_model, n_panels = self._get_axes(
            model_names, metrics_used, **kwargs
        )

        # List of data for different metrics
        for g, results in enumerate(data):
            # loop over each model creating one panel per model
            for k, model_name in enumerate(model_names):
                if np.ndim(axes) == 1:
                    ax = axes[k]
                else:
                    ax = axes[g, k]

                if g == 0:
                    ax.set_title(
                        model_name, fontsize=self.FONT_SIZES["small"], alpha=0.8
                    )

                if only_one_model:
                    ax.set_xlabel(xlabels[g])

                if num_vars_to_plot is None:
                    num_vars_to_plot == len(sorted_var_names)

                sorted_var_names = list(
                    results[f"{method}_rankings__{model_name}"].values
                )

                sorted_var_names = sorted_var_names[
                    : min(num_vars_to_plot, len(sorted_var_names))
                ]

                sorted_var_names = sorted_var_names[::-1]

                scores = [
                    results[f"{method}_scores__{model_name}"].values[i, :]
                    for i in range(len(sorted_var_names))
                ]

                scores = scores[::-1]

                if "pass" in method:
                    # Get the original score (no permutations)
                    original_score = results[f"original_score__{model_name}"].values

                    # Get the original score (no permutations)
                    # Check if the permutation importance is bootstrapped
                    bootstrapped, original_score_mean = self.is_bootstrapped(
                        original_score
                    )

                    sorted_var_names.append("No Permutations")
                    scores.append(original_score)
                else:
                    bootstrapped = True if np.shape(scores)[1] > 1 else False

                 # Set very small values to zero. 
                scores = np.where(np.absolute(np.round(scores,17)) < 1e-15, 0, scores)

                if plot_correlated_features:
                    self._add_correlated_brackets(
                        ax, corr_matrix, sorted_var_names, rho_threshold
                    )

                # Get the colors for the plot
                colors_to_plot = [
                    self.variable_to_color(var, feature_colors)
                    for var in sorted_var_names
                ]
                # Get the predictor names
                variable_names_to_plot = [
                    f" {var}"
                    for var in self.convert_vars_to_readable(
                        sorted_var_names,
                        display_feature_names,
                    )
                ]

                if bootstrapped:
                    scores_to_plot = np.array([np.mean(score) for score in scores])
                    ci = np.array(
                        [
                            np.abs(np.mean(score) - np.percentile(score, [2.5, 97.5]))
                            for score in scores
                        ]
                    ).transpose()

                else:
                    if "pass" in method:
                        scores.append(original_score_mean)
                    else:
                        scores = [score[0] for score in scores]

                    scores_to_plot = np.array(scores)

                # Despine
                self.despine_plt(ax)

                if bootstrapped:
                    ax.barh(
                        np.arange(len(scores_to_plot)),
                        scores_to_plot,
                        linewidth=1,
                        alpha=0.8,
                        color=colors_to_plot,
                        xerr=ci,
                        capsize=2.5,
                        ecolor="grey",
                        error_kw=dict(alpha=0.4),
                        zorder=2,
                    )
                else:
                    ax.barh(
                        np.arange(len(scores_to_plot)),
                        scores_to_plot,
                        alpha=0.8,
                        linewidth=1,
                        color=colors_to_plot,
                        zorder=2,
                    )

                if num_vars_to_plot > 10:
                    size = self.FONT_SIZES["teensie"] - 1
                else:
                    size = self.FONT_SIZES["teensie"]

                # Put the variable names _into_ the plot
                if model_output == "probability" and 'perm_based' not in method:
                    x_pos = 0.0
                    ha = ["left"] * len(variable_names_to_plot)
                elif model_output == "raw" and 'perm_based' not in method:
                    x_pos = 0.05
                    ha = ["left"] * len(variable_names_to_plot)
                else:
                    x_pos = 0
                    ha = ["left" if score > 0 else "right" for score in scores_to_plot] 

                # Put the variable names _into_ the plot
                if (method == "ale_variance_interactions" or method == 'perm_based_interactions') \
                            and plot_correlated_features:
                    results_dict = is_correlated(
                        corr_matrix, sorted_var_names, rho_threshold=rho_threshold
                    )

                for i in range(len(variable_names_to_plot)):
                    color = "k"
                    if (
                        (method == "ale_variance_interactions" or method == 'perm_based_interactions')
                        and plot_correlated_features
                    ):
                        correlated = results_dict.get(sorted_var_names[i], False)
                        color = "xkcd:medium green" if correlated else "k"

                    if 'pass' not in method:
                        var = variable_names_to_plot[i].replace('__', ' & ')
                    else:
                        var = variable_names_to_plot[i]
                        
                    ax.annotate(
                        var,
                        xy=(x_pos, i),
                        va="center",
                        ha=ha[i],
                        size=size,
                        alpha=0.8,
                        color=color,
                    )

                if model_output == "probability" and "pass" in method:
                    # Add vertical line
                    ax.axvline(
                        original_score_mean,
                        linestyle="dashed",
                        color="grey",
                        linewidth=0.7,
                        alpha=0.7,
                    )
                    ax.text(
                        original_score_mean,
                        len(variable_names_to_plot) / 2,
                        "Original Score",
                        va="center",
                        ha="left",
                        size=self.FONT_SIZES["teensie"],
                        rotation=270,
                        alpha=0.7,
                    )

                #if (
                #    model_output == "probability"
                #    and "ale_variance" not in method
                #    and xticks is None
                #):
                    # Most probability-based scores are between 0-1 (AUC, BSS, NAUPDC,etc.)
                #    xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

                ax.tick_params(axis="both", which="both", length=0)
                ax.set_yticks([])
                
                if model_output == "probability" and "pass" in method:
                    upper_limit = min(1.1 * np.nanmax(scores_to_plot), 1.0)
                    ax.set_xlim([0, upper_limit])
                elif "perm_based" in method:
                    upper_limit = max(1.1 * np.nanmax(scores_to_plot), 0.01)
                    lower_limit = min(1.1 * np.nanmin(scores_to_plot), -0.01)
                    ax.set_xlim([lower_limit, upper_limit])    
                else:
                    upper_limit = 1.1 * np.nanmax(scores_to_plot)
                    ax.set_xlim([0, upper_limit])

                if xticks is not None:
                    ax.set_xticks(xticks)
                else:
                    self.set_n_ticks(ax, option="x")  
                
                #print(xticks, upper_limit) 

                # make the horizontal plot go with the highest value at the top
                # ax.invert_yaxis()
                vals = ax.get_xticks()
                for tick in vals:
                    ax.axvline(
                        x=tick, linestyle="dashed", alpha=0.4, color="#eeeeee", zorder=1
                    )

                if k == 0:
                    pad = -0.2 if plot_correlated_features else -0.15
                    ax.annotate(
                        "higher ranking",
                        xy=(pad, 0.8),
                        xytext=(pad, 0.5),
                        arrowprops=dict(arrowstyle="->", color="xkcd:blue grey"),
                        xycoords=ax.transAxes,
                        rotation=90,
                        size=6,
                        ha="center",
                        va="center",
                        color="xkcd:blue grey",
                        alpha=0.65,
                    )

                    ax.annotate(
                        "lower ranking",
                        xy=(pad + 0.05, 0.2),
                        xytext=(pad + 0.05, 0.5),
                        arrowprops=dict(arrowstyle="->", color="xkcd:blue grey"),
                        xycoords=ax.transAxes,
                        rotation=90,
                        size=6,
                        ha="center",
                        va="center",
                        color="xkcd:blue grey",
                        alpha=0.65,
                    )

        if len(xlabels) == 1 and not only_one_model:
            major_ax = self.set_major_axis_labels(
                fig,
                xlabel=xlabels[0],
                ylabel_left="",
                labelpad=5,
                fontsize=self.FONT_SIZES["tiny"],
            )

        if ylabels is not None:
            if isinstance(ylabels, str):
                major_ax.set_ylabel(ylabels, fontsize=self.FONT_SIZEs["tiny"])
            else:
                self.set_row_labels(ylabels, axes)

        if model_output == "probability":
            pos = (0.9, 0.09)
        else:
            pos = (0.9, 0.9)
        self.add_alphabet_label(n_panels, axes, pos=pos)

    def _add_correlated_brackets(self, ax, corr_matrix, top_features, rho_threshold):
        """
        Add bracket connecting features above a given correlation threshold.
        """
        _, pair_indices = find_correlated_pairs_among_top_features(
            corr_matrix,
            top_features,
            rho_threshold=rho_threshold,
        )
        x = 0.0001
        dx = 0.0002
        bottom_indices = []
        top_indices = []
        for p in pair_indices:
            if p[0] > p[1]:
                bottom_idx = p[1]
                top_idx = p[0]
            else:
                bottom_idx = p[0]
                top_idx = p[1]

            if bottom_idx in bottom_indices or bottom_idx in top_indices:
                bottom_idx += 0.1

            if top_idx in top_indices or top_idx in bottom_indices:
                top_idx += 0.1

            bottom_indices.append(bottom_idx)
            top_indices.append(top_idx)

            self.annotate_bars(ax, bottom_idx=bottom_idx, top_idx=top_idx, x=x)
            x += dx

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

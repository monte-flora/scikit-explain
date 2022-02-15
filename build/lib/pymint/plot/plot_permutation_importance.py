import numpy as np
import collections

from ..common.utils import find_correlated_pairs_among_top_features, is_correlated
from ..common.utils import is_list
from .base_plotting import PlotStructure
import random

class PlotImportance(PlotStructure):
    """
    PlotImportance handles plotting feature ranking plotting. The class
    is designed to be generic enough to handle all possible ranking methods
    computed within PyMint. 
    """
    SINGLE_VAR_METHODS = ['multipass', 'singlepass', 'ale_variance', 'coefs', 
                          'shap_sum', 'gini', 'combined', 'sage', 'grouped', 'grouped_only']

    DISPLAY_NAMES_DICT = {'multipass': 'Multiple Pass', 
                   'singlepass' : 'Single Pass', 
                   'perm_based': 'Permutation-based Interactions', 
                   'ale_variance': 'ALE-Based',
                   'ale_variance_interactions': 'ALE-Based Interactions', 
                   'coefs' : 'Coefficients',
                   'shap_sum' : 'SHAP',
                   'hstat' : 'H-Statistic',
                   'gini' : 'Gini Impurity-Based',
                   'combined' : 'Method-Average Ranking',
                   'sage' : 'SAGE',
                   'grouped': 'Grouped', 
                   'grouped_only' : 'Grouped Only',
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
            kwargs['figsize'] = kwargs.get("figsize", (3, 2.5))
        elif n_panels == 2 or n_panels == 3:
            kwargs['figsize'] = kwargs.get("figsize", (6, 2.5))
        else:
            figsize = kwargs.get("figsize", (8, 5))

        # create subplots, one for each feature
        fig, axes = self.create_subplots(
            n_panels=n_panels,
            **kwargs
        )
        
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

        Args:
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
        """
        xlabels = kwargs.get('xlabels', None) 
        ylabels = kwargs.get('ylabels', None) 
        xticks = kwargs.get("xticks", None)
        title = kwargs.get("title", "")
        p_values = kwargs.get('p_values', None)
        colinear_predictors = kwargs.get('colinear_predictors', None)
        rho_threshold = kwargs.get('rho_threshold', 0.8)
        
        only_one_method = all([m[0]==panels[0][0] for m in panels])
        only_one_estimator = all([m[1]==panels[0][1] for m in panels])
        
        if not only_one_method:
            kwargs["hspace"] = kwargs.get('hspace', 0.6)
        
        if plot_correlated_features:
            X = kwargs.get("X", None)
            if X is None or X.empty:
                raise ValueError("Must provide X to InterpretToolkit to compute the correlations!")
            corr_matrix = X.corr().abs()
            
        data = [data] if not is_list(data) else data
        n_panels = len(panels) 
        
        fig, axes= self._get_axes(n_panels, **kwargs)
        ax_iterator = self.axes_to_iterator(n_panels, axes)

        for i, (panel, ax) in enumerate(zip(panels, ax_iterator)):
            method, estimator_name = panel
            results = data[i]
            if xlabels is not None:
                ax.set_xlabel(xlabels[i], fontsize=self.FONT_SIZES["small"])
            else:    
                if not only_one_method:
                    ax.set_xlabel(self.DISPLAY_NAMES_DICT.get(method, method), fontsize=self.FONT_SIZES["small"])
            if not only_one_estimator:
                ax.set_title(estimator_name)
     
            sorted_var_names = list(results[f"{method}_rankings__{estimator_name}"].values)
                
            if num_vars_to_plot is None:
                num_vars_to_plot == len(sorted_var_names) 
                
            sorted_var_names = sorted_var_names[
                    : min(num_vars_to_plot, len(sorted_var_names))
                ]

            sorted_var_names = sorted_var_names[::-1]
            scores = results[f"{method}_scores__{estimator_name}"].values

            scores = scores[:min(num_vars_to_plot, len(sorted_var_names))]
                        
            # Reverse the order.
            scores = scores[::-1]

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

            if method == 'combined':
                scores_to_plot = np.nanpercentile(scores, 50, axis=1)
                # Compute the confidence intervals (ci)
                ci = np.abs( np.nanpercentile(scores, 50, axis=1) - np.nanpercentile(scores, [25, 75], axis=1))
            else:
                scores_to_plot = np.nanmean(scores, axis=1)
                ci = np.abs(np.nanpercentile(scores, 50, axis=1) - np.nanpercentile(scores, [2.5, 97.5], axis=1))
     
            # Despine
            self.despine_plt(ax)
            
            ax.barh(
                        np.arange(len(scores_to_plot)),
                        scores_to_plot,
                        linewidth=1.75,
                        edgecolor='white',
                        alpha=0.5,
                        color=colors_to_plot,
                        xerr=ci,
                        capsize=3.,
                        ecolor="k",
                        error_kw=dict(alpha=0.2, elinewidth=0.9,),
                        zorder=2,
                    )
     

            if num_vars_to_plot >= 20:
                size = kwargs.get('fontsize', self.FONT_SIZES["teensie"] - 3)
            elif num_vars_to_plot > 10:
                size = kwargs.get('fontsize', self.FONT_SIZES["teensie"] - 2)
            else:
                size = kwargs.get('fontsize', self.FONT_SIZES["teensie"] - 1) 

            # Put the variable names _into_ the plot
            if estimator_output == "probability" and method not in ['perm_based', 'coefs']:
                x_pos = 0.0
                ha = ["left"] * len(variable_names_to_plot)
            elif estimator_output == "raw" and method not in ['perm_based', 'coefs']:
                x_pos = 0.05
                ha = ["left"] * len(variable_names_to_plot)
            else:
                x_pos = 0
                ha = ["left" if score > 0 else "right" for score in scores_to_plot] 

                    
            # Put the variable names _into_ the plot
            if (method not in self.SINGLE_VAR_METHODS and plot_correlated_features):
                    results_dict = is_correlated(
                        corr_matrix, sorted_var_names, rho_threshold=rho_threshold
                    )

            # First regular is for the 'No Permutations'
            if p_values is None:
                colors = ['k'] +['k']*len(variable_names_to_plot)
            else:
                # Bold text if value is insignificant. 
                colors = ['k']+['xkcd:bright blue' if v else 'k' for v in p_values[i]]
                
            if colinear_predictors is None:
                fontweight = ['light'] +['light']*len(variable_names_to_plot)
            else:
                # Italicize text if the VIF > threshold (indicates a multicolinear predictor)
                fontweight = ['light']+['bold' if v else 'light' for v in colinear_predictors[i]]
            
            # Reverse the order since the variable names are reversed. 
            colors = colors[::-1]
            fontweight = fontweight[::-1]
                
            for i in range(len(variable_names_to_plot)):
                color = "k"
                if (method not in self.SINGLE_VAR_METHODS and plot_correlated_features):
                    correlated = results_dict.get(sorted_var_names[i], False)
                    color = "xkcd:medium green" if correlated else "k"

                if p_values is not None:
                    color = colors[i] 

                if method not in self.SINGLE_VAR_METHODS:
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
                        fontweight=fontweight[i], 
                    )

            """
            depricated. The methods used return proper importance scores. 
            if estimator_output == "probability" and "pass" in method:
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
            """

            ax.tick_params(axis="both", which="both", length=0)
            ax.set_yticks([])
                
            if estimator_output == "probability" and "pass" in method:
                upper_limit = min(1.1 * np.nanmax(scores_to_plot+ci[1,:]), 1.0)
                ax.set_xlim([0, upper_limit])
            elif ("perm_based" in method) or ('coefs' in method):
                upper_limit = max(1.1 * np.nanmax(scores_to_plot+ci[1,:]), 0.01)
                lower_limit = min(1.1 * np.nanmin(scores_to_plot-ci[0,:]), -0.01)
                ax.set_xlim([lower_limit, upper_limit])    
            else:
                upper_limit = 1.1 * np.nanmax(scores_to_plot+ci[1,:])
                ax.set_xlim([0, upper_limit])

            if xticks is not None:
                ax.set_xticks(xticks)
            else:
                self.set_n_ticks(ax, option="x")  

            # make the horizontal plot go with the highest value at the top
            # ax.invert_yaxis()
            vals = ax.get_xticks()
            for tick in vals:
                ax.axvline(
                        x=tick, linestyle="dashed", alpha=0.4, color="#eeeeee", zorder=1
                    )
  
        xlabel = self.DISPLAY_NAMES_DICT.get(method, method) if (only_one_method and xlabels is None) else ''
        major_ax = self.set_major_axis_labels(
                fig,
                xlabel=xlabel,
                ylabel_left='',
                ylabel_right='',
                title = title,
                fontsize=self.FONT_SIZES["small"],
                **kwargs,
            )
   
        pad = -0.2 if plot_correlated_features else -0.15
        diff = 0.2 if n_panels > 3 else 0.0
        ax_iterator[0].annotate(
                        "higher ranking",
                        xy=(pad, 0.8+diff),
                        xytext=(pad, 0.5),
                        arrowprops=dict(arrowstyle="->", color="xkcd:blue grey"),
                        xycoords=ax_iterator[0].transAxes,
                        rotation=90,
                        size=6,
                        ha="center",
                        va="center",
                        color="xkcd:blue grey",
                        alpha=0.65,
                    )

        ax_iterator[0].annotate(
                        "lower ranking",
                        xy=(pad + 0.05, 0.2-diff),
                        xytext=(pad + 0.05, 0.5),
                        arrowprops=dict(arrowstyle="->", color="xkcd:blue grey"),
                        xycoords=ax_iterator[0].transAxes,
                        rotation=90,
                        size=6,
                        ha="center",
                        va="center",
                        color="xkcd:blue grey",
                        alpha=0.65,
                    )

        if ylabels is not None:
            self.set_row_labels(labels=ylabels, 
                                axes=axes, 
                                pos=-1, 
                                pad=1.15, 
                                rotation=270, 
                                **kwargs)
        
        if estimator_output == "probability":
            pos = (0.9, 0.09)
        else:
            pos = (0.9, 0.9)
        self.add_alphabet_label(n_panels, axes, pos=pos)

        return fig, axes
        
    def _add_correlated_brackets(self, ax, corr_matrix, top_features, rho_threshold):
        """
        Add bracket connecting features above a given correlation threshold.
        """
        get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
        colors = get_colors(5) # sample return:  ['#8af5da', '#fbc08c', '#b741d0', '#e599f1', '#bbcb59', '#a2a6c0']
        
        _, pair_indices = find_correlated_pairs_among_top_features(
            corr_matrix,
            top_features,
            rho_threshold=rho_threshold,
        )
        colors =  get_colors(len(pair_indices))
        
        
        x = 0.0001
        dx = 0.0002
        bottom_indices = []
        top_indices = []
        for p, color in zip(pair_indices, colors):
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

            self.annotate_bars(ax, bottom_idx=bottom_idx, top_idx=top_idx, x=x, color=color)
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


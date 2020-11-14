import numpy as np
import shap

from .base_plotting import PlotStructure
from ..common.utils import combine_like_features
import matplotlib.pyplot as plt
from .dependence import dependence_plot
from matplotlib.lines import Line2D

class PlotFeatureContributions(PlotStructure):
    
    def _contribution_plot(
        self,
        dict_to_use,
        key,
        ax=None,
        to_only_varname=None,
        display_feature_names={}, 
        n_vars=12,
        other_label="Other Predictors",
    ):
        """
        Plot the feature contributions. 
        """
        contrib = []
        varnames = []

        # return nothing if dictionary is empty
        if len(dict_to_use) == 0:
            return

        for var in list(dict_to_use.keys()):
            contrib.append(dict_to_use[var])

            if to_only_varname is None:
                varnames.append(var)
            else:
                varnames.append(to_only_varname(var))

        final_pred = np.sum(contrib)

        if to_only_varname is not None:
            contrib, varnames = combine_like_features(contrib, varnames)

        bias_index = varnames.index("Bias")
        bias = contrib[bias_index]

        # Remove the bias term (neccesary for cases where
        # the data resampled to be balanced; the bias is 50% in that cases
        # and will much higher than separate contributions of the other predictors)
        varnames.pop(bias_index)
        contrib.pop(bias_index)

        varnames = np.array(varnames)
        contrib = np.array(contrib)

        varnames = np.append(varnames[:n_vars], other_label)
        contrib = np.append(contrib[:n_vars], sum(contrib[n_vars:]))

        sorted_idx = np.argsort(contrib)[::-1]
        contrib = contrib[sorted_idx]
        varnames = varnames[sorted_idx]

        bar_colors = ["xkcd:pastel red" if c > 0 else 'xkcd:powder blue' for c in contrib]
        y_index = range(len(contrib))

        # Despine
        self.despine_plt(ax)

        ax.barh(
            y=y_index, width=contrib, height=0.8, alpha=0.8, color=bar_colors, zorder=2
        )

        ax.tick_params(axis="both", which="both", length=0, labelsize=7)

        vals = ax.get_xticks()
        for tick in vals:
            ax.axvline(x=tick, linestyle="dashed", alpha=0.4, color="#eeeeee", zorder=1)

        ax.set_yticks(y_index)
        self.set_tick_labels(ax, varnames, display_feature_names)

        neg_factor = 2.25 if np.max(np.abs(contrib)) > 1.0 else 0.08
        factor = 0.25 if np.max(contrib) > 1.0 else 0.01

        for i, c in enumerate(np.round(contrib, 2)):
            if c > 0:
                ax.text(
                    c + factor,
                    i + 0.25,
                    str(c),
                    color="k",
                    fontweight="bold",
                    alpha=0.8,
                    fontsize=6,
                )
            else:
                ax.text(
                    c - neg_factor,
                    i + 0.25,
                    str(c),
                    color="k",
                    fontweight="bold",
                    alpha=0.8,
                    fontsize=6,
                )

        ax.set_xlim([np.min(contrib) - neg_factor, np.max(contrib) + factor])
        
        pos_contrib_ratio = abs(np.max(contrib)) > abs(np.min(contrib))
         
        if pos_contrib_ratio:
            ax.text(
                0.7,
                0.1,
                f"Bias : {bias:.2f}",
                fontsize=6,
                alpha=0.7,
                ha="left",
                va="center",
                ma="left",
                transform=ax.transAxes,
            )
            ax.text(
                0.7,
                0.15,
                f"Final Pred. : {final_pred:.2f}",
                fontsize=6,
                alpha=0.7,
                ha="left",
                va="center",
                ma="left",
                transform=ax.transAxes,
            )

        else:
            ax.text(
                0.1,
                0.90,
                f"Bias : {bias:.2f}",
                fontsize=6,
                alpha=0.7,
                ha="left",
                va="center",
                ma="left",
                transform=ax.transAxes,
            )
            ax.text(
                0.1,
                0.95,
                f"Final Pred. : {final_pred:.2f}",
                fontsize=6,
                alpha=0.7,
                ha="left",
                va="center",
                ma="left",
                transform=ax.transAxes,
            )

        # make the horizontal plot go with the highest value at the top
        ax.invert_yaxis()

    def plot_contributions(self, result_dict, model_names, to_only_varname=None, 
                           display_feature_names={}, **kwargs):
        """
        Plot the results of feature contributions

        Args:
        ---------------
            result : pandas.Dataframe
                a single row/example from the 
                result dataframe from tree_interpreter_simple
        """

        hspace = kwargs.get("hspace", 0.2)

        if "non_performance" in result_dict[model_names[0]].keys():
            n_panels=1
            n_columns=1
            figsize = (3, 2.5)
            wspace = kwargs.get("wspace", 0.5)
        else:
            n_panels = len(result_dict.keys()) * 4
            n_columns = 4 
            figsize= (14, 8)
            wspace = kwargs.get("wspace", 1.5)

        # create subplots, one for each feature
        fig, axes = self.create_subplots(
                    n_panels=n_panels,
                    n_columns=n_columns,
                    hspace=hspace,
                    wspace=wspace,
                    sharex=False,
                    sharey=False,
                    figsize=figsize,
                )
        
        # try for all_data/average data
        if "non_performance" in result_dict[model_names[0]].keys():
                fig = self._contribution_plot(
                    result_dict[model_name]["non_performance"],
                    to_only_varname=to_only_varname,
                    display_feature_names=display_feature_names,
                    key='',
                    ax=axes
                )
                return fig

        # loop over each model creating one panel per model
        c=0
        for i, model_name in enumerate(model_names):
            k=0
            for perf_key in result_dict[model_name].keys():
                ax = axes[i,k] 
                #print(perf_key)
                self._contribution_plot(
                        result_dict[model_name][perf_key],
                        ax=ax,
                        key=perf_key,
                        to_only_varname=to_only_varname,
                        display_feature_names=display_feature_names
                    )
                if c == 0:
                    ax.text(0.1, 1.09,
                            perf_key.replace("Forecasts ", "Forecasts\n").upper(),
                            transform = ax.transAxes,
                            fontsize=10,
                            ha='center',
                            va='center',
                            color='xkcd:darkish blue',
                            alpha=0.95)
                k+=1
            c+=1
                
        major_ax = self.set_major_axis_labels(
                fig,
                xlabel='',
                ylabel_left='',
                labelpad=5,
                fontsize=self.FONT_SIZES['tiny'],
            )
        
        self.set_row_labels(labels=model_names, 
                            axes=axes, 
                            pos=-1,
                            rotation=270, 
                            pad=1.5,
                            fontsize=12
                           )
        
        #additional_handles = [
        #                        Line2D([0], [0], color="xkcd:pastel red", alpha=0.8),
        #                         Line2D([0], [0], color='xkcd:powder blue', alpha=0.8),
        #                          ]
        
        additional_labels = ['Positive Contributions', 'Negative Contributions']
        #self.set_legend(n_panels, fig, axes[0,0], 
        #                major_ax, additional_handles, 
        #                additional_labels, bbox_to_anchor=(0.5, -0.25))
        
        self.add_alphabet_label(n_panels, axes, pos=(1.15, 0.0), fontsize=12)

        return fig
    
    def plot_shap(self, shap_values, 
                           examples, 
                           features, 
                           plot_type,
                           display_feature_names=None,
                           display_units={},
                           feature_values=None,
                           target_values=None, 
                           interaction_index="auto",
                           **kwargs
                          ):
        """
        Plot SHAP summary or dependence plot. 
        
        """
        if feature_values  is None:
            feature_values=examples.values

        if display_feature_names is None:
            self.display_feature_names = {}
        else:
            self.display_feature_names = display_feature_names
        self.display_units = display_units
        
        if plot_type == 'summary':
            shap.summary_plot(shap_values, 
                              features=examples, 
                              feature_names=display_feature_names, 
                              max_display=15, 
                              plot_type="dot",
                              alpha=1, 
                              show=False, 
                              sort=True,
                             )
            
        elif plot_type == 'dependence':
            # Set up the font sizes for matplotlib
            left_yaxis_label = 'SHAP values (%)\n(Feature Contributions)'
            n_panels = len(features)
            if n_panels <= 6:
                figsize=(8,5)
            else:
                figsize=(10,8)

            fig, axes = self.create_subplots(
                    n_panels=n_panels,
                    sharex=False,
                    sharey=False,
                    figsize=figsize,
                    wspace = 0.4,
                    hspace=0.5
                )
            
            ax_iterator = self.axes_to_iterator(n_panels, axes)

            if display_feature_names is not None:
                display_features = [display_feature_names[f] for f in self.feature_names] 
            else:
                display_features = self.feature_names
            
            for ax, feature in zip(ax_iterator, features):
                ind = self.feature_names.index(feature)
               
                dependence_plot(ind=ind, 
                                shap_values=shap_values, 
                                features=examples,
                                feature_values=feature_values,
                                display_features=display_features,
                                interaction_index=interaction_index,
                                target_values=target_values,
                                color="#1E88E5", 
                                axis_color="#333333", 
                                cmap=None,
                                dot_size=5, 
                                x_jitter=0, 
                                alpha=1, 
                                ax=ax,
                                fig=fig,
                                **kwargs)
                
                self.set_n_ticks(ax)
                self.set_minor_ticks(ax)
                self.set_axis_label(ax, xaxis_label=''.join(feature), yaxis_label='')
                ax.axhline(y=0.0, color="k", alpha=0.8, linewidth=0.8, linestyle='dashed')
                ax.set_yticks(self.calculate_ticks(ax=ax, nticks=5, center=True))
                ax.tick_params(axis='both', labelsize=8) 
                vertices = ax.collections[0].get_offsets()
                self._to_sci_notation(ax=ax, ydata=vertices[:,1], xdata=vertices[:,0], colorbar=False)

            major_ax = self.set_major_axis_labels(
                fig,
                xlabel=None,
                ylabel_left=left_yaxis_label,
                labelpad = 25,
                **kwargs,
            )
            
            self.add_alphabet_label(n_panels, axes)
            
            return fig, axes 

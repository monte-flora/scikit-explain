import numpy as np

from .base_plotting import PlotStructure

class PlotFeatureContributions(PlotStructure):
    
    def _contribution_plot(
        self,
        dict_to_use,
        key,
        ax=None,
        to_only_varname=None,
        readable_feature_names={}, 
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

        ax.tick_params(axis="both", which="both", length=0)

        vals = ax.get_xticks()
        for tick in vals:
            ax.axvline(x=tick, linestyle="dashed", alpha=0.4, color="#eeeeee", zorder=1)

        ax.set_yticks(y_index)
        self.set_tick_labels(ax, varnames, readable_feature_names)

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
                    fontsize=8,
                )
            else:
                ax.text(
                    c - neg_factor,
                    i + 0.25,
                    str(c),
                    color="k",
                    fontweight="bold",
                    alpha=0.8,
                    fontsize=8,
                )

        ax.set_xlim([np.min(contrib) - neg_factor, np.max(contrib) + factor])
        
        pos_contrib_ratio = abs(np.max(contrib)) > abs(np.min(contrib))
         
        if pos_contrib_ratio:
            ax.text(
                0.7,
                0.1,
                f"Bias : {bias:.2f}",
                fontsize=7,
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
                fontsize=7,
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
                fontsize=7,
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
                fontsize=7,
                alpha=0.7,
                ha="left",
                va="center",
                ma="left",
                transform=ax.transAxes,
            )

        # make the horizontal plot go with the highest value at the top
        ax.invert_yaxis()

    def plot_contributions(self, result_dict, to_only_varname=None, 
                           readable_feature_names={}, **kwargs):
        """
        Plot the results of feature contributions

        Args:
        ---------------
            result : pandas.Dataframe
                a single row/example from the 
                result dataframe from tree_interpreter_simple
        """

        hspace = kwargs.get("hspace", 0.4)
        wspace = kwargs.get("wspace", 1.0)

        # get the number of panels which will be the number of ML models in dictionary
        n_panels = len(result_dict.keys())

        # loop over each model creating one panel per model
        for model_name in result_dict.keys():

            # try for all_data/average data
            if "non_performance" in result_dict[model_name].keys():
                # create subplots, one for each feature
                fig, ax = self.create_subplots(
                    n_panels=1,
                    n_columns=1,
                    hspace=hspace,
                    wspace=wspace,
                    sharex=False,
                    sharey=False,
                    figsize=(3, 2.5),
                )

                fig = self._contribution_plot(
                    result_dict[model_name]["non_performance"], 
                    to_only_varname=to_only_varname,
                    readable_feature_names=readable_feature_names,
                    key='',
                    ax=ax
                )

            # must be performanced based
            else:

                # create subplots, one for each feature
                fig, axes = self.create_subplots(
                    n_panels=4,
                    n_columns=2,
                    hspace=hspace,
                    wspace=wspace,
                    sharex=False,
                    sharey=False,
                    figsize=(8, 6),
                )

                for ax, perf_key in zip(axes.flat, result_dict[model_name].keys()):
                    print(perf_key)
                    self._contribution_plot(
                        result_dict[model_name][perf_key],
                        ax=ax,
                        key=perf_key,
                        to_only_varname=to_only_varname,
                        readable_feature_names=readable_feature_names
                    )
                    ax.set_title(perf_key.upper().replace("_", " "), fontsize=15)

        return fig
    
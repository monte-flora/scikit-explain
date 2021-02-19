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
        feature_values,
        key,
        ax=None,
        to_only_varname=None,
        display_feature_names={},
        display_units={},
        n_vars=12,
        other_label="Other Predictors",
    ):
        """
        Plot the feature contributions.
        """
        contrib = []
        feat_values = []
        varnames = []

        # return nothing if dictionary is empty
        if len(dict_to_use) == 0:
            return

        for var in list(dict_to_use.keys()):
            contrib.append(dict_to_use[var])
            feat_values.append(feature_values[var])

            if to_only_varname is None:
                varnames.append(var)
            else:
                varnames.append(to_only_varname(var))

        final_pred = abs(np.sum(contrib))

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
        feat_values = np.array(feat_values)

        varnames = np.append(varnames[:n_vars], other_label)
        contrib = np.append(contrib[:n_vars], sum(contrib[n_vars:]))

        sorted_idx = np.argsort(contrib)[::-1]
        contrib = contrib[sorted_idx]
        varnames = varnames[sorted_idx]

        bar_colors = [
            "xkcd:pastel red" if c > 0 else "xkcd:powder blue" for c in contrib
        ]
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

        # ax.set_yticks(y_index)
        tick_labels = self.set_tick_labels(
            ax, varnames, display_feature_names, return_labels=True
        )

        ax.set_yticks([])

        neg_factor = 2.25 if np.max(np.abs(contrib)) > 1.0 else 0.08
        factor = 0.25 if np.max(contrib) > 1.0 else 0.01

        for i, pair in enumerate(zip(np.round(contrib, 2), varnames, tick_labels)):
            c, v, label = pair
            if v == other_label:
                text = other_label
            else:
                units = display_units.get(v, "")

                feat_val = feature_values[v]
                if feat_val <= 1 and feat_val > 0:
                    # val bewteen 0-1 -> convert to sci. notation
                    num_and_exp = f"{feat_val:.1e}".split("e")
                    base = float(num_and_exp[0])
                    exp = int(num_and_exp[1])
                    feat_val = fr"{base} $\times$ 10$^{{{exp}}}$"
                else:
                    feat_val = round(feat_val)

                special_label = label.replace(" ", " \ ").replace("$", "")
                if units == "":
                    text = fr"$\bf{special_label}$" + f" ({feat_val})"
                else:
                    text = fr"$\bf{special_label}$" + f" ({feat_val}" + f" {units})"

            if c > 0:
                # Plot the contribution value
                c + factor
                ax.text(
                    c + 0.05,
                    i + 0.05,
                    str(c),
                    color="k",
                    fontweight="bold",
                    alpha=0.8,
                    fontsize=6,
                    ha="left",
                )

                # Plots the feature name and value
                ax.text(
                    0 - 0.05,
                    i + 0.25,
                    text,
                    color="xkcd:crimson",
                    alpha=0.8,
                    fontsize=6,
                    ha="right",
                )

            else:
                # c - neg_factor
                ax.text(
                    c - 0.05,
                    i + 0.05,
                    str(c),
                    color="k",
                    fontweight="bold",
                    alpha=0.8,
                    fontsize=6,
                    ha="right",
                )
                ax.text(
                    0 + 0.05,
                    i + 0.25,
                    text,
                    color="xkcd:medium blue",
                    alpha=0.8,
                    fontsize=6,
                    ha="left",
                )

        ax.set_xlim([np.min(contrib) - neg_factor, np.max(contrib) + factor])

        pos_contrib_ratio = abs(np.max(contrib)) > abs(np.min(contrib))

        # make the horizontal plot go with the highest value at the top
        ax.invert_yaxis()

        return final_pred, bias

    def plot_contributions(
        self,
        contrib_dict,
        feature_values,
        model_names,
        to_only_varname=None,
        display_feature_names={},
        display_units={},
        model_output=None,
        **kwargs,
    ):
        """
        Plot the results of feature contributions

        Args:
        ---------------
            result : pandas.Dataframe
                a single row/example from the
                result dataframe from tree_interpreter_simple
        """
        hspace = kwargs.get("hspace", 0.2)

        only_one_model = True if len(model_names) == 1 else False

        if "non_performance" in contrib_dict[model_names[0]].keys():
            n_panels = 1
            n_columns = 1
            figsize = (3, 2.5)
            wspace = kwargs.get("wspace", 0.5)
        else:
            n_perf_keys = len(contrib_dict[model_names[0]].keys())
            n_panels = len(model_names) * n_perf_keys

            if only_one_model and model_output == "raw":
                n_columns = 1
            elif only_one_model and model_output == "probability":
                n_columns = 2
            else:
                n_columns = 4

            if n_columns == 4:
                figsize = (14, 8)
            else:
                figsize = (5, 5)

            wspace = kwargs.get("wspace", 0.5)
            hspace = kwargs.get("hspace", 0.5)

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
        if "non_performance" in contrib_dict[model_names[0]].keys():
            final_pred, bias = self._contribution_plot(
                contrib_dict[model_names[0]]["non_performance"],
                feature_values[model_names[0]]["non_performance"],
                to_only_varname=to_only_varname,
                display_feature_names=display_feature_names,
                display_units=display_units,
                key="",
                ax=axes,
            )

            axes.set_title(
                f"Prediction : {final_pred:.2f} Bias : {bias:.2f}",
                alpha=0.8,
                fontsize=self.FONT_SIZES["small"],
            )

            return None

        ax_iterator = axes.flat

        # loop over each model creating one panel per model
        c = 0
        for i, model_name in enumerate(model_names):
            for perf_key in contrib_dict[model_name].keys():
                # ax = axes[i,k]
                ax = ax_iterator[c]
                final_pred, bias = self._contribution_plot(
                    contrib_dict[model_name][perf_key],
                    feature_values[model_name][perf_key],
                    ax=ax,
                    key=perf_key,
                    to_only_varname=to_only_varname,
                    display_feature_names=display_feature_names,
                    display_units=display_units,
                )
                if i == 0:

                    ax.text(
                        0.5,
                        1.2,
                        perf_key.replace("Forecasts ", "Forecasts\n").upper(),
                        color="xkcd:darkish blue",
                        ha="center",
                        transform=ax.transAxes,
                        fontsize=8,
                    )
                    ax.text(
                        -0.05,
                        1.12,
                        f"Avg. Prediction : {final_pred:.2f}",
                        ha="left",
                        color="xkcd:bluish grey",
                        transform=ax.transAxes,
                        fontsize=7,
                    )
                    ax.text(
                        -0.05,
                        1.05,
                        f"Avg. Bias : {bias:.2f}",
                        ha="left",
                        color="xkcd:bluish grey",
                        fontsize=7,
                        transform=ax.transAxes,
                    )
                c += 1

        xlabel = (
            "Feature Contributions (%)"
            if model_output == "probability"
            else "Feature Contributions"
        )

        major_ax = self.set_major_axis_labels(
            fig,
            xlabel="Feature Contributions (%)",
            ylabel_left="",
            labelpad=5,
            fontsize=self.FONT_SIZES["normal"],
        )

        if not only_one_model:
            self.set_row_labels(
                labels=model_names,
                axes=axes,
                pos=-1,
                rotation=270,
                pad=1.5,
                fontsize=12,
            )

        # additional_handles = [
        #                        Line2D([0], [0], color="xkcd:pastel red", alpha=0.8),
        #                         Line2D([0], [0], color='xkcd:powder blue', alpha=0.8),
        #                          ]

        # additional_labels = ['Positive Contributions', 'Negative Contributions']
        # self.set_legend(n_panels, fig, axes[0,0],
        #                major_ax, additional_handles,
        #                additional_labels, bbox_to_anchor=(0.5, -0.25))

        self.add_alphabet_label(n_panels, axes, pos=(1.15, 0.0), fontsize=12)

        return fig

    def plot_shap(
        self,
        shap_values,
        examples,
        features,
        plot_type,
        display_feature_names=None,
        display_units={},
        feature_values=None,
        target_values=None,
        interaction_index="auto",
        **kwargs,
    ):
        """
        Plot SHAP summary or dependence plot.

        """
        if feature_values is None:
            feature_values = examples.values

        self.display_units = display_units

        self.display_feature_names = display_feature_names

        if display_feature_names is not None:
            display_feature_names_list = [
                display_feature_names[f] for f in self.feature_names
            ]
        else:
            display_feature_names_lists = self.feature_names

        if plot_type == "summary":
            shap.summary_plot(
                shap_values,
                features=examples,
                feature_names=display_feature_names_list,
                max_display=15,
                plot_type="dot",
                alpha=1,
                show=False,
                sort=True,
            )

        elif plot_type == "dependence":
            # Set up the font sizes for matplotlib
            self.display_feature_names = display_feature_names
            left_yaxis_label = "SHAP values (%)\n(Feature Contributions)"
            n_panels = len(features)
            if n_panels <= 6:
                figsize = (8, 5)
            else:
                figsize = (10, 8)

            fig, axes = self.create_subplots(
                n_panels=n_panels,
                sharex=False,
                sharey=False,
                figsize=figsize,
                wspace=0.4,
                hspace=0.5,
            )

            ax_iterator = self.axes_to_iterator(n_panels, axes)

            for ax, feature in zip(ax_iterator, features):
                ind = self.feature_names.index(feature)

                dependence_plot(
                    ind=ind,
                    shap_values=shap_values,
                    features=examples,
                    feature_values=feature_values,
                    display_features=display_feature_names_list,
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
                    **kwargs,
                )

                self.set_n_ticks(ax)
                self.set_minor_ticks(ax)
                self.set_axis_label(ax, xaxis_label="".join(feature), yaxis_label="")
                ax.axhline(
                    y=0.0, color="k", alpha=0.8, linewidth=0.8, linestyle="dashed"
                )
                ax.set_yticks(self.calculate_ticks(ax=ax, nticks=5, center=True))
                ax.tick_params(axis="both", labelsize=8)
                vertices = ax.collections[0].get_offsets()
                self._to_sci_notation(
                    ax=ax, ydata=vertices[:, 1], xdata=vertices[:, 0], colorbar=False
                )

            major_ax = self.set_major_axis_labels(
                fig,
                xlabel=None,
                ylabel_left=left_yaxis_label,
                labelpad=25,
                **kwargs,
            )

            self.add_alphabet_label(n_panels, axes)

            return fig, axes

import numpy as np
import math
import shap

from .base_plotting import PlotStructure
from ..common.utils import combine_like_features
import matplotlib.pyplot as plt
from .dependence import dependence_plot
from matplotlib.lines import Line2D


class PlotFeatureContributions(PlotStructure):
    def _contribution_plot(
        self,
        data,
        key,
        model_name,
        features, 
        ax=None,
        to_only_varname=None,
        display_feature_names={},
        display_units={},
        n_vars=12,
        other_label="Other Predictors",
        label_fontsize=6, 
        **kwargs
    ):
        """
        Plot the feature contributions.
        """
        all_features = data.attrs['feature_names']
        
        colors = kwargs.get('color', ["xkcd:pastel red", "xkcd:powder blue"])
        
        vars_c = [f'{var}_contrib' for var in features if 'Bias' not in var]
        vars_val = [f'{var}_val' for var in features if 'Bias' not in var]

        contribs = data.loc[key].loc[model_name, vars_c]
        feat_vals = data.loc[key].loc[model_name, vars_val]

        # Convert names 
        #if to_only_varname is not None:
        #    features = [to_only_varname(f) for f in features]
        
        #if to_only_varname is not None:
        #    contribs, features = combine_like_features(contribs, features)
       
        bias = data.loc[key].loc[model_name, 'Bias_contrib']
        final_pred = np.sum(data.loc[key].loc[model_name, [f'{var}_contrib' for var in all_features] + ['Bias_contrib']])
        
        feature_names = np.array(features)

        # Rank contributions with highest on first. 
        sorted_idx = np.argsort(contribs)[::-1]
        contribs_sorted = contribs[sorted_idx]
        feature_names_sorted = feature_names[sorted_idx]

        if other_label != None:
            feature_names_trunc = np.append(feature_names_sorted[:n_vars], other_label)
            contribs_trunc = np.append(contribs_sorted[:n_vars], sum(contribs_sorted[n_vars:]))
        else:
            feature_names_trunc = feature_names_sorted[:n_vars]
            contribs_trunc = contribs_sorted[:n_vars]
            sum_remaining_predictions = sum(contribs_sorted[n_vars:])
            
        bar_colors = [
            colors[0] if c > 0 else colors[1] for c in contribs_trunc
        ]
        y_index = range(len(contribs_trunc))

        # Despine
        self.despine_plt(ax)

        ax.barh(
            y=y_index, width=contribs_trunc, height=0.8, alpha=0.8, color=bar_colors, zorder=2
        )

        ax.tick_params(axis="both", which="both", length=0, labelsize=7)

        vals = ax.get_xticks()
        for tick in vals:
            ax.axvline(x=tick, linestyle="dashed", alpha=0.4, color="#eeeeee", zorder=1)

        # ax.set_yticks(y_index)
        tick_labels = self.set_tick_labels(
            ax, feature_names_trunc, display_feature_names, return_labels=True
        )

        ax.set_yticks([])

        neg_factor = 2.25 if np.max(np.abs(contribs_trunc)) > 1.0 else 0.08
        factor = 0.25 if np.max(contribs_trunc) > 1.0 else 0.01

        for i, pair in enumerate(zip(np.round(contribs_trunc, 2), feature_names_trunc, tick_labels)):
            c, v, label = pair
            if v == other_label:
                text = other_label
            else:
                units = display_units.get(v, "")

                feat_val = feat_vals[v+'_val']

                if feat_val <= 1 and feat_val > -1 and not math.isclose(feat_val, 0, abs_tol = 0.00001):
                        # If the value is not reasonably zero 
                        # and bewteen -1 to 1 -> convert to sci. notation
                        num_and_exp = f"{feat_val:.1e}".split("e")
                        base = float(num_and_exp[0])
                        exp = int(num_and_exp[1])
                        feat_val = fr"{base} $\times$ 10$^{{{exp}}}$"
                elif feat_val > 0.01 and feat_val <= 10:
                    feat_val = round(feat_val, 1)
                else:
                    feat_val = round(feat_val) if feat_val < 100 else int(round(feat_val))

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
                    fontsize=label_fontsize,
                    ha="left",
                )

                # Plots the feature name and value
                ax.text(
                    0 - 0.05,
                    i + 0.25,
                    text,
                    color="xkcd:crimson",
                    alpha=0.8,
                    fontsize=label_fontsize,
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
                    fontsize=label_fontsize,
                    ha="right",
                )
                ax.text(
                    0 + 0.05,
                    i + 0.25,
                    text,
                    color="xkcd:medium blue",
                    alpha=0.8,
                    fontsize=label_fontsize,
                    ha="left",
                )

        max_value = np.max(np.absolute(contribs_trunc))            
        ax.set_xlim([-max_value-neg_factor, max_value+factor])
        
        # make the horizontal plot go with the highest value at the top
        ax.invert_yaxis()

        return final_pred, bias

    def plot_contributions(
        self,
        data,
        estimator_names,
        features,
        to_only_varname=None,
        display_feature_names={},
        display_units={},
        model_output='raw',
        n_vars=12,
        other_label="Other Predictors",
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
        only_one_model = True if len(estimator_names) == 1 else False
        outer_indexs = list(set([f[0] for f in data.index.values]))
        
        if model_output=='probability' and "non_performance" not in outer_indexs:
            outer_indexs = ["Best Hits",
                                "Worst False Alarms", 
                                "Worst Misses",
                                "Best Corr. Negatives", 
                            ]
        elif model_output=='raw' and "non_performance" not in outer_indexs:
            outer_indexs = ["Least Error Predictions",
                                "Most Error Predictions"
                               ]
        perf_keys = kwargs.get('perf_keys', outer_indexs)
        
        
        
        if "non_performance" in outer_indexs:
            n_panels = len(estimator_names) 
            n_columns = len(estimator_names) 
            if n_columns==1:
                figsize = (3, 2.5)
            else:
                figsize = (8, 2.5)
            wspace = kwargs.get("wspace", 0.1)
            hspace = kwargs.get("hspace", 0.1)
            
        else:
            n_perf_keys = len(perf_keys)
            n_panels = len(estimator_names) * n_perf_keys
            wspace = kwargs.get("wspace", 0.5)
            
            if only_one_model and model_output == "raw":
                n_columns = 1
            elif only_one_model and model_output == "probability":
                n_columns = 2
            else:
                n_columns = max(2, len(estimator_names)) 

            if n_columns == 4:
                figsize = (12, 4)
            elif not only_one_model and len(outer_indexs)==4:
                figsize = (4+(0.65*len(estimator_names)), 9) 
            else:
                figsize = (8, 4)
                
            figsize = kwargs.get('figsize', figsize)
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
        
        if n_panels > 1:
            ax_iterator = axes.flat
        else:
            ax_iterator = axes 
        
        # try for all_data/average data
        if "non_performance" in outer_indexs:
            for i, model_name in enumerate(estimator_names):
                if n_panels > 1:
                    ax = ax_iterator[i]
                else:
                    ax = axes
                
                final_pred, bias = self._contribution_plot(
                    data=data,
                    features=features,
                    model_name=model_name,
                    to_only_varname=to_only_varname,
                    display_feature_names=display_feature_names,
                    display_units=display_units,
                    key="non_performance",
                    ax=ax,
                    n_vars=n_vars,
                    label_fontsize=6,
                    other_label=other_label,
                    **kwargs,
                )

                ax.set_title(
                    f"{model_name}\nPrediction : {final_pred:.2f} Bias : {bias:.2f}",
                    alpha=0.8,
                    fontsize=self.FONT_SIZES["small"],
                )
        else:
            # loop over each model creating one panel per model
            
            c = 0
            for i, model_name in enumerate(estimator_names):
                # Hard coded in to maintain correct ordering
                if model_output=='probability':
                    outer_indexs = ["Best Hits",
                                "Worst False Alarms", 
                                "Worst Misses",
                                "Best Corr. Negatives", 
                            ]
                else:
                    outer_indexs = ["Least Error Predictions",
                                "Most Error Predictions"
                               ]
                perf_keys = kwargs.get('perf_keys', outer_indexs)    
                    
            
                for k, perf_key in enumerate(perf_keys):
                    if not only_one_model:
                        ax = axes[k,i]
                    else:
                        ax = ax_iterator[c]
                    final_pred, bias = self._contribution_plot(
                        data=data,
                        ax=ax,
                        key=perf_key,
                        model_name=model_name,
                        features=features,
                        to_only_varname=to_only_varname,
                        display_feature_names=display_feature_names,
                        display_units=display_units,
                        label_fontsize=4,
                        other_label=other_label,
                        **kwargs,
                    )
                    # Add Final prediction and Bias 
                    ax.text(
                            -0.05,
                            1.12,
                            f"Avg. Prediction : {final_pred:.2f}",
                            ha="left",
                            color="xkcd:bluish grey",
                            transform=ax.transAxes,
                            fontsize=6,
                        )
                    ax.text(
                            -0.05,
                            1.05,
                            f"Avg. Bias : {bias:.2f}",
                            ha="left",
                            color="xkcd:bluish grey",
                            fontsize=6,
                            transform=ax.transAxes,
                        )
                    
                    # Add outer indexs as titles if only one model
                    if i == 0 and only_one_model:
                        ax.text(
                            0.5,
                            1.2,
                            perf_key,
                            color="xkcd:darkish blue",
                            ha="center",
                            transform=ax.transAxes,
                            fontsize=8,
                        )
                   
                    # Else add the model names as titles. 
                    elif k==0:
                        ax.text(
                            0.5,
                            1.2,
                            model_name,
                            color="xkcd:darkish blue",
                            ha="center",
                            transform=ax.transAxes,
                            fontsize=8,
                        )
                        
                    
                    c += 1

        xlabel = (
            "Feature Contributions (%)"
            if model_output == "probability"
            else "Feature Contributions"
        )

        labelpad = 2.5 if "non_performance" in outer_indexs else 5 
        
        major_ax = self.set_major_axis_labels(
            fig,
            xlabel=xlabel,
            ylabel_left="",
            labelpad=labelpad,
            fontsize=self.FONT_SIZES["normal"],
        )

        if "non_performance" not in outer_indexs and not only_one_model:
            self.set_row_labels(
                labels=outer_indexs,
                axes=axes,
                pos=-1,
                rotation=270,
                pad=1.5,
                fontsize=self.FONT_SIZES["small"],
                )
            
        if 'non_performance' in outer_indexs:
            pos = (0.95, 0.05)
        else:
            pos=(1.15, -0.025)
        
        self.add_alphabet_label(n_panels, axes, pos=pos, fontsize=10)

        return fig, axes

    def plot_shap(
        self,
        shap_values,
        X,
        features,
        plot_type,
        display_feature_names={},
        display_units={},
        feature_values=None,
        target_values=None,
        interaction_index="auto",
        **kwargs,
    ):
        """
        Plot SHAP summary or dependence plot.

        """
        self.display_units = display_units
        self.display_feature_names = display_feature_names

        display_feature_names_list = [
                display_feature_names.get(f,f) for f in self.feature_names
            ]
        
        if plot_type == "summary":
            shap.summary_plot(
                shap_values,
                features=X,
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
                dependence_plot(
                    feature=feature,
                    shap_values=shap_values,
                    X=X,
                    display_feature_names=display_feature_names_list,
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

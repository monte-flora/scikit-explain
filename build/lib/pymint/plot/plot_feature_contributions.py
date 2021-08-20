import numpy as np
import math
import shap

from .base_plotting import PlotStructure
from ..common.utils import combine_like_features
import matplotlib.pyplot as plt
from .dependence import dependence_plot
from matplotlib.lines import Line2D
import matplotlib

import re
from shap.plots import colors
def format_value(s, format_str):
    """ Strips trailing zeros and uses a unicode minus sign.
    """
    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = u"\u2212" + s[1:]
    return s


def waterfall(data,key,model_name, features, ax=None, fig=None, display_feature_names={}, display_units={},
              label_fontsize=8, **kwargs):
    
    """ Plots an explantion of a single prediction as a waterfall plot.
    The SHAP value of a feature represents the impact of the evidence provided by that feature on the model's
    output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature
    move the model output from our prior expectation under the background data distribution, to the final model
    prediction given the evidence of all the features. Features are sorted by the magnitude of their SHAP values
    with the smallest magnitude features grouped together at the bottom of the plot when the number of features
    in the models exceeds the max_display parameter.
    
    Parameters
    ----------
    shap_values : Explanation
        A one-dimensional Explanation object that contains the feature values and SHAP values to plot.
    max_display : str
        The maximum number of features to plot.
    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """
    max_display = kwargs.get('max_display')
    all_features = data.attrs['feature_names']
    feature_names = np.array([display_feature_names.get(f,f) for f in features])
    units = np.array([display_units.get(f,"") for f in features]) 
    
    vars_c = [f'{var}_contrib' for var in features if 'Bias' not in var]
    vars_val = [f'{var}_val' for var in features if 'Bias' not in var]

    values = data.loc[key].loc[model_name, vars_c]
    features = data.loc[key].loc[model_name, vars_val]
    base_values = data.loc[key].loc[model_name, 'Bias_contrib']
    
    lower_bounds = getattr(values, "lower_bounds", None)
    upper_bounds = getattr(values, "upper_bounds", None)
    
    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for i in range(num_features + 1)]

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            ax.plot([loc, loc], [rng[i] -1 - 0.4, rng[i] + 0.4], color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            if abs(features[order[i]]) < 1 :
                fmt = "%0.03f"
            elif abs(features[order[i]]) > 10:
                fmt = "%0.f"
            else:
                fmt = "%0.02f"
            yticklabels[rng[i]] = format_value(features[order[i]], fmt) + " " + units[order[i]] + " = " + \
            feature_names[order[i]] 
    
    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = "%d other features" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
            c = colors.red_rgb
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)
            c = colors.blue_rgb

    points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + list(np.array(neg_lefts) + np.array(neg_widths))
    dataw = np.max(points) - np.min(points)
    
    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
    ax.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw, 
            left=np.array(pos_lefts) - 0.01*dataw, color=colors.red_rgb, alpha=0)
    label_padding = np.array([-0.1*dataw  if -w < 1 else 0 for w in neg_widths])
    ax.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw, 
            left=np.array(neg_lefts) + 0.01*dataw, color=colors.blue_rgb, alpha=0)
    
    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = ax.get_xlim()[1] - ax.get_xlim()[0]
    xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()
    
    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = ax.arrow(
            pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
            head_length=min(dist, hl_scaled),
            color=colors.red_rgb, width=bar_width,
            head_width=bar_width
        )
        
        if pos_low is not None and i < len(pos_low):
            ax.errorbar(
                pos_lefts[i] + pos_widths[i], pos_inds[i], 
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=colors.light_red_rgb
            )

        txt_obj = ax.text(
            pos_lefts[i] + 0.5*dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        
        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
            
            txt_obj = ax.text(
                pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=colors.red_rgb,
                fontsize=label_fontsize
            )
    
    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]
        
        arrow_obj = ax.arrow(
            neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
            head_length=min(-dist, hl_scaled),
            color=colors.blue_rgb, width=bar_width,
            head_width=bar_width
        )

        if neg_low is not None and i < len(neg_low):
            ax.errorbar(
                neg_lefts[i] + neg_widths[i], neg_inds[i], 
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=colors.light_blue_rgb
            )
        
        txt_obj = ax.text(
            neg_lefts[i] + 0.5*dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=label_fontsize
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        
        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
            
            txt_obj = ax.text(
                neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                fontsize=label_fontsize
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax.set_yticks(list(range(num_features)) + list(np.arange(num_features)+1e-8) ,)
    ax.set_yticklabels(yticklabels[:-1] + [l.split('=')[-1] for l in yticklabels[:-1]], fontsize=label_fontsize)
    
    # put horizontal lines for each feature row
    for i in range(num_features):
        ax.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
    
    # mark the prior expected value and the model prediction
    ax.axvline(base_values, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    ax.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    
    # clean up the main axis
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #ax.tick_params(labelsize=13)
    #pl.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin,xmax = ax.get_xlim()
    ax2=ax.twiny()
    ax2.set_xlim(xmin,xmax)
    ax2.set_xticks([base_values, base_values+1e-8]) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax2.set_xticklabels(["\n$E[f(X)]$","\n$ = "+format_value(base_values, "%0.03f")+"$"], 
                        fontsize=label_fontsize, ha="left")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # draw the f(x) tick mark
    ax3=ax2.twiny()
    ax3.set_xlim(xmin,xmax)
    ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8]) 
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    
    ax3.set_xticklabels(["$f(x)$","$ = "+format_value(fx, "%0.03f")+"$"], 
                        fontsize=label_fontsize, ha="left")
    
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-10/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(12/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_color("#999999")
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-20/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(22/72., -1/72., fig.dpi_scale_trans))
    
    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")
         
    return fx, base_values


class PlotFeatureContributions(PlotStructure):
    """
    PlotFeatureContributions handles plotting contribution-based plotting for 
    single examples or some subset. This class also handles plotting SHAP-style 
    plots, which include summary and dependence plots. 
    """
    def __init__(self, BASE_FONT_SIZE=12):
        super().__init__(BASE_FONT_SIZE=BASE_FONT_SIZE)
    
    def plot_contributions(
        self,
        data,
        estimator_names,
        features,
        to_only_varname=None,
        display_feature_names={},
        display_units={},
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
        kwargs["max_display"] = kwargs.get('max_display', 10) 
        estimator_output = kwargs.get('estimator_output', None) 
        
        only_one_model = True if len(estimator_names) == 1 else False
        outer_indexs = list(set([f[0] for f in data.index.values]))
        
        if estimator_output=='probability' and "non_performance" not in outer_indexs:
            outer_indexs = ["Best Hits",
                                "Worst False Alarms", 
                                "Worst Misses",
                                "Best Corr. Negatives", 
                            ]
        elif estimator_output=='raw' and "non_performance" not in outer_indexs:
            outer_indexs = ["Least Error Predictions",
                                "Most Error Predictions"
                               ]
        perf_keys = kwargs.get('perf_keys', outer_indexs)
        kwargs["wspace"] = kwargs.get("wspace", 0.75)
        kwargs["hspace"] = kwargs.get("hspace", 0.9)
        
        n_perf_keys = len(perf_keys)
        n_panels = len(estimator_names) if "non_performance" in outer_indexs else len(estimator_names) * n_perf_keys
            
        figsize = (3, 2.5) if (n_panels == 1 and "non_performance" not in outer_indexs) else (8, 6.5)
        kwargs["figsize"] = kwargs.get("figsize", figsize)    
        n_columns = 1
        additional_columns = 1 if (n_panels>1 and not (estimator_output=='raw' and "non_performance")) else 0
        
        n_columns += additional_columns
        kwargs["n_columns"] = kwargs.get("n_columns", n_columns)
        
        # create subplots, one for each feature
        fig, axes = self.create_subplots(
            n_panels=n_panels,
            sharex=False,
            sharey=False,
            **kwargs,
        )
        
        ax_iterator = axes.flat if n_panels > 1 else axes 
        
        # try for all_data/average data
        if "non_performance" in outer_indexs:
            for i, model_name in enumerate(estimator_names):
                if n_panels > 1:
                    ax = ax_iterator[i]
                else:
                    ax = axes
                #self._contribution
                final_pred, bias = waterfall(
                    data=data,
                    features=features,
                    model_name=model_name,
                    to_only_varname=to_only_varname,
                    display_feature_names=display_feature_names,
                    display_units=display_units,
                    key="non_performance",
                    ax=ax,
                    fig=fig,
                    label_fontsize=self.FONT_SIZES["tiny"],
                    **kwargs,
                )

                if not only_one_model:
                    ax.set_title(
                        model_name,
                        alpha=0.8,
                        fontdict = {'fontsize' : self.FONT_SIZES["teensie"]},
                    )
                    
        else:
            # Hard coded in to maintain correct ordering
            if estimator_output=='probability':
                outer_indexs = ["Best Hits",
                                "Worst False Alarms", 
                                "Worst Misses",
                                "Best Corr. Negatives", 
                            ]
            else:
                outer_indexs = ["Least Error Predictions",
                                "Most Error Predictions"
                               ]
            # loop over each model creating one panel per model
            c = 0
            for i, model_name in enumerate(estimator_names):
                perf_keys = kwargs.get('perf_keys', outer_indexs)    
                for k, perf_key in enumerate(perf_keys):
                    ax  = axes[k,i] if not only_one_model else ax_iterator[c]
                    waterfall(
                        data=data,
                        ax=ax,
                        fig=fig, 
                        key=perf_key,
                        model_name=model_name,
                        features=features,
                        display_feature_names=display_feature_names,
                        display_units=display_units,
                        label_fontsize=self.FONT_SIZES["tiny"],
                        **kwargs,
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
                            fontsize=self.FONT_SIZES["small"],
                        )
                   
                    # Else add the model names as titles. 
                    elif k==0:
                        ax.set_title(
                            model_name,
                            color="xkcd:darkish blue",
                            fontdict={'fontsize' : 20},
                        )
                        
                    c += 1
                    
        if "non_performance" not in outer_indexs and not only_one_model:
            self.set_row_labels(
                labels=outer_indexs,
                axes=axes,
                pos=-1,
                rotation=270,
                pad=1.5,
                fontsize=self.FONT_SIZES["tiny"],
                )
            
        if 'non_performance' in outer_indexs:
            pos = (0.95, 0.05)
        else:
            pos=(1.15, -0.025)
        
        self.add_alphabet_label(n_panels, axes, pos=pos, fontsize=self.FONT_SIZES['tiny'])

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

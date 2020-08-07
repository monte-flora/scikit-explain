#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MaxNLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

from .utils import combine_like_features, is_outlier
import shap 

# Setting the font style to serif
rcParams['font.family'] = 'serif'

# Set up the font sizes for matplotlib
FONT_SIZE = 16
BIG_FONT_SIZE = FONT_SIZE + 2
LARGE_FONT_SIZE = FONT_SIZE + 4
HUGE_FONT_SIZE = FONT_SIZE + 6
SMALL_FONT_SIZE = FONT_SIZE - 2
TINY_FONT_SIZE = FONT_SIZE - 4
TEENSIE_FONT_SIZE = FONT_SIZE - 8
font_sizes = {
    "teensie": TEENSIE_FONT_SIZE,
    "tiny": TINY_FONT_SIZE,
    "small": SMALL_FONT_SIZE,
    "normal": FONT_SIZE,
    "big": BIG_FONT_SIZE,
    "large": LARGE_FONT_SIZE,
    "huge": HUGE_FONT_SIZE,
}
plt.rc("font", size=FONT_SIZE)
plt.rc("axes", titlesize=FONT_SIZE)
plt.rc("axes", labelsize=FONT_SIZE)
plt.rc("xtick", labelsize=TINY_FONT_SIZE - 4)
plt.rc("ytick", labelsize=TEENSIE_FONT_SIZE)
plt.rc("legend", fontsize=TEENSIE_FONT_SIZE - 1)
plt.rc("figure", titlesize=BIG_FONT_SIZE)

line_colors = ["xkcd:fire engine red", 
               "xkcd:water blue", 
               "xkcd:medium green", 
               "xkcd:very dark purple", 
               "xkcd:burnt sienna"]


class InterpretabilityPlotting:
    def create_subplots(self, n_panels, **kwargs):
        """
        Create a series of subplots (MxN) based on the 
        number of panels and number of columns (optionally)
        """
        figsize = kwargs.get("figsize", (6.4, 4.8))
        wspace = kwargs.get("wspace", 0.4)
        hspace = kwargs.get("hspace", 0.3)
        sharex = kwargs.get("sharex", False)
        sharey = kwargs.get("sharey", False)

        delete = True
        if n_panels <= 3:
            n_columns = n_panels
            delete = False
        else:
            n_columns = kwargs.get("n_columns", 3)

        n_rows = int(n_panels / n_columns)
        extra_row = 0 if (n_panels % n_columns) == 0 else 1

        fig, axes = plt.subplots(
            n_rows + extra_row,
            n_columns,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            dpi=300,
        )
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

        if delete:
            n_axes_to_delete = len(axes.flat) - n_panels

            if n_axes_to_delete > 0:
                for i in range(n_axes_to_delete):
                    fig.delaxes(axes.flat[-(i + 1)])

        return fig, axes

    def set_major_axis_labels(
        self, fig, xlabel=None, ylabel_left=None, ylabel_right=None, **kwargs
    ):
        """
        Generate a single X- and Y-axis labels for 
        a series of subplot panels 
        """

        fontsize = kwargs.get("fontsize", 15)
        labelpad = kwargs.get("labelpad", 30)

        # add a big axis, hide frame
        ax = fig.add_subplot(111, frameon=False)

        # hide tick and tick label of the big axis
        plt.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )

        # set axes labels
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel(ylabel_left, fontsize=fontsize, labelpad=labelpad)

        if ylabel_right is not None:
            ax_right = fig.add_subplot(1, 1, 1, sharex=ax, frameon=False)
            plt.tick_params(
                labelcolor="none", top=False, bottom=False, left=False, right=False
            )

            ax_right.yaxis.set_label_position("right")
            ax_right.set_ylabel(ylabel_right, labelpad=labelpad, fontsize=fontsize)
        
        return ax
    
    def set_row_labels(self, labels, axes, pos=-1):
        """
        Give a label to each row in a series of subplots
        """
        pad=1.15
        
        if np.ndim(axes) == 2:
            iterator = axes[:,pos]
        else:
            iterator = [axes[pos]]
        
        
        for ax, row in zip(iterator, labels):
            ax.yaxis.set_label_position("right")
            ax.annotate(row, xy=(1, 1), xytext=(pad, 0.5), xycoords = ax.transAxes, rotation=270,
                    size=8, ha='center', va='center', color='xkcd:vermillion', alpha=0.65)
    

    def add_alphabet_label(self, axes):
        """
        A alphabet character to each subpanel.
        """
        alphabet_list = [chr(x) for x in range(ord("a"), ord("z") + 1)]

        if len(axes) == 1:
            iterator = axes
        else:
            iterator = axes.flat
    
        for i, ax in enumerate(iterator):
            ax.text(
                    0.9,
                    0.09,
                    f"({alphabet_list[i]})",
                    fontsize=10,
                    alpha=0.8,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

    def add_histogram_axis(self, ax, data, bins=15, min_value=None, 
            max_value=None, density=False, **kwargs):
        """
        Adds a background histogram of data for a given feature. 
        """

        color = kwargs.get("color", "lightblue")
        edgecolor = kwargs.get("color", "white")

        #if min_value is not None and max_value is not None:
        #    data = np.clip(data, a_min=min_value, a_max=max_value)
        
        cnt, bins, patches = ax.hist(
            data,
            bins=bins,
            alpha=0.3,
            color=color,
            density=density,
            edgecolor=edgecolor,
            zorder=1
        )
        ax.set_yscale("log")
        ax.set_yticks([1e0, 1e1, 1e2, 1e3, 1e4])
        if density:
            return "Relative Frequency"
        
        return "Frequency"

    def line_plot(self, ax, xdata, ydata, label, **kwargs):
        """
        Plots a curve of data
        """

        linewidth = kwargs.get("linewidth", 2.0)
        linestyle = kwargs.get("linestyle", "-")

        if "color" not in kwargs:
            kwargs["color"] = blue

        ax.plot(xdata, ydata, linewidth=linewidth, linestyle=linestyle, 
                label=label, alpha=0.8, **kwargs, zorder=2)

    def confidence_interval_plot(self, ax, xdata, ydata, label, **kwargs):
        """
        Plot a line plot with an optional confidence interval polygon
        """

        facecolor = kwargs.get("facecolor", "r")
        color = kwargs.get("color", "blue")

        # get mean curve
        mean_ydata = np.mean(ydata, axis=0)

        # plot mean curve
        self.line_plot(ax, xdata, mean_ydata, color=color, label=label)

        # get confidence interval bounds
        lower_bound, upper_bound = np.percentile(ydata, [2.5, 97.5], axis=0)

        # fill between CI bounds
        ax.fill_between(xdata, lower_bound, upper_bound, facecolor=facecolor, alpha=0.4)

    def calculate_ticks(self, ax, nticks, round_to=1, center=False):
        """
        Calculate the y-axis ticks marks for the line plots
        """
        upperbound = round(ax.get_ybound()[1], round_to)
        lowerbound = round(ax.get_ybound()[0], round_to)
        
        max_value = max(abs(upperbound), abs(lowerbound))
        if max_value > 10:
            round_to = 0
 
        def round_to_a_base(a_number, base=5):
            return base * round(a_number/base)

        if max_value > 5:
            max_value = round_to_a_base(max_value)
        
        if center:
            values = np.linspace(-max_value, max_value, nticks)
            values = np.round(values, round_to)
        else:
            dy = upperbound - lowerbound
            fit = np.floor(dy / (nticks - 1)) + 1
            dy_new = (nticks - 1) * fit
            values = np.linspace(lowerbound, lowerbound + dy_new, nticks)
            values = np.round(values, round_to)
            
        return values 

    def make_twin_ax(self, ax):
        """
        Create a twin axis on an existing axis with a shared x-axis
        """
        # align the twinx axis
        twin_ax = ax.twinx()
        
        # Turn twin_ax grid off.
        twin_ax.grid(False)

        # Set ax's patch invisible
        ax.patch.set_visible(False)
        # Set axtwin's patch visible and colorize it in grey
        twin_ax.patch.set_visible(True)

        # move ax in front
        ax.set_zorder(twin_ax.get_zorder() + 1)
        
        return twin_ax
    
    def set_axis_label(self, ax, xaxis_label=None, yaxis_label = None):
        """
        Setting the x- and y-axis labels.
        """
        if xaxis_label is not None: 
            xaxis_label_pretty = self.readable_feature_names.get(xaxis_label, xaxis_label)
            units = self.feature_units.get(xaxis_label, '')
            if units == '':
                xaxis_label_with_units = fr'{xaxis_label_pretty}'
            else:
                xaxis_label_with_units = fr'{xaxis_label_pretty} ({units})'
        
            ax.set_xlabel(xaxis_label_with_units, fontsize=8)
        
        if yaxis_label is not None: 
            yaxis_label_pretty = self.readable_feature_names.get(yaxis_label, yaxis_label)
            units = self.feature_units.get(yaxis_label, '')
            if units == '':
                yaxis_label_with_units = fr'{yaxis_label_pretty}'
            else:
                yaxis_label_with_units = fr'{yaxis_label_pretty} ({units})'

            ax.set_ylabel(yaxis_label_with_units, fontsize=10)
        
    def set_legend(self, n_panels, fig, ax, major_ax):
        """
        Set a single legend for the plots. 
        """
        handles, labels = ax.get_legend_handles_labels()
        
        if n_panels > 3:
            bbox_to_anchor=(0.5, -0.3)
        else:
            bbox_to_anchor=(0.5, -0.5)
        
        # Shrink current axis's height by 10% on the bottom
        box = major_ax.get_position()
        major_ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

        # Put a legend below current axis
        major_ax.legend(handles, labels, loc='lower center', 
                        bbox_to_anchor=bbox_to_anchor,
                        fancybox=True, shadow=True, 
                        ncol=3
                       )
        
    def set_minor_ticks(self, ax):
        """
        Adds minor ticks to the x- and y-axis. 
        """
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
    def set_n_ticks(self, ax):
        """
        Set the max number of ticks per x- and y-axis.
        """
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        
    def plot_1d_curve(self, 
                      feature_dict,
                      features, 
                      model_names, 
                      shap_values=None,
                      readable_feature_names={}, 
                      feature_units={}, 
                      unnormalize=None, 
                      **kwargs):
        """
        Generic function for 1-D ALE and PD.
        """
        self.readable_feature_names = readable_feature_names
        self.feature_units = feature_units
        ci_plot = kwargs.get("ci_plot", False)
        hspace = kwargs.get("hspace", 0.5)
        facecolor = kwargs.get("facecolor", "gray")
        left_yaxis_label = kwargs.get("left_yaxis_label")
        add_zero_line = kwargs.get("add_zero_line", False)

        # get the number of panels which will be length of feature dictionary
        n_panels = len(feature_dict.keys())
        
        majoraxis_fontsize = 10
        
        if n_panels == 1:
            kwargs['figsize'] = (3,2.5)
        elif n_panels == 2:
            kwargs['figsize'] = (6, 2.5)
        elif n_panels == 3: 
            kwargs['figsize'] = (6, 3)
            hspace = 0.6
        else:
            kwargs['figsize'] = kwargs.get("figsize", (8,5))
            majoraxis_fontsize = 15
        
        # create subplots, one for each feature
        fig, axes = self.create_subplots(
            n_panels=n_panels, hspace=hspace, **kwargs
        )

        if n_panels == 1:
            iterator = [axes]
        else:
            iterator = axes.flat
        
        # loop over each feature and add relevant plotting stuff
        for lineplt_ax, feature in zip(iterator, features):

            xdata = feature_dict[feature][model_names[0]]["xdata1"]
            hist_data = feature_dict[feature][model_names[0]]["hist_data"]
            if unnormalize:
                hist_data = unnormalize(hist_data)
            # add histogram
            hist_ax = self.make_twin_ax(lineplt_ax)
            twin_yaxis_label=self.add_histogram_axis(hist_ax, hist_data, 
                                                     min_value=xdata[0], 
                                                     max_value=xdata[-1])
    
            for i, model_name in enumerate(model_names):

                ydata = feature_dict[feature][model_name]["values"]
                
                # depending on number of bootstrap examples, do CI plot or just mean
                if ydata.shape[0] > 1:
                    self.confidence_interval_plot(
                        lineplt_ax,
                        xdata,
                        ydata,
                        color=line_colors[i],
                        facecolor=line_colors[i],
                        label=model_name
                    )
                else:
                    self.line_plot(lineplt_ax, xdata, ydata[0, :], 
                                   color=line_colors[i], label=model_name.replace('Classifier',''))
   
            self.set_n_ticks(lineplt_ax)
            self.set_minor_ticks(lineplt_ax)
            self.set_axis_label(lineplt_ax, xaxis_label=''.join(feature))
            if add_zero_line:
                lineplt_ax.axhline(y=0.0, color="k", alpha=0.8, linewidth=0.8, linestyle='dashed')
            lineplt_ax.set_yticks(self.calculate_ticks(lineplt_ax, 5, center=True))

        kwargs['fontsize'] = majoraxis_fontsize
        major_ax = self.set_major_axis_labels(
            fig,
            xlabel=None,
            ylabel_left=left_yaxis_label,
            ylabel_right = twin_yaxis_label,
            **kwargs,
        )
        self.set_legend(n_panels, fig, lineplt_ax, major_ax)
        self.add_alphabet_label(axes)
        
        return fig, axes

    def plot_contours(self, model_names, feature_dict, readable_feature_names={}, 
                      feature_units={}, **kwargs):

        """
        Generic function for 2-D PDP
        """
        self.readable_feature_names = readable_feature_names
        self.feature_units = feature_units
        
        hspace = kwargs.get("hspace", 0.5)
        wspace = kwargs.get("wspace", 0.6)
        
        cmap = "bwr" 
        colorbar_label = kwargs.get("left_yaxis_label")

        # get the number of panels which will be length of feature dictionary
        n_panels = len(feature_dict.keys())

        # create subplots, one for each feature
        fig, axes = self.create_subplots(
            n_panels=n_panels, hspace=hspace, wspace=0.6, figsize=(6, 3)
        )
        
        max_z = [ ]
        min_z = [ ]
        for ax, feature in zip(axes.flat, feature_dict.keys()):

            zdata = feature_dict[feature][model_names[0]]["values"]
            max_z.append(np.max(np.mean(zdata,axis=0)))
            min_z.append(np.min(np.mean(zdata,axis=0)))
        
            peak = max(abs(np.min(min_z)), abs(np.max(max_z)))
            print(-peak, peak)
            colorbar_rng = np.linspace(-peak,peak, 10)
       
        # loop over each feature and add relevant plotting stuff
        for ax, feature in zip(axes.flat, feature_dict.keys()):
            model_names = list(feature_dict[feature].keys())
            xdata1 = feature_dict[feature][model_names[0]]["xdata1"]
            xdata2 = feature_dict[feature][model_names[0]]["xdata2"]
            zdata = feature_dict[feature][model_names[0]]["values"]

            # can only do a contour plot with 2-d data
            x1, x2 = np.meshgrid(xdata1, xdata2)

            # Get the mean of the bootstrapping. 
            if zdata.shape[0] > 1:
                zdata = np.mean(zdata, axis=0)
            
            cf = ax.contourf(x1, x2, zdata.squeeze(), cmap=cmap, levels=colorbar_rng, alpha=0.75)
            #ax.contour(x1, x2, zdata.squeeze(), levels=colorbar_rng, alpha=0.5, 
            #           linewidths=0.5, colors='k')

            self.set_minor_ticks(ax)
            self.set_axis_label(ax, 
                                xaxis_label=feature[0], 
                                yaxis_label=feature[1]
                               )
         
        fig.suptitle(model_names[0].replace('Classifier', ''), x=0.5, y=1.05, fontsize=12)
        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), 
                            shrink=0.65,
                            orientation = 'horizontal',
                            label = colorbar_label,
                            pad=0.335,
                           )

        return fig, axes
    

    def plot_shap(self, shap_values, examples, 
                  features=None, 
                  display_feature_names=None,
                  plot_type='summary', **kwargs):
        """
        """
        plt.rc("xtick", labelsize=TINY_FONT_SIZE)
        hspace = kwargs.get("hspace", 0.5)
        
        original_column_names = list(examples.columns)
        
        if display_feature_names is None:
            display_feature_names = list(examples.columns)
        else:
            examples = examples.to_numpy()
        
        if plot_type == 'summary':
            shap.summary_plot(shap_values =shap_values, 
                         features = examples, 
                         feature_names = display_feature_names,
                         max_display=20,
                         plot_size=0.5,
                         plot_type='dot',
                         show=False)
        else:
            n_panels = len(features)
            
            print(n_panels)
            
            # create subplots, one for each feature
            fig, axes = self.create_subplots(
                n_panels=n_panels, hspace=hspace, **kwargs
            )      
            
            if n_panels == 1:
                ax_iterator = [axes]
            else:
                ax_iterator = axes.flat
            
            for ax, feature in zip(ax_iterator, features):
                
                print('Processing : ', feature)
                
                # summarize the effects of all the features
                feature_idx = original_column_names.index(feature)
                
                shap.dependence_plot(
                     shap_values=shap_values[:,feature_idx], 
                     features=examples,
                     feature_names=display_feature_names,
                     interaction_index=None,
                     ax=ax, 
                     show=False)
                
            self.add_alphabet_label(axes)
                  
    def set_tick_labels(self, ax, feature_names, readable_feature_names):
        """
        Setting the tick labels for the tree interpreter plots. 
        """
        labels = [readable_feature_names.get(feature_name, 
                                             feature_name) 
                  for feature_name in feature_names ]
        labels = [fr'{l}' for l in labels] 
        ax.set_yticklabels(labels)
    
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
            try:
                contrib.append(dict_to_use[var]["Mean Contribution"])
            except:
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
                0.685,
                0.1,
                f"Bias : {bias:.2f}",
                fontsize=7,
                alpha=0.7,
                ha="center",
                va="center",
                ma="left",
                transform=ax.transAxes,
            )
            ax.text(
                0.75,
                0.15,
                f"Final Pred. : {final_pred:.2f}",
                fontsize=7,
                alpha=0.7,
                ha="center",
                va="center",
                ma="left",
                transform=ax.transAxes,
            )

        else:
            ax.text(
                0.2,
                0.90,
                f"Bias : {bias:.2f}",
                fontsize=7,
                alpha=0.7,
                ha="center",
                va="center",
                ma="left",
                transform=ax.transAxes,
            )
            ax.text(
                0.25,
                0.95,
                f"Final Pred. : {final_pred:.2f}",
                fontsize=7,
                alpha=0.7,
                ha="center",
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
    
    def despine_plt(self, ax):
        """
        remove all four spines of plot
        """
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        
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
        
    def plot_variable_importance(
        self,
        importance_dict_set,
        model_names,
        multipass=True,
        readable_feature_names={},
        feature_colors=None,
        relative=False,
        num_vars_to_plot=15,
        metric=None,
        **kwargs,
    ):

        """Plots any variable importance method for a particular estimator
        
        Args:
            importance_dict_set : list 
            multipass : boolean
                if True, plots the multipass results
            readable_feature_names : dict
                A dict mapping feature names to readable, "pretty" feature names
            feature_colors : dict
                A dict mapping features to various colors. Helpful for color coding groups of features
            relative : boolean 
            num_vars_to_plot : int
                Number of top variables to plot (defalut is None and will use number of multipass results)
            metric : str
                Metric used to compute the predictor importance, which will display as the X-axis label. 
        """
        if not isinstance(importance_dict_set, list):
            importance_dict_set = [importance_dict_set]

        hspace = kwargs.get("hspace", 0.5)
        wspace = kwargs.get("wspace", 0.2)
        xticks = kwargs.get("xticks", [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ylabels = kwargs.get('ylabels', '')
        title = kwargs.get('title', '')

        # get the number of panels which will be the number of ML models in dictionary
        n_keys = [list(importance_dict.keys()) for importance_dict in importance_dict_set]
        n_panels = len([item for sublist in n_keys for item in sublist])

        if n_panels == 1:
            figsize = (3,2.5)
        elif n_panels == 2:
            figsize = (6,2.5)
        else:
            figsize = kwargs.get("figsize", (8,2))
        
        # create subplots, one for each feature
        fig, axes = self.create_subplots(
            n_panels=n_panels, hspace=hspace, wspace=wspace, figsize=figsize
        )
        
        if n_panels==1:
            axes = [axes]
        
        for g, importance_dict in enumerate(importance_dict_set):
            # loop over each model creating one panel per model
            for k, model_name in enumerate(model_names):
                if len(importance_dict_set) == 1:
                        ax = axes[k]
                else:
                    ax = axes[g,k]
                if g == 0:
                    ax.set_title(model_name, fontsize=12, alpha=0.8)
                # Despine
                self.despine_plt(ax)
               
                importance_obj = importance_dict[model_name]

                rankings = (
                    importance_obj.retrieve_multipass()
                    if multipass
                    else importance_obj.retrieve_singlepass()
                    )

                if num_vars_to_plot is None:
                    num_vars_to_plot == len(list(rankings.keys()))
    
                # Get the original score (no permutations)
                original_score = importance_obj.original_score
                # Check if the permutation importance is bootstrapped
                bootstrapped, original_score_mean = self.is_bootstrapped(original_score)

                # Sort by increasing rank
                sorted_var_names = list(rankings.keys())
                sorted_var_names.sort(key=lambda k: rankings[k][0])
                sorted_var_names = sorted_var_names[: min(num_vars_to_plot, len(rankings))]
                scores = [rankings[var][1] for var in sorted_var_names]

                # Get the colors for the plot
                colors_to_plot = [
                    self.variable_to_color(var, feature_colors)
                    for var in ["No Permutations",] + sorted_var_names
                ]
                # Get the predictor names
                variable_names_to_plot = [" {}".format(var)
                    for var in self.convert_vars_to_readable(
                        ["No Permutations",] + sorted_var_names, readable_feature_names
                        )
                        ]

                if bootstrapped:
                    if relative:
                        scores_to_plot = (
                            np.array(
                                [original_score_mean,]
                                + [np.mean(score) for score in scores]
                            )
                            / original_score_mean
                        )
                    else:
                        scores_to_plot = np.array(
                            [original_score_mean,] + [np.mean(score) for score in scores]
                        )
                    ci = np.array(
                        [
                            np.abs(np.mean(score) - np.percentile(score, [2.5, 97.5]))
                            for score in np.r_[[original_score,], scores]
                        ]
                    ).transpose()
                else:
                    if relative:
                        scores_to_plot = (
                            np.array([original_score_mean,] + scores) / original_score_mean
                        )
                    else:
                        scores_to_plot = np.array([original_score_mean,] + scores)
                    ci = np.array(
                        [[0, 0] for score in np.r_[[original_score,], scores]]
                    ).transpose()

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

                # Put the variable names _into_ the plot
                for i in range(len(variable_names_to_plot)):
                    ax.text(
                        0,
                        i,
                        variable_names_to_plot[i],
                        va="center",
                        ha="left",
                        size=font_sizes["teensie"] - 4,
                        alpha=0.8,
                    )
                if relative:
                    ax.axvline(1, linestyle=":", color="grey")
                    ax.text(
                        1,
                        len(variable_names_to_plot) / 2,
                        "Original Score = %0.3f" % original_score_mean,
                        va="center",
                        ha="left",
                        size=font_sizes["teensie"],
                        rotation=270,
                        alpha=0.7
                    )
                    ax.set_xlabel("Percent of Original Score")
                    ax.set_xlim([0, 1.2])
                else:
                    ax.axvline(original_score_mean, linestyle=":", color="grey")
                    ax.text(
                        original_score_mean,
                        len(variable_names_to_plot) / 2,
                        "Original Score",
                        va="center",
                        ha="left",
                        size=font_sizes["teensie"] - 2,
                        rotation=270,
                        alpha=0.7
                    )

                ax.tick_params(axis="both", which="both", length=0)
                ax.set_yticks([])
                ax.set_xticks(xticks)

                upper_limit = min(1.05 * np.amax(scores_to_plot), 1.0)
                ax.set_xlim([0, upper_limit])

                # make the horizontal plot go with the highest value at the top
                ax.invert_yaxis()
                vals = ax.get_xticks()
                for tick in vals:
                    ax.axvline(
                        x=tick, linestyle="dashed", alpha=0.4, color="#eeeeee", zorder=1
                    )

                if k == 0:
                    pad = -0.15       
                    ax.annotate('higher ranking', xy=(0, 0), xytext=(pad, 0.5), xycoords = ax.transAxes, rotation=90,
                        size=6, ha='center', va='center', color='xkcd:vermillion', alpha=0.65)

                    ax.annotate(r'$\rightarrow$', xy=(0, 0), xytext=(pad, 0.75), xycoords = ax.transAxes, rotation=90,
                        size=12, ha='center', va='center', color='xkcd:vermillion', alpha=0.65)    
                    
                    ax.annotate('lower ranking', xy=(0, 0), xytext=(pad+0.05, 0.5), xycoords = ax.transAxes, rotation=90,
                            size=6, ha='center', va='center', color='xkcd:dodger blue', alpha=0.65)

                    ax.annotate(r'$\leftarrow$', xy=(0, 0), xytext=(pad+0.05, 0.24), xycoords = ax.transAxes, rotation=90,
                       size=12, ha='center', va='center', color='xkcd:dodger blue', alpha=0.65)
          
        self.set_major_axis_labels(
                fig,
                xlabel=metric,
                ylabel_left='',
                labelpad=5,
                fontsize=10,
            )
        self.set_row_labels(ylabels, axes)
        self.add_alphabet_label(axes)

            
    def save_figure(self, fname, fig=None, bbox_inches="tight", dpi=300, aformat="png"):
        """ Saves the current figure """
        plt.savefig(fname=fname, bbox_inches=bbox_inches, dpi=dpi, format=aformat)
        if fig is not None:
            plt.closefig(fig)

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
            return 'xkcd:pastel red'
        else:
            if VARIABLES_COLOR_DICT is None:
                return "xkcd:powder blue"
            else:
                return VARIABLES_COLOR_DICT[var]

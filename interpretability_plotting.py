import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

import waterfall_chart


# Set up the font sizes for matplotlib
FONT_SIZE = 16
BIG_FONT_SIZE = FONT_SIZE + 2
LARGE_FONT_SIZE = FONT_SIZE + 4
HUGE_FONT_SIZE = FONT_SIZE + 6
SMALL_FONT_SIZE = FONT_SIZE - 2
TINY_FONT_SIZE = FONT_SIZE - 4
TEENSIE_FONT_SIZE = FONT_SIZE - 8
font_sizes = {
    'teensie': TEENSIE_FONT_SIZE,
    'tiny': TINY_FONT_SIZE,
    'small': SMALL_FONT_SIZE,
    'normal': FONT_SIZE,
    'big': BIG_FONT_SIZE,
    'large': LARGE_FONT_SIZE,
    'huge': HUGE_FONT_SIZE,
}
plt.rc('font', size=FONT_SIZE)
plt.rc('axes', titlesize=FONT_SIZE)
plt.rc('axes', labelsize=FONT_SIZE)
plt.rc('xtick', labelsize=TINY_FONT_SIZE)
plt.rc('ytick', labelsize=TINY_FONT_SIZE)
plt.rc('legend', fontsize=FONT_SIZE)
plt.rc('figure', titlesize=BIG_FONT_SIZE)

line_colors = ['orangered', 'darkviolet', 'darkslategray', 'darkorange', 'darkgreen']

# class BasePlot():
# 
#     def __init__(n_panels, **kwargs):
# 
#         self._n_panels = n_panels
#         self._n_columns = kwargs.get('n_columns', 3)
#         self._figsize = kwargs.get('figsize', (6.4,4.8))
#         self._wspace = kwargs.get('wspace', 0.4)
#         self._hspace = kwargs.get('hspace', 0.3)
#         self._sharex = kwargs.get('sharex', False)
#         self._sharey = kwargs.get('sharey', False)
#         
#     def create_subplots(self):
# 
#         """
#         Create a series of subplots (MxN) based on the 
#         number of panels and number of columns (optionally)
#         """
#         
#         n_rows    = int(self._n_panels / self._n_columns)
#         extra_row = 0 if (self._n_panels % self._n_columns) == 0 else 1
# 
#         fig, axes = plt.subplots(self._n_rows+extra_row, self._n_columns, 
#                     sharex=self._sharex, sharey=self._sharey, figsize=self._figsize)
# 
#         plt.subplots_adjust(wspace = self._wspace, hspace = self._hspace)
# 
#         n_axes_to_delete = len(axes.flat) - self._n_panels
# 
#         if (n_axes_to_delete > 0):
#             for i in range(n_axes_to_delete):
#                 fig.delaxes(axes.flat[-(i+1)])
#     
#         return fig, axes
    
class InterpretabilityPlotting:

    def create_subplots(self, n_panels, **kwargs):

        """
        Create a series of subplots (MxN) based on the 
        number of panels and number of columns (optionally)
        """

        n_columns = kwargs.get('n_columns', 3)
        figsize   = kwargs.get('figsize', (6.4,4.8))
        wspace    = kwargs.get('wspace', 0.4)
        hspace    = kwargs.get('hspace', 0.3)
        sharex    = kwargs.get('sharex', False)
        sharey    = kwargs.get('sharey', False)
        
        n_rows    = int(n_panels / n_columns)
        extra_row = 0 if (n_panels % n_columns) ==0 else 1

        fig, axes = plt.subplots(n_rows+extra_row, n_columns, sharex=sharex, 
                                    sharey=sharey, figsize=figsize, dpi=300)
        plt.subplots_adjust(wspace = wspace, hspace = hspace)

        n_axes_to_delete = len(axes.flat) - n_panels

        if n_axes_to_delete > 0:
            for i in range(n_axes_to_delete):
                fig.delaxes(axes.flat[-(i+1)])
    
        return fig, axes

    def set_major_axis_labels(self, fig, xlabel=None, ylabel_left=None, 
                                ylabel_right=None, **kwargs):
        """
        Generate a single X- and Y-axis labels for 
        a series of subplot panels 
        """

        fontsize = kwargs.get('fontsize', 15)
        labelpad = kwargs.get('labelpad', 25)

        # add a big axis, hide frame
        ax = fig.add_subplot(111, frameon=False)

        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        # set axes labels
        ax.set_xlabel(xlabel,      fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel(ylabel_left, fontsize=fontsize, labelpad=labelpad)
        
        if ylabel_right is not None:
            ax_right = fig.add_subplot(1,1,1, sharex=ax, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

            ax_right.yaxis.set_label_position("right")
            ax_right.set_ylabel(ylabel_right, labelpad=labelpad, fontsize=fontsize)

    def add_histogram_axis(self, ax, data, **kwargs):

        color     = kwargs.get('color', 'lightblue')
        edgecolor = kwargs.get('color', 'white')

        cnt, bins, patches = ax.hist( data, bins='auto', alpha=0.3, color=color,
                                        density=True, edgecolor=edgecolor)
        
        area = np.dot(cnt, np.diff(bins))
        hist_ax = ax.twinx()
        hist_ax.grid('off')

        # align the twinx axis
        lb, ub = ax.get_ylim()
        hist_ax.set_ylim(lb / area, ub / area)

        return hist_ax

    def line_plot(self, ax, xdata, ydata, **kwargs):

        """
        Plots a curve of data
        """

        linewidth = kwargs.get('linewidth', 2.0)    
        linestyle = kwargs.get('linestyle', '-')
        color     = kwargs.get('color', 'blue')

        ax.plot(xdata, ydata, color=color, linewidth=linewidth, linestyle=linestyle)

    def confidence_interval_plot(self, ax, xdata, ydata, **kwargs):

        """
        Plot Confidence Intervals
        """

        facecolor = kwargs.get('facecolor', 'r')

        # get mean curve
        mean_ydata = np.mean(ydata, axis=0)

        # plot mean curve
        self.line_plot(ax, xdata, mean_ydata, **kwargs)

        # get confidence interval bounds
        lower_bound, upper_bound = np.percentile(ydata, [2.5, 97.5], axis=0)

        # fill between CI bounds
        ax.fill_between(xdata, lower_bound, upper_bound, facecolor=facecolor, alpha=0.4)


    def plot_1d_pd(self, feature_dict, **kwargs):

        """
        Generic function for 1-D PDP
        """

        hspace = kwargs.get('hspace', 0.5)
        ylim   = kwargs.get('ylim', [25,50])
        color  = kwargs.get('color', 'blue')

        # get the number of panels which will be length of feature dictionary
        n_panels = len(feature_dict.keys())

        # create subplots, one for each feature
        fig, axes = self.create_subplots(n_panels=n_panels, hspace=hspace, figsize=(8,6))

        # loop over each feature and add relevant plotting stuff
        for ax, feature in zip(axes.flat, feature_dict.keys()):

            for i, model in enumerate(feature_dict[feature].keys()):

                xdata     = feature_dict[feature][model]['xdata1']
                ydata     = feature_dict[feature][model]['pd_values']
                hist_data = feature_dict[feature][model]['hist_data']

                # add histogram
                hist_ax = self.add_histogram_axis(ax, np.clip(hist_data, xdata[0], xdata[-1]))
            
                # depending on number of bootstrap examples, do CI plot or just mean
                if (ydata.shape[0] > 1):
                    self.confidence_interval_plot(hist_ax, xdata, ydata, **kwargs)
                else:
                    self.line_plot(hist_ax, xdata, ydata[0,:], **kwargs)

                ax.set_xlabel(feature, fontsize=10)
                hist_ax.axhline(y=0.0, color="k", alpha=0.8)
                hist_ax.set_ylim([ydata.min(), ydata.max()])

        self.set_major_axis_labels(fig, xlabel=None, ylabel_left='Relative Frequency',
                                ylabel_right='Mean Probability (%)', **kwargs)

        plt.show()

        return fig, axes

    def plot_2d_pd(self, feature_dict, **kwargs):

        """
        Generic function for 2-D PDP
        """

        hspace = kwargs.get('hspace', 0.5)
        ylim   = kwargs.get('ylim', [25,50])
        cmap   = kwargs.get('cmap', 'bwr')
        levels = 20

        # get the number of panels which will be length of feature dictionary
        n_panels = len(feature_dict.keys())

        # create subplots, one for each feature
        fig, axes = self.create_subplots(n_panels=n_panels, hspace=hspace, figsize=(8,6))

        # loop over each feature and add relevant plotting stuff
        for ax, feature in zip(axes.flat, feature_dict.keys()):

            xdata1     = feature_dict[feature]['xdata1']
            xdata2     = feature_dict[feature]['xdata2']
            ydata      = feature_dict[feature]['pd_values']

            print(ydata[0,:,:])

            # can only do a contour plot with 2-d data
            x, y = np.meshgrid(xdata1, xdata2)
        
            cf = ax.contourf(x, y, ydata[0,:,:], cmap=cmap, levels=levels, alpha=0.75)
        
            fig.colorbar(cf, ax)
      
            ax.set_xlabel(feature[0], fontsize=10)
            ax.set_ylabel(feature[1], fontsize=10)

            ax.set_ylim(ylim)

        self.set_major_axis_labels(fig, xlabel=None, ylabel_left='Relative Frequency',
                                ylabel_right='Mean Probability (%)', **kwargs)

        plt.show()

        return fig, axes

    def plot_ale(self, feature_dict, **kwargs):

        """
        Generic function for 1st order ALE
        """

        hspace = kwargs.get('hspace', 0.5)
        ylim   = kwargs.get('ylim', [-15,15])
        color  = kwargs.get('color', 'blue')

        # get the number of panels which will be length of feature dictionary
        n_panels = len(feature_dict.keys())

        # create subplots, one for each feature
        fig, axes = self.create_subplots(n_panels=n_panels, hspace=hspace, figsize=(8,6))

        # loop over each feature and add relevant plotting stuff
        for ax, feature in zip(axes.flat, feature_dict.keys()):

            for i, model in enumerate(feature_dict[feature].keys()):

                xdata     = feature_dict[feature][model]['xdata1']
                xdata = 0.5 * (xdata[1:] + xdata[:-1])

                ydata     = feature_dict[feature][model]['ale_values']
                hist_data = feature_dict[feature][model]['hist_data']

                # add histogram
                hist_ax = self.add_histogram_axis(ax, np.clip(hist_data, xdata[0], xdata[-1]))
            
                # depending on number of bootstrap examples, do CI plot or just mean
                if (ydata.shape[0] > 1):
                    self.confidence_interval_plot(hist_ax, xdata, ydata, **kwargs)
                else:
                    self.line_plot(hist_ax, xdata, ydata[0,:], **kwargs)

                ax.set_xlabel(feature, fontsize=10)
                hist_ax.axhline(y=0.0, color="k", alpha=0.8)
                hist_ax.set_ylim(ylim)

        self.set_major_axis_labels(fig, xlabel=None, ylabel_left='Relative Frequency',
                                ylabel_right='Mean Probability (%)', **kwargs)

        plt.show()

        return fig, axes


    def ti_plot(self, dict_to_use, ax=None, n_vars=10, other_label='Other Predictors'):
        """
        Plot the tree interpreter.
        """
        contrib  = []
        varnames = []

        # return nothing if dictionary is empty
        if len(dict_to_use) == 0: return

        for var in list(dict_to_use.keys()):
            try:
                contrib.append(dict_to_use[var]["Mean Contribution"])
            except:
                contrib.append(dict_to_use[var])

            varnames.append(var)         
        """
        fig = waterfall_chart.plot(ax,
            varnames,
            contrib,
            rotation_value=90,
            sorted_value=True,
            threshold=0.02,
            net_label="Final prediction",
            other_label="Others",
            y_lab="Probability",
        )
        """
        bias_index = varnames.index('Bias')
        varnames.pop(bias_index)
        contrib.pop(bias_index)
        
        varnames=np.array(varnames)
        contrib=np.array(contrib)
        
        varnames = np.append(varnames[:n_vars], other_label)
        contrib = np.append(contrib[:n_vars],sum(contrib[n_vars:]))
        
        
        bar_colors = ['seagreen' if c > 0 else 'tomato' for c in contrib]
        y_index = range(len(contrib))
        ax.barh(y=y_index, 
                width=contrib,
                height=0.8,
                alpha=0.8,
                color = bar_colors,
               )
        ax.set_yticks(y_index)
        ax.set_yticklabels(varnames)
        
        pos_extra = 0.5
        neg_extra = 1.5
           
        if all(contrib>0):
            neg_extra=0
            
        elif all(contrib<0):
            pos_extra=0
            extra=0
        
        for i, c in enumerate(np.round(contrib, 1)):
            if c > 0:   
                ax.text(c + pos_extra, i + .25, str(c), 
                        color='k', 
                        fontweight='bold', 
                        alpha=0.8, fontsize=10)
            else:
                ax.text(c - neg_extra, i + .25, str(c), 
                        color='k', 
                        fontweight='bold', 
                        alpha=0.8, fontsize=10)
                
        ax.set_xlim([np.min(contrib)-neg_extra-0.75, np.max(contrib)+pos_extra+1.5])

        # make the horizontal plot go with the highest value at the top
        ax.invert_yaxis()
        
    def plot_treeinterpret(self, result_dict, **kwargs):
        '''
        Plot the results of tree interpret

        Args:
        ---------------
            result : pandas.Dataframe
                a single row/example from the 
                result dataframe from tree_interpreter_simple
        '''

        hspace = kwargs.get('hspace', 0.5)
        wspace = kwargs.get('wspace', 0.8)

        # get the number of panels which will be the number of ML models in dictionary
        n_panels = len(result_dict.keys())

        # loop over each model creating one panel per model
        for model_name in result_dict.keys():
    
            # try for all_data/average data
            if 'all_data' in result_dict[model_name].keys():

                fig = self.ti_plot(result_dict[model_name]['all_data'])
        
            # must be performanced based
            else:   

                # create subplots, one for each feature
                fig, sub_axes = self.create_subplots(n_panels=4, n_columns=2, 
                                                     hspace=hspace, 
                                                     wspace=wspace,
                                                     sharex=False, 
                                                     sharey=False, figsize=(8,6))

                for sax, perf_key in zip(sub_axes.flat, list(result_dict[model_name].keys())):
                    print(perf_key)
                    self.ti_plot(result_dict[model_name][perf_key], ax=sax)
                    sax.set_title(perf_key.upper().replace('_', ' '), fontsize=15)

        return fig

    def plot_variable_importance(self, importance_dict, multipass=True, ax=None, filename=None, 
                        readable_feature_names={}, feature_colors=None, metric = "Validation AUPRC",
                             relative=False, num_vars_to_plot=None, diagnostics=0, title='', **kwargs):

        """Plots any variable importance method for a particular estimator
        :param importance_dict: Dictionary of ImportanceResult objects returned by PermutationImportance
        :param filename: string to place the file into (including directory and '.png')
        :param multipass: whether to plot multipass or singlepass results. Default to True
        :param relative: whether to plot the absolute value of the results or the results relative to the original. Defaults
            to plotting the absolute results
        :param num_vars_to_plot: number of top variables to actually plot (cause otherwise it won't fit)
        :param diagnostics: 0 for no printouts, 1 for all printouts, 2 for some printouts. defaults to 0
        """

        hspace = kwargs.get('hspace', 0.5)

        # get the number of panels which will be the number of ML models in dictionary
        n_panels = len(importance_dict.keys())

        # create subplots, one for each feature
        fig, axes = self.create_subplots(n_panels=n_panels, hspace=hspace, figsize=(8,6))

        # loop over each model creating one panel per model
        for model_name, ax in zip(importance_dict.keys(), axes.flat):

            importance_obj = importance_dict[model_name]

            rankings = importance_obj.retrieve_multipass(
                ) if multipass else importance_obj.retrieve_singlepass()
        
            if num_vars_to_plot is None and multipass:
                num_vars_to_plot == len(list(rankings.keys()))

            original_score = importance_obj.original_score

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

            # Sort by increasing rank
            sorted_var_names = list(rankings.keys())
            sorted_var_names.sort(key=lambda k: rankings[k][0])
            sorted_var_names = sorted_var_names[:min(num_vars_to_plot, len(rankings))]
            scores = [rankings[var][1] for var in sorted_var_names]

            colors_to_plot = [self.variable_to_color(var, feature_colors) for var in [
                "Original Score", ] + sorted_var_names]
            variable_names_to_plot = [" {}".format(
                var) for var in self.convert_vars_to_readable(["Original Score", ] + sorted_var_names, readable_feature_names)]

            if bootstrapped:
                if relative:
                    scores_to_plot = np.array([original_score_mean, ] + [np.mean(score)
                                                                     for score in scores]) / original_score_mean
                else:
                    scores_to_plot = np.array(
                        [original_score_mean, ] + [np.mean(score) for score in scores])
                ci = np.array([np.abs(np.mean(score) - np.percentile(score, [2.5, 97.5]))
                           for score in np.r_[[original_score, ], scores]]).transpose()
            else:
                if relative:
                    scores_to_plot = np.array(
                        [original_score_mean, ] + scores) / original_score_mean
                else:
                    scores_to_plot = np.array(
                        [original_score_mean, ] + scores)
                ci = np.array([[0, 0]
                           for score in np.r_[[original_score, ], scores]]).transpose()

            method = "%s Permutation Importance" % (
                "Multipass" if multipass else "Singlepass")

            # Actually make plot
#             if ax is None:
#                 fig, ax = plt.subplots(figsize=(8, 6))

            if bootstrapped:
                ax.barh(np.arange(len(scores_to_plot)),
                     scores_to_plot, linewidth=1, edgecolor='black', color=colors_to_plot, 
                     xerr=ci, capsize=4, ecolor='grey', error_kw=dict(alpha=0.4))
            else:
                ax.barh(np.arange(len(scores_to_plot)),
                     scores_to_plot, linewidth=1, edgecolor='black', color=colors_to_plot)

            # Put the variable names _into_ the plot
            for i in range(len(variable_names_to_plot)):
                ax.text(0, i, variable_names_to_plot[i],
                     va="center", ha="left", size=font_sizes['teensie'])
            if relative:
                ax.axvline(1, linestyle=':', color='grey')
                ax.text(1, len(variable_names_to_plot) / 2, "original score = %0.3f" % original_score_mean,
                     va='center', ha='left', size=font_sizes['teensie'], rotation=270)
                ax.set_xlabel("Percent of Original Score")
                ax.set_xlim([0, 1.2])
            else:
                ax.axvline(original_score_mean, linestyle=':', color='grey')
                ax.text(original_score_mean, len(variable_names_to_plot) / 2, "original score",
                     va='center', ha='left', size=font_sizes['teensie'], rotation=270)

            ax.set_yticks([])
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

            # make the horizontal plot go with the highest value at the top
            ax.invert_yaxis()

    def save_figure(self, fig, fname, bbox_inches="tight", dpi=300, aformat="png"):
        """ Saves the current figure """
        return plt.savefig(fname, bbox_inches=bbox_inches, dpi=dpi, format=aformat)


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
        '''
        Returns the color for each variable.
        '''
        if var == 'Original Score':
            return 'lightcoral'
        else:
            if VARIABLES_COLOR_DICT is None:
                return 'lightgreen'
            else:
                return VARIABLES_COLOR_DICT[var]

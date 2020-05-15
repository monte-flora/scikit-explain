import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import waterfall_chart
from model_clarify import ModelClarify
from matplotlib.ticker import FormatStrFormatter

# Set up the font sizes for matplotlib
FONT_SIZE = 14
BIG_FONT_SIZE = FONT_SIZE + 2
LARGE_FONT_SIZE = FONT_SIZE + 4
HUGE_FONT_SIZE = FONT_SIZE + 6
SMALL_FONT_SIZE = FONT_SIZE - 2
TINY_FONT_SIZE = FONT_SIZE - 4
TEENSIE_FONT_SIZE = FONT_SIZE - 6

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
plt.rc('xtick', labelsize=TEENSIE_FONT_SIZE)
plt.rc('ytick', labelsize=TEENSIE_FONT_SIZE)
plt.rc('legend', fontsize=FONT_SIZE)
plt.rc('figure', titlesize=BIG_FONT_SIZE)
plt.rcParams["font.family"] = "serif"

line_colors = ['orangered', 'darkviolet', 'darkslategray', 'darkorange', 'darkgreen']

class ClarifierPlot(ModelClarify):

    """
    ClarifierPlot is a python class that uses the calculcations 
    from ModelClarify to make publication-quality figures 
    from a variety of ML interpretation techniques.
    """

    def __init__(self, models, examples, targets=None, feature_names=None):
        super().__init__(model=models, examples=examples, targets=targets, feature_names=feature_names)
        
<<<<<<< HEAD
    def plot_ale(self, features, subsample=1.0, nbootstrap=1):
        compute_func    = self.calc_ale
        self.subsample  = subsample
        self.nbootstrap = nbootstrap
=======
    def plot_ale(self, features, subsample=1.0, nbootstrap=1, to_readable_name=None, **kwargs):
        """
        Plot accumulate local effect from one or more features.

        Args: 
        --------------------------
            features : str or list of strs
                One or more features to compute ALE for
            subsample : float
            nbootstrap : int
            to_readable_name : callable
            kwargs : dict 
                keyword arguments for plotting 
        """
        compute_func = self.calc_ale
        self.subsample = subsample
        self.nbootstrap =nbootstrap
        self.to_readable_name = to_readable_name
>>>>>>> master
        ylim = [-7.5, 7.5]
        fig, axes = self.plot_interpret_curve(features, compute_func, ylim, **kwargs)
        return fig, axes

    def plot_pdp(self, features, subsample=1.0, nbootstrap=1, to_readable_name=None, **kwargs):
        compute_func = self.calc_pdp
        self.subsample = subsample
        self.nbootstrap =nbootstrap
        self.to_readable_name = to_readable_name
        ylim = [0, 100.]
        fig, axes = self.plot_interpret_curve(features, compute_func, ylim, **kwargs)
        return fig, axes

    def _create_base_subplots(self, n_panels, **kwargs):
        """
        Create a series of subplots (MxN) based on the 
        number of panels and number of columns (optionally)
        """
        if n_panels <= 4:
            n_columns = 2
            wspace = 0.35
        else:
            n_columns = kwargs.get('n_columns', 3)
            wspace = kwargs.get('wspace', 0.4)
        
        figsize = kwargs.get('figsize', (6.4,4.8))
        hspace = kwargs.get('hspace', 0.3)
        sharex = kwargs.get('sharex', False)
        sharey = kwargs.get('sharey', False)
        
        n_rows = int(n_panels / n_columns)
        extra_row = 0 if (n_panels % n_columns) ==0 else 1

        fig, axes = plt.subplots(n_rows+extra_row, n_columns, sharex=sharex, sharey=sharey, figsize=figsize, dpi=300)
        plt.subplots_adjust(wspace = wspace, hspace = hspace)

        n_axes_to_delete = len(axes.flat) - n_panels

        if n_axes_to_delete > 0:
            for i in range(n_axes_to_delete):
                fig.delaxes(axes.flat[-(i+1)])
    
        return fig, axes

    def _create_panel_labels(self, axes, **kwargs):
        """
        Labels panels by alphabet
        """
        fontsize = kwargs.get('fontsize', 12)
        alphabet_list = [chr(x) for x in range(ord('a'), ord('z') + 1)] 
        n_panels = len(axes.flat)
        for letter, panel in zip(alphabet_list[:n_panels], axes.flat):
            panel.text(0.075, 0.075,f'({letter})', ha='center', va='center', 
                   transform=panel.transAxes, fontsize=fontsize)
    
    def _major_axis_labels(self, fig, xlabel=None, ylabel_left=None, ylabel_right=None, **kwargs):
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
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel(ylabel_left, fontsize=fontsize, labelpad=labelpad)
        
        extrapad=20
        if ylabel_right is not None:
            axR = fig.add_subplot(1,1,1, sharex=ax, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            #axR.yaxis.tick_right()
            axR.yaxis.set_label_position("right")
            axR.set_ylabel(ylabel_right, labelpad=labelpad+extrapad, fontsize=fontsize)


    def line_plot(self, ax, xdata, ydata, **kwargs):
        """
        """
        linewidth = kwargs.get('linewidth', 2.5)    
        linestyle = kwargs.get('linestyle', '-')
        color = kwargs.get('color', 'r')
        label = kwargs.get('label', None)

        ax.plot(xdata,ydata, 
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
                label=label
                )

    def _ax_hist(self, ax, x, **kwargs):
        cnt, bins, patches = ax.hist(
                        x,
                        bins='auto',
                        alpha=0.3,
                        color='lightblue',
                        density=True,
                        edgecolor="white",
                        )
        
        # area under the istogram
        area = np.dot(cnt, np.diff(bins))
        twin_ax = ax.twinx()
        twin_ax.grid('off')

        # align the twinx axis
        #twin_ax.set_yticks(ax.get_yticks() / area)
        lb, ub = ax.get_ylim()
        twin_ax.set_ylim(lb / area, ub / area)

        return twin_ax
        
    def _ci_plot(self, ax, xdata, ydata, **kwargs):
        """
        Plot Confidence Intervals
        """
        facecolor = kwargs.get('facecolor', 'r')
        linecolor = kwargs.get('linecolor', 'r')
        label = kwargs.get('model_name', None)

        mean_ydata = np.mean(ydata,axis=0)
        self.line_plot(ax, xdata, mean_ydata, color=color, label=label)
        uncertainty = np.percentile(ydata, [2.5, 97.5], axis=0)
        ax.fill_between(xdata, uncertainty[0], uncertainty[1], facecolor=facecolor, alpha=0.4)


    def _get_xlabel_fontsize(self, n_panels):
        """
        Return an appropriate X-label fontsize for the
        ALE and PDP plot panels. 
        """
        if n_panels <= 4:
            xlabel_fontsize = SMALL_FONT_SIZE
        else:
            xlabel_fontsize = TINY_FONT_SIZE

        return xlabel_fontsize

    def _get_xlabel(self, feature_name):
        """
        Return the X-label of the ALE and PDP panel plots.
        """
        if self.to_readable_name is None:
            return feature_name
        else:
            return to_readable_name[feature_name]

    def plot_interpret_curve(self, features, compute_func, ylim, **kwargs):
        """
        Generic function for ALE & PDP
        """
        if not isinstance(features, list):
            features=[features]

        hspace = kwargs.get('hspace', 0.45)
        wspace = kwargs.get('wspace', 0.4)

        fig,axes = fig, axes = self._create_base_subplots(n_panels=len(features), 
                                                          hspace=hspace,
                                                          wspace=wspace,
                                                          figsize=(8,6))
        
        xlabel_fontsize = self._get_xlabel_fontsize(len(features)) 
        for ax, feature in zip(axes.flat , features):
            for i, model_name in enumerate(list(self.model_set.keys())):
                feature_examples = self._examples[feature]
                ydata, xdata = compute_func(model=self.model_set[model_name], feature=feature, subsample=self.subsample, nbootstrap=self.nbootstrap)
                twin_ax = self._ax_hist(ax, np.clip(feature_examples, xdata[0], xdata[-1]))
                if 'ale' in compute_func.__name__:
                    xdata = 0.5 * (xdata[1:] + xdata[:-1])
                if np.array(ydata).ndim == 2:
                    self._ci_plot(twin_ax, xdata, ydata, color=line_colors[i], facecolor=line_colors[i], label=model_name)
                else:
                    self.line_plot(twin_ax, xdata, ydata, color=line_colors[i], label=model_name)
                ax.set_xlabel(self._get_xlabel(feature), fontsize=xlabel_fontsize)
                twin_ax.axhline(y=0.0, color="k", alpha=0.8)
                twin_ax.set_ylim(ylim)

        if 'ale' in compute_func.__name__:
            label = 'Accumulated Local Effect (%)'
        else: 
            label = 'Mean Probability (%)'

        self._major_axis_labels(fig, xlabel=None, ylabel_left='Relative Frequency', ylabel_right=label, **kwargs)
        self._create_panel_labels(axes)

        return fig, axes

    def plot_second_order_relationship(ale_data, quantile_tuple, feature_names, ax=None, **kwargs):

        """
		    Plots the second order ALE

		    ale_data: 2d numpy array of data
		    quantile_tuple: tuple of the quantiles/ranges
		    feature_names: tuple of feature names which should be strings
        """
        if ax is None:
            fig, ax = plt.subplots()

        # get quantiles/ranges for both features
        x = quantile_tuple[0]
        y = quantile_tuple[1]

        X, Y = np.meshgrid(x, y)

        CF = ax.pcolormesh(X, Y, ale_data, cmap="bwr", alpha=0.7)
        plt.colorbar(CF)

        ax.set_xlabel(f"Feature: {feature_names[0]}")
        ax.set_ylabel(f"Feature: {feature_names[1]}")

    def get_highest_predictions(self, result,num):
        """
        Return "num" highest predictions from a treeinterpreter result
        """
        highest_pred = result.sum(axis=1).values
        idx = np.argsort(highest_pred)[-num:]

        example = result.iloc[idx,:]

        return example

    def combine_like_features(self, contrib, varnames):
        """
        Combine the contributions of like features. E.g., 
        multiple statistics of a single variable
        """
        duplicate_vars = {}
        for var in varnames:
            duplicate_vars[var] = [idx for idx, v in enumerate(varnames) if v == var]

        new_contrib = []
        new_varnames = []
        for var in list(duplicate_vars.keys()):
            idxs = duplicate_vars[var]
            new_varnames.append(var)
            new_contrib.append(np.array(contrib)[idxs].sum())

        return new_contrib, new_varnames

    def plot_treeinterpret(self, result, ax=None, to_only_varname=None):
        '''
        Plot the results of tree interpret

        Args:
        ---------------
            result : pandas.Dataframe
                a single row/example from the 
                result dataframe from tree_interpreter_simple
            to_only_varname : callable
                A function that would convert predictors to 
                just their variable name. For example,
                if using multiple statistcs (max, mean, min, etc)
                of a single variable, to_only_varname, should convert
                the name of those predictors to just the name of the 
                single variable. This allows the results to combine 
                contributions from the different statistics of a
                single variable into a single variable. 
        '''
        if ax is None:
            fig, ax = plt.subplots()
        contrib=[]
        varnames=[]
        for i, var in enumerate(list(result.keys())):
            try:
                contrib.append(result[var]["Mean Contribution"])
            except:
                contrib.append(result[var])
            if to_only_varname is None:
                varnames.append(var)
            else:
                varnames.append(to_only_varname(var))

        if to_only_varname is not None:
            contrib, varnames = combine_like_features(contrib, varnames)

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

        return fig

    def plot_performance_based_contributions(self, to_only_varname=None, n_examples=10):
        """
        Performance
        """
        fig, axes = self._create_base_subplots(
                                               n_panels=4, 
                                               n_columns=2, 
                                               figsize=(12,8),
                                               wspace =0.2,
                                               sharex=False,
                                               sharey=False
                                               )
        result = self.get_top_contributors(n_examples=n_examples)
        for ax, key in zip(axes.flat, list(result.keys())):
            self.plot_treeinterpret(ax=ax, result=result[key], to_only_varname=None)
            ax.set_title(key.upper().replace('_', ' '), fontsize=15)


    def plot_variable_importance(self, importance_obj, multipass=True, ax=None, filename=None, 
                        readable_feature_names={}, feature_colors=None, metric = "Validation AUPRC",
                             relative=False, num_vars_to_plot=None, diagnostics=0, title=''):
        """Plots any variable importance method for a particular estimator
        :param importance_obj: ImportanceResult object returned by PermutationImportance
        :param filename: string to place the file into (including directory and '.png')
        :param multipass: whether to plot multipass or singlepass results. Default to True
        :param relative: whether to plot the absolute value of the results or the results relative to the original. Defaults
            to plotting the absolute results
        :param num_vars_to_plot: number of top variables to actually plot (cause otherwise it won't fit)
        :param diagnostics: 0 for no printouts, 1 for all printouts, 2 for some printouts. defaults to 0
        """
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

        colors_to_plot = [variable_to_color(var, feature_colors) for var in [
            "Original Score", ] + sorted_var_names]
        variable_names_to_plot = [" {}".format(
            var) for var in convert_vars_to_readable(["Original Score", ] + sorted_var_names, readable_feature_names)]

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
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        if bootstrapped:
            ax.barh(np.arange(len(scores_to_plot)),
                 scores_to_plot, linewidth=1, edgecolor='black', color=colors_to_plot, xerr=ci, capsize=4, ecolor='grey', error_kw=dict(alpha=0.4))
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
def convert_vars_to_readable(variables_list, VARIABLE_NAMES_DICT):
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
def variable_to_color(var, VARIABLES_COLOR_DICT):
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




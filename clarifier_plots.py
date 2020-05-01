import numpy as np
import matplotlib.pyplot as plt
import waterfall_chart
from model_clarify import ModelClarify

class ClarifierPlot(ModelClarify):
    """
    ClarifierPlot is a python class that uses the calculcations 
    from ModelClarify to make publication-quality figures 
    from a variety of ML interpretation techniques.
    """
    def __init__(self, models, examples, targets=None, feature_names=None):
        super().__init__(model=models, examples=examples, targets=targets, feature_names=feature_names)
        
    def plot_ale(self, features):
        compute_func = self.calc_ale
        ylim = [-7.5, 7.5]
        fig, axes = self.plot_interpret_curve(features, compute_func, ylim)
        return fig, axes

    def plot_pdp(self, features):
        compute_func = self.calc_pdp
        ylim = [0, 100.]
        fig, axes = self.plot_interpret_curve(features, compute_func, ylim)
        return fig, axes

    def _create_base_subplots(self, n_panels, **kwargs):
        """
        Create a series of subplots (MxN) based on the 
        number of panels and number of columns (optionally)
        """
        n_columns = kwargs.get('n_columns', 3)
        figsize = kwargs.get('figsize', (6.4,4.8))
        wspace = kwargs.get('wspace', 0.4)
        hspace = kwargs.get('hspace', 0.3)
        sharex = kwargs.get('sharex', False)
        sharey = kwargs.get('sharey', False)
        
        n_rows = int(n_panels / n_columns)
        extra_row = 0 if (n_panels % n_columns) ==0 else 1

        fig, axes = plt.subplots(n_rows+extra_row, n_columns, sharex=sharex, sharey=sharey, figsize=figsize)
        plt.subplots_adjust(wspace = wspace, hspace = hspace)

        n_axes_to_delete = len(axes.flat) - n_panels

        if n_axes_to_delete > 0:
            for i in range(n_axes_to_delete):
                fig.delaxes(axes.flat[-(i+1)])
    
        return fig, axes

    def _create_panel_labels(axes, **kwargs):
        """
        Labels panels by alphabet
        """
        fontsize = kwargs.get('fontsize', 12)
        alphabet_list = [chr(x) for x in range(ord('a'), ord('z') + 1)] 
        n_panels = len(axes.flat)
        for letter, panel in zip(alphabet_list[:n_panels], axes.flat):
            panel.text(0.125, 0.9,f'({letter})', ha='center', va='center', 
                   transform=panel.transAxes, fontsize=fontsize)
    
    def _major_axis_labels(fig, xlabel=None, ylabel=None,**kwargs):
        """
        Generate a single X- and Y-axis labels for 
        a series of subplot panels 
        """
        fontsize = kwargs.get('fontsize', 12)
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
    
    def line_plot(self, ax, xdata, ydata, **kwargs):
        """
        """
        linewidth = kwargs.get('linewidth', 2.0)    
        linestyle = kwargs.get('linestyle', '-')
        ax.plot(xdata,ydata, 
                linewidth=linewidth,
                linestyle=linestyle
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
        

    def _ci_plot(ax, xdata, ydata, **kwargs):
        """
        Plot Confidence Intervals
        """
        mean_ydata = np.mean(ydata)
        line_plot(self, ax, xdata, mean_ydata, **kwargs)
        uncertainty = np.percentile(ydata, [2.5, 97.5], axis=0)
        ax.fill_between(xdata, uncertainty[0], uncertainty[1], facecolor='r', alpha=0.4)

    def plot_interpret_curve(self,features, compute_func, ylim):
        """
        Generic function for ALE & PDP
        """
        if not isinstance(features, list):
            features=[features]

        fig,axes = fig, axes = self._create_base_subplots(n_panels=len(features), hspace=0.5, figsize=(8,6))
        for ax, feature in zip(axes.flat,features):
            feature_examples = self._examples[feature]
            ydata, xdata = compute_func(feature)
            #twin_ax = ax.twinx()
            twin_ax = self._ax_hist(ax, np.clip(feature_examples, xdata[0], xdata[-1]))
             
            centered_xdata = 0.5 * (xdata[1:] + xdata[:-1])
            if ydata.ndim == 2:
                _ci_plot(twin_ax, centered_xdata, ydata)
            else:
                self.line_plot(twin_ax, centered_xdata, ydata)
            ax.set_xlabel(feature, fontsize=10)
            twin_ax.axhline(y=0.0, color="k", alpha=0.8)
            twin_ax.set_ylim(ylim)

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

    def plot_treeinterpret(self, result, ax=None, save_name=None, to_only_varname=None):
        '''
        Plot the results of tree interpret

        Args:
        ---------------
            result : pandas.Dataframe
                a single row/example from the 
                result dataframe from tree_interpreter_simple
            save_name : str
                file path & name to save the figure
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
                contrib.append(result[var].values[0])
            if to_only_varname is None:
                varnames.append(var)
            else:
                varnames.append(to_only_varname(var))

        if to_only_varname is not None:
            contrib, varnames = combine_like_features(contrib, varnames)

        plt = waterfall_chart.plot(ax,
            varnames,
            contrib,
            rotation_value=90,
            sorted_value=True,
            threshold=0.02,
            net_label="Final prediction",
            other_label="Others",
            y_lab="Probability",
        )
        if save_name is not None:
            plt.savefig(save_name, bbox_inches="tight", dpi=300)

    def plot_performance_based_contributions(self, save_name=None, to_only_varname=None, n_examples=10):
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
            self.plot_treeinterpret(ax=ax, result=result[key], save_name=None, to_only_varname=None)




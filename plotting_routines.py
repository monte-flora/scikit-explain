import numpy as np
import matplotlib.pyplot as plt
import waterfall_chart

def _ax_title(ax, title, subtitle):
    """
        Prints title on figure.

        Parameters
        ----------
        fig : matplotlib.axes.Axes
                Axes objet where to print titles.
        title : string
                Main title of figure.
        subtitle : string
                Sub-title for figure.
        """
    ax.set_title(title + "\n" + subtitle)
    # fig.suptitle(subtitle, fontsize=10, color="#919191")


def _ax_labels(ax, xlabel, ylabel):
    """
        Prints labels on axis' plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
                Axes object where to print labels.
        xlabel : string
                Label of X axis.
        ylabel : string
                Label of Y axis.
        """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _ax_quantiles(ax, quantiles, twin="x"):
    """
        Plot quantiles of a feature over opposite axis.

        Parameters
        ----------
        ax : matplotlib.Axis
                Axis to work with.
        quantiles : array-like
                Quantiles to plot.
        twin : string
                Possible values are 'x' or 'y', depending on which axis to plot quantiles.
        """
    print(("Quantiles :", quantiles))
    if twin == "x":
        ax_top = ax.twiny()
        ax_top.set_xticks(quantiles)
        ax_top.set_xticklabels(
            [
                "{1:0.{0}f}%".format(
                    int(i / (len(quantiles) - 1) * 100 % 1 > 0),
                    i / (len(quantiles) - 1) * 100,
                )
                for i in range(len(quantiles))
            ],
            color="#545454",
            fontsize=7,
        )
        ax_top.set_xlim(ax.get_xlim())
    elif twin == "y":
        ax_right = ax.twinx()
        ax_right.set_yticks(quantiles)
        ax_right.set_yticklabels(
            [
                "{1:0.{0}f}%".format(
                    int(i / (len(quantiles) - 1) * 100 % 1 > 0),
                    i / (len(quantiles) - 1) * 100,
                )
                for i in range(len(quantiles))
            ],
            color="#545454",
            fontsize=7,
        )
        ax_right.set_ylim(ax.get_ylim())


def _ax_scatter(ax, points):
    print(points)
    ax.scatter(points.values[:, 0], points.values[:, 1], alpha=0.5, edgecolor=None)


def _ax_grid(ax, status):
    ax.grid(status, linestyle="-", alpha=0.4)


def _ax_hist(ax, x, **kwargs):
    ax.hist(
        x,
        bins='auto',
        alpha=0.3,
        color='lightblue',
        density=True,
        edgecolor="white",
    )
    #ax.set_ylabel("Relative Frequency", fontsize=15)


def _line_plot(ax, x, y, **kwargs):
    ax.plot(x, y, "ro--", linewidth=2, markersize=12, mec="black", alpha=0.7)


def _ci_plot(ax, x, y_bottom, y_top, **kwargs):
    """
    Plot Confidence Intervals
    """
    ax.fill_between(x, y_bottom, y_top, facecolor='r', alpha=0.4)


def plot_first_order_ale(quantiles, 
        feature_name, feature_examples=None, ax=None, **kwargs):
    """
		Plots the first order ALE

		ale_data: 1d numpy array of data
		quantiles: range of values your data takes on
		feature_name: name of the feature of type string
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax_plt = ax.twinx()

    _ax_labels(ax_plt, "Feature '{}'".format(feature_name), "")
    _ax_grid(ax_plt, True)
    if feature_examples is not None:
        _ax_hist(ax, np.clip(feature_examples, quantiles[0], quantiles[-1]), **kwargs)
    centered_quantiles = 0.5 * (quantiles[1:] + quantiles[:-1])
    #if len(ale_data.shape) > 1:
    #    mean_ale = np.mean(ale_data, axis=0)
    #    _line_plot(ax_plt, centered_quantiles, mean_ale, **kwargs)
    #
        # Plot error bars
    #    y_95 = np.percentile(ale_data, 97.5, axis=0)
    #    y_5 = np.percentile(ale_data, 2.5, axis=0)
    #    _ci_plot(ax=ax_plt, x=centered_quantiles, y_bottom=y_5, y_top=y_95)
    #else:
    #    _line_plot(ax_plt, centered_quantiles, ale_data, **kwargs)

    #ax_plt.set_ylabel("Accum. Local Effect (%)", fontsize=15)
    ax.set_xlabel(feature_name, fontsize=10)
    ax_plt.axhline(y=0.0, color="k", alpha=0.8)
    ax_plt.set_ylim([-7.5, 7.5])

    return ax_plt, centered_quantiles

def plot_second_order_ale(ale_data, quantile_tuple, feature_names, ax=None, **kwargs):

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

    # ALE_interp = scipy.interpolate.interp2d(quantiles[0], quantiles[1], ALE)

    CF = ax.pcolormesh(X, Y, ale_data, cmap="bwr", alpha=0.7)
    plt.colorbar(CF)

    ax.set_xlabel(f"Feature: {feature_names[0]}")
    ax.set_ylabel(f"Feature: {feature_names[1]}")


def plot_categorical_ale(ale_data, feature_values, feature_name, **kwargs):

    """
		Plots ALE for a categorical variable

		ale_data: 1d numpy array of data
		feature_values: tuple of the quantiles/ranges
		feature_name: name of the feature of type string
	"""

    fig, ax = plt.subplots()

    ax.boxplot(feature_values, ale_data, "ko--")
    ax.set_xlabel(f"Feature: {feature_name}")
    ax.set_ylabel("Accumulated Local Effect")

    plt.show()


def plot_monte_carlo_ale(ale_data, quantiles, feature_name, **kwargs):

    """
		ale_data: 2d numpy array of data [n_monte_carlo, n_quantiles]
		quantile_tuple: numpy array of quantiles (typically 10-90 percentile values)
		feature_name: string representing the feature name
	"""

    fig, ax = plt.subplots()

    # get number of monte_carlo sims
    n_simulations = ale_data.shape[0]

    # get mean
    mean_ale = np.mean(ale_data, axis=0)

    # plot individual monte_sim
    for i in range(n_simulations):
        ax.plot(quantiles, ale_data[i, :], color="#1f77b4", alpha=0.06)

    # plot mean last
    ax.plot(quantiles, mean_ale, "ro--", linewidth=2, markersize=12, mec="black")

    ax.set_xlabel(f"Feature: {feature_name}")
    ax.set_ylabel("Accumulated Local Effect")

    plt.show()


def plot_pdp_1d(pdp_data, quantiles, feature_name, feature_examples, ax=None, **kwargs):

    """
                Plots the first order ALE

                ale_data: 1d numpy array of data
                quantiles: range of values your data takes on
                feature_name: name of the feature of type string
        """
    if ax is None:
        fig, ax = plt.subplots()

    ax_plt = ax.twinx()

    _ax_labels(ax_plt, "Feature '{}'".format(feature_name), "")
    _ax_grid(ax_plt, True)
    _ax_hist(
        ax,
        np.clip(feature_examples, quantiles[0], quantiles[-1]),
        **kwargs,
    )
    _line_plot(ax_plt, quantiles, pdp_data * 100.0, color="black", **kwargs)
    ax_plt.set_ylabel("Mean Probability (%)", fontsize=15)
    ax.set_xlabel(feature_name, fontsize=15)
    #ax_plt.axhline(y=0.0, color="k", alpha=0.8)
    ax_plt.set_ylim([0,100])
    
def plot_2d_partial_dependence(pdp_data, feature_names, variable_ranges, **kwargs):

    """
		Plots 2D partial dependence plot

		feature_names: tuple of two features for plotting
		variable_ranges: tuple of two ranges for plotting

	"""

    fig, ax = plt.subplots()

    X, Y = np.meshgrid(variable_ranges[0], variable_ranges[1])
    CF = ax.pcolormesh(X, Y, pdp_data, cmap="rainbow", alpha=0.7)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])

    plt.colorbar(CF)
    plt.show()

def get_highest_predictions(result,num):
    """
    Return "num" highest predictions from a treeinterpreter result
    
    """
    highest_pred = result.sum(axis=1).values
    idx = np.argsort(highest_pred)[-num:]

    example = result.iloc[idx,:]

    return example

def combine_like_features(contrib, varnames):
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

def plot_treeinterpret(result, save_name, to_only_varname=None):
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
    contrib=[]
    varnames=[]
    for i, var in enumerate(list(result.keys())):
        try:
            contrib.append(result[var]["Mean Contribution"])
        except:
            contrib.append(result[var].values)
        if to_only_varname is None:
            varnames.append(var)
        else:
            varnames.append(to_only_varname(var))

    print(contrib, varnames)
    if to_only_varname is not None:
        contrib, varnames = combine_like_features(contrib, varnames)

    plt = waterfall_chart.plot(
        varnames,
        contrib,
        rotation_value=90,
        sorted_value=True,
        threshold=0.02,
        net_label="Final prediction",
        other_label="Others",
        y_lab="Probability",
    )
    plt.savefig(save_name, bbox_inches="tight", dpi=300)


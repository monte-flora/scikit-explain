import numpy as np
import matplotlib.pyplot as plt

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
        #fig.suptitle(subtitle, fontsize=10, color="#919191")

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


def _ax_quantiles(ax, quantiles, twin='x'):
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
        if twin == 'x':
                ax_top = ax.twiny()
                ax_top.set_xticks(quantiles)
                ax_top.set_xticklabels(["{1:0.{0}f}%".format(int(i / (len(quantiles) - 1) * 100 % 1 > 0), i / (len(quantiles) - 1) * 100) for i in range(len(quantiles))], color="#545454", fontsize=7)
                ax_top.set_xlim(ax.get_xlim())
        elif twin =='y':
                ax_right = ax.twinx()
                ax_right.set_yticks(quantiles)
                ax_right.set_yticklabels(["{1:0.{0}f}%".format(int(i / (len(quantiles) - 1) * 100 % 1 > 0), i / (len(quantiles) - 1) * 100) for i in range(len(quantiles))], color="#545454", fontsize=7)
                ax_right.set_ylim(ax.get_ylim())

def _ax_scatter(ax, points):
        print(points)
        ax.scatter(points.values[:,0], points.values[:,1], alpha=0.5, edgecolor=None)

def _ax_grid(ax, status):
        ax.grid(status, linestyle='-', alpha=0.4)

def _ax_hist(ax, x, **kwargs):
        ax.hist( x, bins=10, alpha = 0.5, color = kwargs['facecolor'], density=True, edgecolor ='white')
        ax.set_ylabel('Relative Frequency', fontsize=15)

def _line_plot(ax, x, y, **kwargs):
    ax.plot(
        x,
        y,
        "ro--",
        linewidth=2,
        markersize=12,
        mec="black",
        alpha = 0.7
    )

def plot_first_order_ale(ale_data, quantiles, feature_name, examples, ax=None, **kwargs):

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
    _ax_hist(ax, np.clip(examples[feature_name].values, quantiles[0], quantiles[-1]), **kwargs)
    centered_quantiles = 0.5*(quantiles[1:] + quantiles[:-1])
    _line_plot(ax_plt, centered_quantiles, ale_data, color="black", **kwargs)
    ax_plt.set_ylabel('Accum. Local Effect (%)', fontsize=15)
    ax.set_xlabel(feature_name, fontsize=15) 
    ax_plt.axhline(y=0.0, color='k', alpha=0.8)

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

    CF = ax.pcolormesh(X, Y, ale_data, cmap="bwr",alpha=0.7)
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


def plot_1d_partial_dependence(pdp_data, feature_name, variable_range, **kwargs):

    """
		Plots 1D partial dependence plot.

		feature_name: name of the feature you are plotting (string)
		variable_range: range of values your data takes on

	"""

    fig, ax = plt.subplots()

    # Plot the mean PDP
    ax.plot(
        variable_range,
        pdp_data * 100.0,
        "ro--",
        linewidth=2,
        markersize=12,
        mec="black",
    )

    ax.set_xlabel(feature_name, fontsize=15)
    ax.set_ylabel("Mean Probabilitiy (%)", fontsize=12)

    plt.show()


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

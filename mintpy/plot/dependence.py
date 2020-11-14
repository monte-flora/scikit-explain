from __future__ import division

import numpy as np
import warnings
try:
    import matplotlib.pyplot as pl
    import matplotlib
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass

from shap.plots import colors
from shap.common import convert_name, approximate_interactions
from matplotlib.ticker import MaxNLocator

from .base_plotting import PlotStructure
base_plot = PlotStructure()

labels = {
    'MAIN_EFFECT': "SHAP main effect value for\n%s",
    'INTERACTION_VALUE': "SHAP interaction value",
    'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
    'VALUE': "SHAP value (impact on model output)",
    'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
    'VALUE_FOR': "SHAP value for\n%s",
    'PLOT_FOR': "SHAP plot for %s",
    'FEATURE': "Feature %s",
    'FEATURE_VALUE': "Feature value",
    'FEATURE_VALUE_LOW': "Low",
    'FEATURE_VALUE_HIGH': "High",
    'JOINT_VALUE': "Joint SHAP value",
    'MODEL_OUTPUT': "Model output value"
}


def dependence_plot(ind, shap_values, features, feature_names=None, feature_values=None, display_features=None,
                    interaction_index="auto", target_values=None, 
                    color="#1E88E5", axis_color="#333333", cmap=None,
                    dot_size=5, x_jitter=0, alpha=1, title=None, xmin=None, xmax=None, ax=None, fig=None, **kwargs):
    """ Create a SHAP dependence plot, colored by an interaction feature.

    Plots the value of the feature on the x-axis and the SHAP value of the same feature
    on the y-axis. This shows how the model depends on the given feature, and is like a
    richer extenstion of the classical parital dependence plots. Vertical dispersion of the
    data points represents interaction effects. Grey ticks along the y-axis are data
    points where the feature's value was NaN.


    Parameters
    ----------
    ind : int or string
        If this is an int it is the index of the feature to plot. If this is a string it is
        either the name of the feature to plot, or it can have the form "rank(int)" to specify
        the feature with that rank (ordered by mean absolute SHAP value over all the samples).

    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features).

    features : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features).

    feature_names : list
        Names of the features (length # features).

    display_features : numpy.array or pandas.DataFrame
        Matrix of feature values for visual display (such as strings instead of coded values).

    interaction_index : "auto", None, int, or string
        The index of the feature used to color the plot. The name of a feature can also be passed
        as a string. If "auto" then shap.common.approximate_interactions is used to pick what
        seems to be the strongest interaction (note that to find to true stongest interaction you
        need to compute the SHAP interaction values).

    x_jitter : float (0 - 1)
        Adds random jitter to feature values. May increase plot readability when feature
        is discrete.

    alpha : float
        The transparency of the data points (between 0 and 1). This can be useful to the
        show density of the data points when using a large dataset.

    xmin : float or string
        Represents the lower bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.

    xmax : float or string
        Represents the upper bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.

    ax : matplotlib Axes object
         Optionally specify an existing matplotlib Axes object, into which the plot will be placed.
         In this case we do not create a Figure, otherwise we do.

    """
    unnormalize = kwargs.get('unnormalize', None)
    cmap = colors.red_blue
    
    original_feature_names = list(features.columns)
    if feature_values is None:
        original_feature_values = features.values
    else:
        original_feature_values = feature_values

    if unnormalize is not None:
        feature_values = unnormalize._full_inverse_transform(original_feature_values)

    # allow vectors to be passed
    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, len(shap_values), 1)
    if len(features.shape) == 1:
        features = np.reshape(features, len(features), 1)

    ind = convert_name(ind, shap_values, original_feature_names)

    # guess what other feature as the stongest interaction with the plotted feature
    if not hasattr(ind, "__len__"):
        if interaction_index == "auto":
            interaction_index = approximate_interactions(ind, shap_values, original_feature_values)[0]
        interaction_index = convert_name(interaction_index, shap_values, original_feature_names)

    assert shap_values.shape[0] == features.shape[0], \
        "'shap_values' and 'features' values must have the same number of rows!"
    assert shap_values.shape[1] == features.shape[1], \
        "'shap_values' must have the same number of columns as 'features'!"

    # get both the raw and display feature values
    oinds = np.arange(shap_values.shape[0]) # we randomize the ordering so plotting overlaps are not related to data ordering
    np.random.shuffle(oinds)
    
    xdata = feature_values[oinds, ind].astype(np.float64)
    s = shap_values[oinds, ind]
    if target_values is not None:
        target_values = target_values[oinds] 

    # get both the raw and display color values
    if interaction_index is not None:
        cdata = feature_values[:, interaction_index]
        clow = np.nanpercentile(cdata.astype(np.float), 5)
        chigh = np.nanpercentile(cdata.astype(np.float), 95)
        if clow == chigh:
            clow = np.nanmin(cdata.astype(np.float))
            chigh = np.nanmax(cdata.astype(np.float))

    # optionally add jitter to feature values
    if x_jitter > 0:
        if x_jitter > 1: x_jitter = 1
        xvals = xdata.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(np.float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals) # returns a sorted array
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(xvals))
            jitter_amount = x_jitter * smallest_diff
            xdata += (np.random.ranf(size = len(xdata))*jitter_amount) - (jitter_amount/2)

    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    xdata_nan = np.isnan(xdata)
    xdata_notnan = np.invert(xdata_nan)
    if interaction_index is not None:
        cdata_imp = cdata.copy()
        cdata_imp[np.isnan(cdata)] = (clow + chigh) / 2.0
        cdata[cdata_imp > chigh] = chigh
        cdata[cdata_imp < clow] = clow
        p = ax.scatter(
            xdata[xdata_notnan], s[xdata_notnan], s=dot_size, linewidth=0, c=cdata[xdata_notnan],
            cmap=cmap, alpha=alpha, vmin=clow, vmax=chigh,
            rasterized=len(xdata) > 500
        )
        p.set_array(cdata[xdata_notnan])
    elif target_values is not None:
        p = ax.scatter(
            xdata[xdata_notnan], s[xdata_notnan], s=dot_size, linewidth=0, c=target_values[xdata_notnan],
            cmap=cmap, alpha=alpha, vmin=np.min(target_values), vmax=np.max(target_values),
            rasterized=len(xdata) > 500
        )
    else:
        p = ax.scatter(xdata, s, s=dot_size, linewidth=0, color=color,
                       alpha=alpha, rasterized=len(xdata) > 500)

    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        cb = pl.colorbar(p, ticks=MaxNLocator(5), ax=ax)
        cb.set_label(display_features[interaction_index], size=8)
        cb.ax.tick_params(labelsize=8)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.7) * 20)
        base_plot._to_sci_notation(ax=None,colorbar=cb, ydata=cdata)

    # plot any nan feature values as tick marks along the y-axis
    xlim = ax.get_xlim()
    if interaction_index is not None:
        p = ax.scatter(
            xlim[0] * np.ones(xdata_nan.sum()), s[xdata_nan], marker=1,
            linewidth=2, c=cdata_imp[xdata_nan], cmap=cmap, alpha=alpha,
            vmin=clow, vmax=chigh
        )
        p.set_array(cdata[xdata_nan])
    else:
        ax.scatter(
            xlim[0] * np.ones(xdata_nan.sum()), s[xdata_nan], marker=1,
            linewidth=2, color=color, alpha=alpha
        )

    xmin = np.nanmin(xdata)
    xmax = np.nanmax(xdata)
    ax.set_xlim([xmin,xmax]) 

    # make the plot more readable
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)

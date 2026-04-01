import matplotlib.pyplot as plt
import pandas as pd


def sobol_plot(
    results, est_name=None, ax=None, display_feature_names=None, n_features=None,
    kind="bar", fontsize=None,
):
    """Plot Sobol sensitivity indices (1st order and higher order) as a stacked bar or barh chart.

    Parameters
    ----------
    results : xarray.Dataset
        Results from ExplainToolkit.sobol (contains sobol_total_rankings, sobol_1st_scores,
        and sobol_interact_scores variables).
    est_name : str, optional
        Estimator name suffix used in the dataset variable names. If None, inferred
        from the first data variable.
    ax : matplotlib Axes, optional
        Pre-existing axes for the plot. If None, a new figure and axes are created.
    display_feature_names : dict, optional
        Maps internal feature names to readable display names.
    n_features : int, optional
        Number of top features to plot. If None, all features are plotted.
    kind : {'bar', 'barh'}, default 'bar'
        Type of bar chart. 'bar' for vertical, 'barh' for horizontal.

    Returns
    -------
    ax : matplotlib Axes
    """
    if display_feature_names is None:
        display_feature_names = {}

    if ax is None:
        f, ax = plt.subplots(dpi=300, figsize=(6, 4))

    if est_name is None:
        est_name = list(results.data_vars)[0].split("__")[1]

    display_feature_names_list = [
        display_feature_names.get(f, f) for f in results[f"sobol_total_rankings__{est_name}"].values
    ]

    if n_features is None:
        n_features = len(display_feature_names_list)

    df_result = pd.DataFrame(
        {
            "variable": display_feature_names_list[:n_features],
            "1st Order": results[f"sobol_1st_scores__{est_name}"].values[:n_features, 0],
            "Higher Order": results[f"sobol_interact_scores__{est_name}"].values[:n_features, 0],
        }
    )

    if kind == "bar":
        rot = 90
    else:
        rot = 0

    ax = df_result.plot(ax=ax, x="variable", kind=kind, stacked=True, rot=rot)

    # Scale tick and label font sizes
    if fontsize is None:
        fontsize = 11
    tick_fontsize = max(8, fontsize - 2)

    if kind == "bar":
        ax.set_xlabel("")
        ax.set_ylabel("Total Sobol Index\n(1st order + higher order)", fontsize=fontsize)
    else:
        ax.set_ylabel("")
        ax.set_xlabel("Total Sobol Index\n(1st order + higher order)", fontsize=fontsize)
        ax.invert_yaxis()

    ax.tick_params(axis="both", labelsize=tick_fontsize)

    return ax

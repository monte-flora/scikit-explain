import matplotlib.pyplot as plt
import pandas as pd


def sobol_plot(
    results, est_name=None, ax=None, display_feature_names={}, n_features=None, kind="bar"
):
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

    if kind == "bar":
        ax.set_xlabel("")
        ax.set_ylabel("Total Sobol Index\n(1st order + higher order)")
    else:
        ax.set_ylabel("")
        ax.set_xlabel("Total Sobol Index\n(1st order + higher order)")
        ax.invert_yaxis()

    return ax

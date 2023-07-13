# =========================================================
# Libraries for the feature contribution computations.
# =========================================================
import pandas as pd
import numpy as np
import xarray as xr 
from sklearn.preprocessing import MinMaxScaler

from .utils import to_xarray

def group_local_values(explain_ds, groups, X, inds=None):
    """
    Using a dictionary of feature groups, compute the grouped 
    SHAP values. 
    """
    if inds is None:
        inds = np.arange(len(X))
    
    estimator_name = explain_ds.attrs['estimators used']
    method = explain_ds.attrs['method']
    
    explain_df = pd.DataFrame(explain_ds[f'{method}_values__{estimator_name}'], 
                           columns=explain_ds.attrs['features'])

    names=[]
    vals =[]

    for name, features in groups.items():
        names.append(name)
        these_vals = explain_df[features].values[inds,:]
        sum_vals = np.sum(these_vals, axis=1)
        vals.append(sum_vals)

    dataset={}
    dataset[f"{method}_values__{estimator_name}"] = (
                    ["n_examples", "n_features"],
                    np.array(vals).T,
                )
    #dataset[f"{method}_bias__{estimator_name}"] = (
    #                ["n_examples"],
    #                bias.astype(np.float64),
    #            )

    dataset["X"] = (["n_examples", "n_features"], X.values)
    
    ds = to_xarray(dataset)
    
    ds.attrs['features'] = groups.keys()
    ds.attrs['method'] = method
    
    return ds


def group_feature_values(X, groups, inds=None, func=np.mean):
    if inds is None:
        inds = np.arange(len(X))
    
    # Perform min-max scaling 
    X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X.values[inds]), columns=X.columns)

    return pd.DataFrame(
        np.array([func(X_scaled[groups[feature]], axis=1) for feature in groups.keys()]).T,
        columns = groups.keys()) 
    

def get_indices_based_on_performance(estimator, X, y, estimator_output, n_samples=None, class_idx=1):
    """
     Determines the best hits, worst false alarms, worst misses, and best
     correct negatives using the data provided during initialization.

     Args:
     ------------------
          estimator : The estimator to process
          n_samples: number of "best/worst" X to return. If None,
              the routine uses the whole dataset

    Return:
          a dictionary containing the indices of each of the 4 categories
          listed above
    """
    if len(y) < 1:
        raise ValueError(
            "y is empty! User likely did not initialize ExplainToolkit with 'y'"
        )

    # default is to use all X
    if n_samples is None:
        n_samples = X.shape[0]

    # make sure user didn't goof the input
    if n_samples <= 0:
        print("n_samples less than or equals 0. Defaulting back to all")
        n_samples = X.shape[0]

    if estimator_output == "probability":
        predictions = estimator.predict_proba(X)[:, class_idx]
    elif estimator_output == "raw":
        predictions = estimator.predict(X)

    diff = y - predictions
    data = {"y": y, "predictions": predictions, "diff": diff}
    df = pd.DataFrame(data)

    if estimator_output == "probability":
        nonevent_X = df[y != class_idx]
        event_X = df[y == class_idx]

        event_X_sorted_indices = event_X.sort_values(
            by="diff", ascending=True
        ).index.values

        nonevent_X_sorted_indices = nonevent_X.sort_values(
            by="diff", ascending=False
        ).index.values

        best_hit_indices = event_X_sorted_indices[:n_samples].astype(int)
        worst_miss_indices = event_X_sorted_indices[-n_samples:][::-1].astype(int)

        best_corr_neg_indices = nonevent_X_sorted_indices[:n_samples].astype(int)
        worst_false_alarm_indices = nonevent_X_sorted_indices[-n_samples:][::-1].astype(
            int
        )

        sorted_dict = {
            "Best Hits": best_hit_indices,
            "Worst Misses": worst_miss_indices,
            "Worst False Alarms": worst_false_alarm_indices,
            "Best Corr. Negatives": best_corr_neg_indices,
        }

    else:
        X_sorted_indices = df.sort_values(by="diff", ascending=True).index.values

        least_error_indices = X_sorted_indices[:n_samples].astype(int)
        most_error_indices = X_sorted_indices[-n_samples:].astype(int)

        sorted_dict = {
            "Least Error Predictions": least_error_indices,
            "Most Error Predictions": most_error_indices,
        }

    return sorted_dict


def avg_and_sort_contributions(contrib_dict, feature_val_dict):
    """
    Get the mean value (of data for a predictory) and contribution from
    each predictor and sort"

    Args:
    -----------
        the_dict: dictionary to process
        performance_dict: if using performance based apporach, this should be
            the dictionary with corresponding indices

    Return:

        a dictionary of mean values and contributions
    """
    avg_contrib_dict = {}
    avg_feature_val_dict = {}

    # for hits, misses, etc.
    for key in contrib_dict.keys():
        contrib_df = contrib_dict[key]
        feature_val_df = feature_val_dict[key]

        contrib_series = contrib_df.mean(axis=0)
        feature_val_series = feature_val_df.mean(axis=0)
        feature_val_series["Bias"] = 0.0

        indices = contrib_series.abs().sort_values(ascending=False).index

        sorted_contrib_df = contrib_series.reindex(indices)
        sorted_feature_val_df = feature_val_series.reindex(indices)

        top_contrib = {
            var: contrib_series[var] for var in list(sorted_contrib_df.index)
        }
        top_values = {
            var: feature_val_series[var] for var in list(sorted_feature_val_df.index)
        }

        avg_contrib_dict[key] = top_contrib
        avg_feature_val_dict[key] = top_values

    return avg_contrib_dict, avg_feature_val_dict


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

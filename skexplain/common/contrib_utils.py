# =========================================================
# Libraries for the feature contribution computations.
# =========================================================
import pandas as pd


def get_indices_based_on_performance(estimator, X, y, estimator_output, n_samples=None):
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
        predictions = estimator.predict_proba(X)[:, 1]
    elif estimator_output == "raw":
        predictions = estimator.predict(X)

    diff = y - predictions
    data = {"y": y, "predictions": predictions, "diff": diff}
    df = pd.DataFrame(data)

    if estimator_output == "probability":
        nonevent_X = df[y == 0]
        event_X = df[y == 1]

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

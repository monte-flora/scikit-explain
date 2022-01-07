import numpy as np
import xarray as xr
import pickle
import pandas as pd
from collections import ChainMap
from sklearn.metrics import brier_score_loss, average_precision_score, precision_recall_curve
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import t
from functools import partial
from sklearn.metrics._base import _average_binary_score
from sklearn.utils.multiclass import type_of_target


def method_average_ranking(data, features, methods, estimator_names, n_features=12):
    """
    Compute the median ranking across the results of different ranking methods. 
    Also, include the 25-75th percentile ranking uncertainty.
    
    Parameters
    ------------
        data : list of xarray.Dataset
            The set of predictor ranking results to average over. 
        
        methods : list of string 
            The ranking methods to use from the data (see plot_importance for examples)
            
        estimator_names : string or list of strings
            Name of the estimator(s).
    
    Returns
    --------
        rankings_dict_avg : dict 
            feature : median ranking pairs 
        rankings_sorted : np.array 
            Sorted median rankings (lower values indicates higher rank)
        feature_sorted : np.array 
            The features corresponding the ranks in ``rankings_sorted``
        xerr
    
    """
    rankings_dict = {f : [] for f in features}
    for d, method in zip(data, methods):
        for estimator_name in estimator_names:
            features = d[f'{method}_rankings__{estimator_name}'].values[:n_features]
            rankings = {f:i for i, f in enumerate(features)}
            for f in features:
                try:
                    rankings_dict[f].append(rankings[f])
                except:
                    rankings_dict[f].append(np.nan)
    
    max_len = np.max([len(rankings_dict[k]) for k in rankings_dict.keys()])
    for k in rankings_dict.keys():
        l = rankings_dict[k]
        if len(l) < max_len:
            delta = max_len - len(l)
            rankings_dict[k] = l + [np.nan]*delta
        
    rankings_dict_avg = {f : np.nanpercentile(rankings_dict[f], 50) for f in rankings_dict.keys()}
    
    features = np.array(list(rankings_dict_avg.keys()))
    rankings = np.array([rankings_dict_avg[f] for f in features ])
    idxs = np.argsort(rankings)

    rankings_sorted = rankings[idxs]
    features_ranked = features[idxs]
    
    scores = np.array([rankings_dict[f] for f in features_ranked])

    data={}
    data[f"combined_rankings__{estimator_name}"] = (
                    [f"n_vars_avg"],
                    features_ranked,
                )
    data[f"combined_scores__{estimator_name}"] = (
                    [f"n_vars_avg", "n_bootstrap"],
                    scores,
    )
    data = xr.Dataset(data)
    
    return data

def to_pymint_importance(importances, estimator_name, feature_names, method):
    """Convert coefficients into a importance dataset from plotting purposes"""
  
    bootstrap=False
    if method == 'sage':
        importances_std = importances.std
        importances = importances.values
    elif method == 'coefs':
        importances = np.absolute(importances)
    elif method == 'shap_std':
        # Compute the std(SHAP) 
        importances = np.std(importances, axis=0)
    elif method == 'shap_sum':
        #Compute sum of abs values
        importances = np.sum(np.absolute(importances), axis=0)
    else:
        if np.ndim(importances) == 2:
            # average over bootstrapping
            bootstrap=True
            importances_to_save = importances.copy()
            importances = np.mean(importances, axis=1)
           
    # Sort from higher score to lower score 
    ranked_indices = np.argsort(importances)[::-1]
    
    if bootstrap:
        scores_ranked = importances_to_save[ranked_indices, :]
    else:
        scores_ranked = importances[ranked_indices]
    
    if method == 'sage':
        std_ranked = importances_std[ranked_indices]
    
    features_ranked = np.array(feature_names)[ranked_indices]

    data={}
    data[f"{method}_rankings__{estimator_name}"] = (
                    [f"n_vars_{method}"],
                    features_ranked,
                )
    
    if not bootstrap:
        scores_ranked = scores_ranked.reshape(len(scores_ranked),1)
        importances = importances.reshape(len(importances), 1)
        
        
    data[f"{method}_scores__{estimator_name}"] = (
                    [f"n_vars_{method}", "n_bootstrap"],
                    scores_ranked,
        )
        
    if method == 'sage':
        data[f"sage_scores_std__{estimator_name}"] = (
                    [f"n_vars_sage"],
                    std_ranked,
        )
    
    data = xr.Dataset(data)

    data.attrs['estimators used'] = estimator_name
    data.attrs['estimator output'] = 'probability'
    
    return data

def flatten_nested_list(list_of_lists):
    """Turn a list of list into a single, flatten list"""
    all_elements_are_lists = all([is_list(item) for item in list_of_lists])
    if not all_elements_are_lists:
        new_list_of_lists=[]
        for item in list_of_lists:
            if is_list(item):
                new_list_of_lists.append(item)
            else:
                new_list_of_lists.append([item])
        list_of_lists = new_list_of_lists
    
    
    return [ item for elem in list_of_lists for item in elem]

def is_dataset(data):
    return isinstance(data, xr.Dataset)

def is_dataframe(data):
    return isinstance(data, pd.DataFrame)

def check_is_permuted(X, X_permuted):
    permuted_features = []
    for f in X.columns:
        if not np.array_equal(X.loc[:,f], X_permuted.loc[:,f]):
            permuted_features.append(f)
            
    return permuted_features
        
def is_correlated(corr_matrix, feature_pairs, rho_threshold=0.8):
    """
    Returns dict where the key are the feature pairs and the items
    are booleans of whether the pair is linearly correlated above the
    given threshold.
    """
    results = {}
    for pair in feature_pairs:
        f1, f2 = pair.split("__")
        corr = corr_matrix[f1][f2]
        results[pair] = round(corr, 3) >= rho_threshold
    return results


def is_fitted(estimator):
    """
    Checks if a scikit-learn estimator/transformer has already been fit.


    Parameters
    ----------
    estimator: scikit-learn estimator (e.g. RandomForestClassifier)
        or transformer (e.g. MinMaxScaler) object


    Returns
    -------
    Boolean that indicates if ``estimator`` has already been fit (True) or not (False).
    """

    attrs = [v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")]

    return len(attrs) != 0


def determine_feature_dtype(X, features):
    """
    Determine if any features are categorical.
    """
    feature_names = list(X.columns)
    non_cat_features = []
    cat_features = []
    for f in features:
        if f not in feature_names:
            raise KeyError(f"'{f}' is not a valid feature.")

        if str(X.dtypes[f]) == "category":
            cat_features.append(f)
        else:
            non_cat_features.append(f)

    return non_cat_features, cat_features


def brier_skill_score(y_values, forecast_probabilities):
    """Computes the brier skill score"""
    climo = np.mean((y_values - np.mean(y_values)) ** 2)
    return 1.0 - brier_score_loss(y_values, forecast_probabilities) / climo


def modified_precision(precision, known_skew, new_skew): 
    """
    Modify the success ratio according to equation (3) from 
    Lampert and Gancarski (2014). 
    """
    precision[precision<1e-5] = 1e-5
    
    term1 = new_skew / (1.0-new_skew)
    term2 = ((1/precision) - 1.0)
    
    denom = known_skew + ((1-known_skew)*term1*term2)
    
    return known_skew / denom 
    
def calc_sr_min(skew):
    pod = np.linspace(0,1,100)
    sr_min = (skew*pod) / (1-skew+(skew*pod))
    return sr_min 

def _binary_uninterpolated_average_precision(
            y_true, y_score, known_skew, new_skew, pos_label=1, sample_weight=None):
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        if known_skew is not None:
            precision = modified_precision(precision, known_skew, new_skew)
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

def min_aupdc(y_true, pos_label, average, sample_weight=None, known_skew=None, new_skew=None):
    """
    Compute the minimum possible area under the performance 
    diagram curve. Essentially, a vote of NO for all predictions. 
    """
    min_score = np.zeros((len(y_true)))
    average_precision = partial(_binary_uninterpolated_average_precision,
                                known_skew=known_skew,
                                new_skew=new_skew,
                                pos_label=pos_label)
    ap_min = _average_binary_score(average_precision, y_true, min_score,
                                 average, sample_weight=sample_weight)

    return ap_min
        
def norm_aupdc(y_true, y_score, known_skew=None, *, average="macro", pos_label=1,
                            sample_weight=None, min_method='random'):
    """
    Compute the normalized modified average precision. Normalization removes 
    the no-skill region either based on skew or random classifier performance. 
    Modification alters success ratio to be consistent with a known skew. 
  
    Parameters:
    -------------------
        y_true, array of (n_samples,)
            Binary, truth labels (0,1)
        y_score, array of (n_samples,)
            Model predictions (either determinstic or probabilistic)
        known_skew, float between 0 and 1 
            Known or reference skew (# of 1 / n_samples) for 
            computing the modified success ratio.
        min_method, 'skew' or 'random'
            If 'skew', then the normalization is based on the minimum AUPDC 
            formula presented in Boyd et al. (2012).
            
            If 'random', then the normalization is based on the 
            minimum AUPDC for a random classifier, which is equal 
            to the known skew. 
    
    
    Boyd, 2012: Unachievable Region in Precision-Recall Space and Its Effect on Empirical Evaluation, ArXiv
    """
    new_skew = np.mean(y_true)
    if known_skew is None:
        known_skew=new_skew
    
    y_type = type_of_target(y_true)
    if y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError("Parameter pos_label is fixed to 1 for "
                         "multilabel-indicator y_true. Do not set "
                         "pos_label or set pos_label to 1.")
    elif y_type == "binary":
        # Convert to Python primitive type to avoid NumPy type / Python str
        # comparison. See https://github.com/numpy/numpy/issues/6784
        present_labels = np.unique(y_true).tolist()
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError(
                f"pos_label={pos_label} is not a valid label. It should be "
                f"one of {present_labels}"
            )
    average_precision = partial(_binary_uninterpolated_average_precision,
                                known_skew=known_skew,
                                new_skew=new_skew,
                                pos_label=pos_label)
    
    ap = _average_binary_score(average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)
    
    if min_method == 'random':
        ap_min = known_skew 
    elif min_method == 'skew':
        ap_min = min_aupdc(y_true, 
                       pos_label, 
                       average,
                       sample_weight=sample_weight,
                       known_skew=known_skew, 
                       new_skew=new_skew)
    
    naupdc = (ap - ap_min) / (1.0 - ap_min)

    return naupdc

def cartesian(array, out=None):
    """Generate a cartesian product of input array.
    Parameters

    Codes comes directly from sklearn/utils/extmath.py
    ----------
    array : list of array-like
        1-D array to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(array)) containing cartesian products
        formed of input array.
    X
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    array = [np.asarray(x) for x in array]
    shape = (len(x) for x in array)
    dtype = array[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(array), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(array):
        out[:, n] = array[n][ix[:, n]]

    return out


def to_dataframe(results, estimator_names, feature_names):
    """
    Convert the feature contribution results to a pandas.DataFrame
    with nested indexing. 
    """
    # results[0] = dict of avg. contributions per estimator 
    # results[1] = dict of avg. feature values per estimator
    feature_names+=['Bias']
    
    nested_key = results[0][estimator_names[0]].keys()
    
    dframes = []
    for key in nested_key:
        data=[]
        for name in estimator_names:
            contribs_dict = results[0][name][key]
            vals_dict = results[1][name][key]
            data.append([contribs_dict[f] for f in feature_names] + [vals_dict[f] for f in feature_names]) 
        column_names = [f+'_contrib' for f in feature_names] + [f+'_val' for f in feature_names]
        df = pd.DataFrame(data, columns=column_names, index=estimator_names)
        dframes.append(df)
    
    result = pd.concat(dframes, keys=list(nested_key))
    
    return result

def to_xarray(data):
    """Converts data dict to xarray.Dataset"""
    ds = xr.Dataset(data)
    return ds


def is_str(a):
    """Check if argument is a string"""
    return isinstance(a, str)


def is_list(a):
    """Check if argument is a list"""
    return isinstance(a, list)


def to_list(a):
    """Convert argument to a list"""
    return [a]


def is_valid_feature(features, official_feature_list):
    """Check if a feature is valid"""
    for f in features:
        if isinstance(f, tuple):
            for sub_f in f:
                if sub_f not in official_feature_list:
                    raise Exception(f"Feature {sub_f} is not a valid feature!")
        else:
            if f not in official_feature_list:
                raise Exception(f"Feature {f} is not a valid feature!")


def is_classifier(estimator):
    """Return True if the given estimator is (probably) a classifier.
    Parameters

    Function from base.py in sklearn
    ----------
    estimator : object
        Estimator object to test.
    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "classifier"


def is_regressor(estimator):
    """Return True if the given estimator is (probably) a regressor.
    Parameters

    Functions from base.py in sklearn
    ----------
    estimator : object
        Estimator object to test.
    Returns
    -------
    out : bool
        True if estimator is a regressor and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "regressor"


def is_all_dict(alist):
    """ Check if every element of a list are dicts """
    return all([isinstance(l, dict) for l in alist])


def load_pickle(fnames):
    """
    Load data from a list of pickle files as dict
    where the key are provided by the user
    """
    if not isinstance(fnames, list):
        fnames = [fnames]

    data = []
    for f in fnames:
        with open(f, "rb") as pkl_file:
            data.append(pickle.load(pkl_file))

    if is_all_dict(data):
        return merge_dict(data)
    else:
        return data


def save_pickle(fname, data):
    """Save data to a pickle file."""
    with open(fname, "wb") as pkl_file:
        pickle.dump(data, pkl_file)


def load_netcdf(fnames):
    """Load multiple netcdf files with xarray"""
    if not is_list(fnames):
        fnames = [fnames]

    data = []
    for f in fnames:
        ds = xr.open_dataset(f)
        data.append(ds)

    try:
        ds_set = xr.merge(data, combine_attrs="no_conflicts", compat="override")
    except:
        estimators_used = [ds.attrs["estimators used"] for ds in data]
        ds_set = xr.merge(data, combine_attrs="override", compat="override")
        ds_set.attrs["estimators used"] = flatten_nested_list(estimators_used)

    # Check that names
    # estimator_names = ds_set.attrs['estimators used']
    # if len(list(set(alist))) != len(alist):
    #        alist = [x+f'_{i}' for i,x in enumerate(alist)]

    return ds_set

def load_dataframe(fnames):
    """Load multiple dataframes with pandas"""
    if not is_list(fnames):
        fnames=[fnames]

    data = [pd.read_pickle(file_name) for file_name in fnames]
   
    attrs = [d.attrs for d in data]
    estimators_used = [d.attrs['estimators used'] for d in data]
    
    attrs = dict(ChainMap(*attrs))
    
    # Merge dataframes
    data_concat = pd.concat(data)
    
    for key in attrs.keys():
        data_concat.attrs[key] = attrs[key]
    
    data_concat.attrs['estimators used'] = flatten_nested_list(estimators_used)
    
    return data_concat
        

def save_netcdf(fname, ds, complevel=5):
    """Save netcdf file with xarray"""
    comp = dict(zlib=True, complevel=complevel)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(path=fname, encoding=encoding)
    ds.close()
    del ds

    
def save_dataframe(fname, dframe, ):
    """Save dataframe as pickle file"""
    dframe.to_pickle(fname) 
    del dframe
    
def combine_top_features(results_dict, n_vars=None):
    """Combines the list of top features from different estimators
    into a single list where duplicates are removed.

    Args:
    -------------
        results_dict : dict
        n_vars : integer
    """
    if n_vars is None:
        n_vars = 1000
    combined_features = []
    for estimator_name in results_dict.keys():
        features = results_dict[estimator_name]
        combined_features.append(features)
    unique_features = list(set.intersection(*map(set, combined_features)))[:n_vars]

    return unique_features

def compute_bootstrap_indices(X, subsample=1.0, n_bootstrap=1, seed=90):
    """
    Routine to generate the indices for bootstrapped X.

    Args:
    ----------------
        X : pandas.DataFrame, numpy.array
        subsample : float or integer
        n_bootstrap : integer

    Return:
    ----------------
        bootstrap_indices : list
            list of indices of the size of subsample or subsample*len(X)
    """
    base_random_state = np.random.RandomState(seed=seed)
    random_num_set = base_random_state.choice(10000, size=n_bootstrap, replace=False)
    random_states = [np.random.RandomState(s) for s in random_num_set]
    
    n_samples = len(X)
    size = int(n_samples * subsample) if subsample <= 1.0 else subsample
    
    bootstrap_indices = [
        random_state.choice(range(n_samples), size=size).tolist()
        for random_state in random_states
    ]
    return bootstrap_indices


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


def merge_dict(dicts):
    """Merge a list of dicts into a single dict """
    return dict(ChainMap(*dicts))


def merge_nested_dict(dicts):
    """
    Merge a list of nested dicts into a single dict
    """
    merged_dict = {}
    for d in dicts:
        for key in d.keys():
            for subkey in d[key].keys():
                if key not in list(merged_dict.keys()):
                    merged_dict[key] = {subkey: {}}
                merged_dict[key][subkey] = d[key][subkey]

    return merged_dict


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def get_indices_based_on_performance(
    estimator, X, y, estimator_output, n_samples=None
):
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
        worst_miss_indices = event_X_sorted_indices[-n_samples:][::-1].astype(
            int
        )

        best_corr_neg_indices = nonevent_X_sorted_indices[:n_samples].astype(
            int
        )
        worst_false_alarm_indices = nonevent_X_sorted_indices[-n_samples:][
            ::-1
        ].astype(int)
        
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


def retrieve_important_vars(results, estimator_names, multipass=True):
    """
    Return a list of the important features stored in the
     ImportanceObject

     Args:
     -------------------
         results : python object
             ImportanceObject from PermutationImportance
         multipass : boolean
             if True, returns the multipass permutation importance results
             else returns the singlepass permutation importance results

     Returns:
         top_features : list
             a list of features with order determined by
             the permutation importance method
    """
    perm_method = "multipass" if multipass else "singlepass"

    important_vars_dict = {}
    for estimator_name in estimator_names:
        top_features = list(results[f"{perm_method}_rankings__{estimator_name}"].values)
        important_vars_dict[estimator_name] = top_features

    return important_vars_dict


def find_correlated_pairs_among_top_features(
    corr_matrix,
    top_features,
    rho_threshold=0.8,
):
    """
    Of the top features, find correlated pairs above some
    linear correlation coefficient threshold

    Args:
    ----------------------
        corr_matrix : pandas.DataFrame
        top_features : list of strings
        rho_threshold : float

    """
    top_feature_indices = {f: i for i, f in enumerate(top_features)}
    
    _top_features = [f for f in top_features if f != 'No Permutations']
    
    sub_corr_matrix = corr_matrix[_top_features].loc[_top_features]
    
    pairs = []
    for feature in _top_features:
        #try:
        most_corr_feature = (
                sub_corr_matrix[feature].sort_values(ascending=False).index[1]
            )
        #except:
        #    continue

        most_corr_value = sub_corr_matrix[feature].sort_values(ascending=False)[1]
        if round(most_corr_value, 5) >= rho_threshold:
            pairs.append((feature, most_corr_feature))

    pairs = list(set([tuple(sorted(t)) for t in pairs]))
    pair_indices = [
        (top_feature_indices[p[0]], top_feature_indices[p[1]]) for p in pairs
    ]

    return pairs, pair_indices


def cmds(D, k=2):
    """Classical multidimensional scaling

    Theory and code references:
    https://en.wikipedia.org/wiki/Multidimensional_scaling#Classical_multidimensional_scaling
    http://www.nervouscomputer.com/hfs/cmdscale-in-python/
    Arguments:
    D -- A squared matrix-like object (array, DataFrame, ....), usually a distance matrix
    """

    n = D.shape[0]
    if D.shape[0] != D.shape[1]:
        raise Exception("The matrix D should be squared")
    if k > (n - 1):
        raise Exception("k should be an integer <= D.shape[0] - 1")

    # (1) Set up the squared proximity matrix
    D_double = np.square(D)
    # (2) Apply double centering: using the centering matrix
    # centering matrix
    center_mat = np.eye(n) - np.ones((n, n)) / n
    # apply the centering
    B = -(1 / 2) * center_mat.dot(D_double).dot(center_mat)
    # (3) Determine the m largest eigenvalues
    # (where m is the number of dimensions desired for the output)
    # extract the eigenvalues
    eigenvals, eigenvecs = np.linalg.eigh(B)
    # sort descending
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    # (4) Now, X=eigenvecs.dot(eigen_sqrt_diag),
    # where eigen_sqrt_diag = diag(sqrt(eigenvals))
    eigen_sqrt_diag = np.diag(np.sqrt(eigenvals[0:k]))
    ret = eigenvecs[:, 0:k].dot(eigen_sqrt_diag)
    return ret


def order_groups(X, feature):
    """Assign an order to the values of a categorical feature.

    The function returns an order to the unique values in X[feature] according to
    their similarity based on the other features.
    The distance between two categories is the sum over the distances of each feature.

    Arguments:
    X -- A pandas DataFrame containing all the features to considering in the ordering
    (including the categorical feature to be ordered).
    feature -- String, the name of the column holding the categorical feature to be ordered.
    """

    features = X.columns
    # groups = X[feature].cat.categories.values
    groups = X[feature].unique()
    D_cumu = pd.DataFrame(0, index=groups, columns=groups)
    K = len(groups)
    for j in set(features) - set([feature]):
        D = pd.DataFrame(index=groups, columns=groups)
        # discrete/factor feature j
        # e.g. j = 'color'
        if (X[j].dtypes.name == "category") | (
            (len(X[j].unique()) <= 10) & ("float" not in X[j].dtypes.name)
        ):
            # counts and proportions of each value in j in each group in 'feature'
            cross_counts = pd.crosstab(X[feature], X[j])
            cross_props = cross_counts.div(np.sum(cross_counts, axis=1), axis=0)
            for i in range(K):
                group = groups[i]
                D_values = abs(cross_props - cross_props.loc[group]).sum(axis=1) / 2
                D.loc[group, :] = D_values
                D.loc[:, group] = D_values
        else:
            # continuous feature j
            # e.g. j = 'length'
            # extract the 1/100 quantiles of the feature j
            seq = np.arange(0, 1, 1 / 100)
            q_X_j = X[j].quantile(seq).to_list()
            # get the ecdf (empiricial cumulative distribution function)
            # compute the function from the data points in each group
            X_ecdf = X.groupby(feature)[j].agg(ECDF)
            # apply each of the functions on the quantiles
            # i.e. for each quantile value get the probability that j will take
            # a value less than or equal to this value.
            q_ecdf = X_ecdf.apply(lambda x: x(q_X_j))
            for i in range(K):
                group = groups[i]
                D_values = q_ecdf.apply(lambda x: max(abs(x - q_ecdf[group])))
                D.loc[group, :] = D_values
                D.loc[:, group] = D_values
        D_cumu = D_cumu + D
    # reduce the dimension of the cumulative distance matrix to 1
    D1D = cmds(D_cumu, 1).flatten()
    # order groups based on the values
    order_idx = D1D.argsort()
    groups_ordered = D_cumu.index[D1D.argsort()]
    return pd.Series(range(K), index=groups_ordered)


def quantile_ied(x_vec, q):
    """
    Inverse of empirical distribution function (quantile R type 1).

    More details in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html
    https://en.wikipedia.org/wiki/Quantile

    Arguments:
    x_vec -- A pandas series containing the values to compute the quantile for
    q -- An array of probabilities (values between 0 and 1)
    """

    x_vec = x_vec.sort_values()
    n = len(x_vec) - 1
    m = 0
    j = (n * q + m).astype(int)  # location of the value
    g = n * q + m - j

    gamma = (g != 0).astype(int)
    quant_res = (1 - gamma) * x_vec.shift(1, fill_value=0).iloc[j] + gamma * x_vec.iloc[
        j
    ]
    quant_res.index = q
    # add min at quantile zero and max at quantile one (if needed)
    if 0 in q:
        quant_res.loc[0] = x_vec.min()
    if 1 in q:
        quant_res.loc[1] = x_vec.max()
    return quant_res


def CI_estimate(x_vec, C=0.95):
    """Estimate the size of the confidence interval of a data sample.

    The confidence interval of the given data sample (x_vec) is
    [mean(x_vec) - returned value, mean(x_vec) + returned value].
    """
    alpha = 1 - C
    n = len(x_vec)
    stand_err = x_vec.std() / np.sqrt(n)
    critical_val = 1 - (alpha / 2)
    z_star = stand_err * t.ppf(critical_val, n - 1)
    return z_star

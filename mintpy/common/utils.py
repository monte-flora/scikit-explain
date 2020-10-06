import numpy as np
import pickle
import pandas as pd
from collections import ChainMap
from sklearn.metrics import brier_score_loss, average_precision_score


def brier_skill_score(target_values, forecast_probabilities):
    climo = np.mean((target_values - np.mean(target_values))**2)
    return 1.0 - brier_score_loss(target_values, forecast_probabilities) / climo

def norm_aupdc(targets, predictions):
    """
    Compute the normalized average precision.
    Equations come from Boyd et al. (2012)

    Unachievable Region in Precision-Recall Space and Its Effect on Empirical Evaluation
    """
    skew = np.mean(targets)

    # Number of positive examples
    pos = np.count_nonzero(targets)
    # Number of negative examples
    neg = len(targets) - pos

    ap_min = (1./pos) * np.sum([i/(i+neg) for i in range(pos)])

    ap = average_precision_score(targets, predictions)
    norm_aupdc = (ap - ap_min) / (1. - ap_min)

    return norm_aupdc

def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.
    Parameters
    
    Codes comes directly from sklearn/utils/extmath.py
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
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
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


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
    where the keys are provided by the user
    """
    if not isinstance(fnames, list):
        fnames = [fnames]
    
    data=[]
    for f in fnames:
        with open(f,'rb') as pkl_file:
            data.append( pickle.load(pkl_file) )
    
    if is_all_dict(data):
        return merge_dict(data)
    else:
        return data
    
def save_pickle(fname, data):
    """Save data to a pickle file."""
    with open(fname,'wb') as pkl_file:
        pickle.dump(data, pkl_file)
        
def combine_top_features(results_dict,nvars=None):
    """
    """
    if nvars is None:
        nvars=1000
    combined_features = []
    for model_name in results_dict.keys():
        features = results_dict[model_name]
        combined_features.append(features)
    unique_features = list(set.intersection(*map(set, combined_features)))[:nvars]

    return unique_features
    
def compute_bootstrap_indices(examples, subsample=1.0, nbootstrap=1):
    """
        Routine to generate bootstrap examples
    """
    n_examples = len(examples)
    size = int(subsample * n_examples)
    bootstrap_indices = [np.random.choice(range(n_examples), size=size).tolist() for _ in range(nbootstrap)]
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
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def get_indices_based_on_performance(model, examples, targets, model_output='probability', n_examples=None):
    """
       Determines the best hits, worst false alarms, worst misses, and best
       correct negatives using the data provided during initialization.

       Args:
       ------------------
            model : The model to process
            n_examples: number of "best/worst" examples to return. If None,
                the routine uses the whole dataset

      Return:
            a dictionary containing the indices of each of the 4 categories
            listed above
    """

    #default is to use all examples
    if (n_examples is None):
        n_examples = examples.shape[0]

    #make sure user didn't goof the input
    if (n_examples <= 0):
        print("n_examples less than or equals 0. Defaulting back to all")
        n_examples = examples.shape[0]
        
    if model_output == 'probability':      
        predictions = model.predict_proba(examples.values)[:,1]
    elif model_output == 'regression':
        predictions = model.predict(examples.values)
        
    diff = (targets-predictions)
    data = {'targets': targets, 'predictions': predictions, 'diff': diff}
    df = pd.DataFrame(data)

    if model_output == 'probability':
        nonevent_examples = df[targets==0]
        event_examples = df[targets==1]

        event_examples_sorted_indices = event_examples.sort_values(by='diff',ascending=True).index.values
        nonevent_examples_sorted_indices = nonevent_examples.sort_values(by='diff',ascending=False).index.values

        best_hit_indices = event_examples_sorted_indices[:n_examples].astype(int)
        worst_miss_indices = event_examples_sorted_indices[-n_examples:][::-1].astype(int)
        best_corr_neg_indices = nonevent_examples_sorted_indices[:n_examples].astype(int)
        worst_false_alarm_indices = nonevent_examples_sorted_indices[-n_examples:][::-1].astype(int)

        sorted_dict = {
                    'High Confidence Forecasts Matched to an Event':  best_hit_indices,
                    'Low Confidence Forecasts Matched to an Event': worst_miss_indices,
                    'High Confidence Forecasts NOT Matched to an Event': worst_false_alarm_indices,
                    'Low Confidence Forecasts NOT Matched to an Event': best_corr_neg_indices
                      }

        return sorted_dict
    else:
        print('Mintpy currently does not support performance-based indexing for regression problems')
        pass

def avg_and_sort_contributions(the_dict):
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
    return_dict = {}
    
    # for hits, misses, etc. 
    for key in the_dict.keys():
        df = the_dict[key]
        series    = df.mean(axis=0)
        sorted_df = series.reindex(series.abs().sort_values(ascending=False).index)
        top_vars={var : series[var] for var in list(sorted_df.index)}

        return_dict[key] = top_vars

    return return_dict

def retrieve_important_vars(results, multipass=True):
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
    important_vars_dict = {}
    for model_name in results.keys():
        perm_imp_obj = results[model_name]
        rankings = (
                perm_imp_obj.retrieve_multipass()
                if multipass
                else perm_imp_obj.retrieve_singlepass()
            )
        features = list(rankings.keys())
        important_vars_dict[model_name] = features
            
    return important_vars_dict





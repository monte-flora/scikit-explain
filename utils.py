import numpy as np
import pickle


def load_pickle(fname):
    """
    Load data from a pickle file.
    """
    with open(fname,'rb') as pkl_file:
        data = pickle.load(pkl_file)
        
    return data
    
def save_pickle(fname,data):
    """
    Save data to a pickle file.
    """
    with open(fname,'wb') as pkl_file:
        pickle.load(data, pkl_file)
        
def combine_top_features(results_dict,nvars):
    """
    """
    combined_features = []
    for model_name in results_dict.keys():
        features = results_dict[model_name][:nvars]
        combined_features.extend(features)
    unique_features = list(set(combined_features))
    
    return unique_features
    
def compute_bootstrap_samples(examples, subsample=1.0, nbootstrap=None):

    """
        Routine to generate bootstrap examples
    """

    # get total number of examples
    n_examples = len(examples)

    # below comprehension gets the indices of bootstrap examples
    bootstrap_replicates = np.asarray(
        [
            [
                np.random.choice(range(n_examples))
                for _ in range(int(subsample * n_examples))
            ]
            for _ in range(nbootstrap)
        ]
    )

    return bootstrap_replicates

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

def merge_nested_dict(dicts):
    """
    Merge a list of nested dicts into a single dict
    """
    merged_dict = {}
    for d in dicts:
        key = list(d.keys())[0]
        subkey = list(d[key].keys())[0]
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



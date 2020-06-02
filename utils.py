import numpy as np
import pickle
import pandas as pd

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


def get_indices_based_on_performance(model, examples, targets, n_examples=None):
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
        
    predictions = model.predict_proba(examples)[:,1]
    diff = (targets-predictions)
    data = {'targets': targets, 'predictions': predictions, 'diff': diff}
    df = pd.DataFrame(data)

    nonevent_examples = df[targets==0]
    event_examples = df[targets==1]

    event_examples_sorted_indices = event_examples.sort_values(by='diff',ascending=True).index.values
    nonevent_examples_sorted_indices = nonevent_examples.sort_values(by='diff',ascending=False).index.values

    best_hit_indices = event_examples_sorted_indices[:n_examples].astype(int)
    worst_miss_indices = event_examples_sorted_indices[-n_examples:][::-1].astype(int)
    best_corr_neg_indices = nonevent_examples_sorted_indices[:n_examples].astype(int)
    worst_false_alarm_indices = nonevent_examples_sorted_indices[-n_examples:][::-1].astype(int)

    sorted_dict = {
                    'hits':  best_hit_indices,
                    'misses': worst_miss_indices,
                    'false_alarms': worst_false_alarm_indices,
                    'corr_negs': best_corr_neg_indices
                      }

    return sorted_dict

def avg_and_sort_contributions(the_dict, examples, performance_dict=None):
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

    for key in list(the_dict.keys()):

        df        = the_dict[key]
        series    = df.mean(axis=0)
        sorted_df = series.reindex(series.abs().sort_values(ascending=False).index)

        if (performance_dict is None):
            idxs = examples.index.to_list()
        else:
            idxs = performance_dict[key]

        top_vars = {}
        for var in list(sorted_df.index):
            if var == 'Bias':
                top_vars[var] = {
                                 'Mean Value': None,
                                 'Mean Contribution' : series[var]
                                 }
            else:
                top_vars[var] = {
                        "Mean Value": np.mean(examples.loc[idxs,var].values),
                        "Mean Contribution": series[var],
                    }

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





import xarray as xr 
import numpy as np 


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

def non_increasing(L):
    # Check for decreasing scores. 
    return all(x>=y for x, y in zip(L, L[1:]))

def compute_importance(results, orientation='negative'):
    """
    Compute the importance scores from the permutation importance results.
    The importance score varies depending on the orientation of the 
    loss metric and whether it is multipass or singlepass. 
    
    
    Parameters
    --------------
    results : InterpretToolkit.permutation_importance results 
              xr.Dataset
    orientation : 'positive' or 'negative'
        To compute proper importance scores, it is easier to convert 
        all metrics to imitate loss metric behavior (i.e., lower values are better).
        If 'positive', then scores are converted internally (1-score). Otherwise, 
        by default, scores are treated as being negatively oriented. 
    
    
    Returns
    --------------
    results : xarray.Dataset
        scores for each estimator and multi/singlepass are
        converted to proper importance scores. 
    """
    estimators = results.attrs['estimators used']
    for estimator in estimators:
        orig_score = results[f'original_score__{estimator}'].values
        for mode in ['singlepass', 'multipass']:
            permute_scores = results[f'{mode}_scores__{estimator}'].values            
            
            decreasing = non_increasing(np.mean(permute_scores, axis=1))
            
            if decreasing:
                if orientation == 'negative':
                    # Singlepass MSE 
                    imp = permute_scores - orig_score 
                else:    
                    # Backward Multipass on AUC/AUPDC (permuted_score - (1-orig_score)).
                    # Most positively-oriented metrics top off at 1. 
                    top = np.max(permute_scores)
                    imp = permute_scores - (top-orig_score)
            else:
                if orientation == 'negative':
                    # Multipass MSE
                    top = np.max(permute_scores)
                    imp = (top+orig_score) - permute_scores
                else:
                    # Singlepass AUC/NAUPDC
                    imp = orig_score - permute_scores
            
            # Normalize the importance score so that range is [0,1]
            imp = imp / (np.percentile(imp, 99) - np.percentile(imp,1))
            
            results[f'{mode}_scores__{estimator}'] = ([f'n_vars_{mode}', 'n_bootstrap'], imp)
            
            
    return results 


def to_skexplain_importance(importances, estimator_name, feature_names, method, normalize=True):
    """
    Convert the feature ranking-based scores from non-permutation-importance methods 
    into a importance dataset for plotting purposes 
    """
  
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
        
    if normalize:
        # Normalize the importance score so that range is [0,1]
        scores_ranked = scores_ranked / (np.percentile(scores_ranked, 99) - np.percentile(scores_ranked,1))
    
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




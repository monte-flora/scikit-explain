import numpy as np
import xarray as xr
import pandas as pd
from collections import ChainMap
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import t


class MissingFeaturesError(Exception):
    """ Raised when features are missing. 
        E.g., All features are require for 
        IAS or MEC
    """
    def __init__(self, estimator_name, missing_features):
        self.message = f"""ALE for {estimator_name} was not computed for all features. 
                        These features were missing: {missing_features}"""
        super().__init__(self.message)
    

def check_all_features_for_ale(ale, estimator_names, features):
    """ Is there ALE values for each feature """
    data_vars = ale.data_vars
    for estimator_name in estimator_names:
        _list = [True if f'{f}__{estimator_name}__ale' in data_vars else False for f in features]
        if not all(_list):
            missing_features = np.array(features)[np.where(~np.array(_list))[0]]
            raise MissingFeaturesError(estimator_name, missing_features) 


def flatten_nested_list(list_of_lists):
    """Turn a list of list into a single, flatten list"""
    all_elements_are_lists = all([is_list(item) for item in list_of_lists])
    if not all_elements_are_lists:
        new_list_of_lists = []
        for item in list_of_lists:
            if is_list(item):
                new_list_of_lists.append(item)
            else:
                new_list_of_lists.append([item])
        list_of_lists = new_list_of_lists

    return [item for elem in list_of_lists for item in elem]


def is_dataset(data):
    return isinstance(data, xr.Dataset)


def is_dataframe(data):
    return isinstance(data, pd.DataFrame)


def check_is_permuted(X, X_permuted):
    permuted_features = []
    for f in X.columns:
        if not np.array_equal(X.loc[:, f], X_permuted.loc[:, f]):
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
    contrib_names = feature_names.copy()
    contrib_names += ["Bias"]

    nested_key = results[0][estimator_names[0]].keys()

    dframes = []
    for key in nested_key:
        data = []
        for name in estimator_names:
            contribs_dict = results[0][name][key]
            vals_dict = results[1][name][key]
            data.append(
                [contribs_dict[f] for f in contrib_names]
                + [vals_dict[f] for f in feature_names]
            )
        column_names = [f + "_contrib" for f in contrib_names] + [
            f + "_val" for f in feature_names
        ]
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


def is_tuple(a):
    """Check if argument is a tuple"""
    return isinstance(a, tuple)


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
    """Check if every element of a list are dicts"""
    return all([isinstance(l, dict) for l in alist])


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


def merge_dict(dicts):
    """Merge a list of dicts into a single dict"""
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

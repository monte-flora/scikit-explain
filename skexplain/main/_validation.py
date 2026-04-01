"""Shared validation and normalization helpers for ExplainToolkit methods."""

import itertools
from ..common.utils import is_str, is_list


def normalize_features(features, all_features, allow_2d=False):
    """Normalize a features argument to a list.

    Handles string shortcuts ('all', 'all_2d'), single strings,
    and passes lists through unchanged.

    Parameters
    ----------
    features : str or list
        Feature specification. Can be 'all', 'all_2d' (if allow_2d),
        a single feature name, or a list of feature names/tuples.
    all_features : list of str
        All available feature names (used when features='all' or 'all_2d').
    allow_2d : bool, default=False
        If True, accepts 'all_2d' which generates all 2-feature combinations.

    Returns
    -------
    list
        Normalized list of features.
    """
    if is_str(features):
        if features == "all":
            return list(all_features)
        elif allow_2d and features == "all_2d":
            return list(itertools.combinations(all_features, r=2))
        else:
            return [features]
    return features


def normalize_estimator_names(names, default_names):
    """Normalize estimator_names to a list, defaulting if None.

    Parameters
    ----------
    names : str, list, or None
        Estimator name(s). If None, uses default_names.
    default_names : list of str
        Default estimator names to use when names is None.

    Returns
    -------
    list of str
        Normalized list of estimator names.
    """
    if names is None:
        return list(default_names)
    if is_str(names):
        return [names]
    return list(names)

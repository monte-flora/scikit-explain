"""Shared validation and normalization helpers for ExplainToolkit methods."""

import itertools
import time
import functools
import logging
from ..common.utils import is_str, is_list

logger = logging.getLogger("skexplain")


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


def track_timing(method):
    """Decorator that records computation time in the returned dataset's attrs.

    Adds ``computation_time_seconds`` to ``self.attrs_dict`` before
    ``_append_attributes`` is called. Also logs the elapsed time.

    Only works on methods whose ``self`` has an ``attrs_dict`` attribute.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = method(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        self.attrs_dict["computation_time_seconds"] = round(elapsed, 3)
        # Update attrs on the returned result if it has attrs (Dataset/DataFrame)
        if hasattr(result, "attrs"):
            result.attrs["computation_time_seconds"] = round(elapsed, 3)
        logger.info(
            "%s completed in %.2fs", method.__name__, elapsed,
        )
        return result
    return wrapper

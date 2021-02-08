"""Various and sundry useful functions which are handy for manipulating data or
results of the variable importance"""

import numpy as np
import pandas as pd

from .error_handling import InvalidDataException

__all__ = ["add_ranks_to_dict", "get_data_subset", "make_data_from_columns"]


def add_ranks_to_dict(result, variable_names, scoring_strategy):
    """Takes a list of (var, score) and converts to a dictionary of 
    {var: (rank, score)}
    :param result: a dict of {var_index: score}
    :param variable_names: a list of variable names
    :param scoring_strategy: a function to be used for determining optimal
        variables. Should be of the form ([floats]) -> index
    """
    if len(result) == 0:
        return dict()

    result_dict = dict()
    rank = 0
    while len(result) > 1:
        var_idxs = list(result.keys())
        idxs = np.argsort(var_idxs)
        # Sort by indices to guarantee order
        variables = list(np.array(var_idxs)[idxs])
        scores = list(np.array(list(result.values()))[idxs])
        best_var = variables[scoring_strategy(scores)]
        score = result.pop(best_var)
        result_dict[variable_names[best_var]] = (rank, score)
        rank += 1
    var, score = list(result.items())[0]
    result_dict[variable_names[var]] = (rank, score)
    return result_dict


def get_data_subset(data, rows=None, columns=None):
    """Returns a subset of the data corresponding to the desired rows and
    columns
    :param data: either a pandas dataframe or a numpy array
    :param rows: a list of row indices
    :param columns: a list of column indices
    :returns: data_subset (same type as data)
    """
    if rows is None:
        rows = np.arange(data.shape[0])

    if isinstance(data, pd.DataFrame):
        if columns is None:
            return data.iloc[rows]
        else:
            return data.iloc[rows, columns]
    elif isinstance(data, np.ndarray):
        if columns is None:
            return data[rows]
        else:
            return data[np.ix_(rows, columns)]
    else:
        raise InvalidDataException(
            data, "Data must be a pandas dataframe or numpy array")


def make_data_from_columns(columns_list, index=None):
    """Synthesizes a dataset out of a list of columns
    :param columns_list: a list of either pandas series or numpy arrays
    :returns: a pandas dataframe or a numpy array
    """
    if len(columns_list) == 0:
        raise InvalidDataException(
            columns_list, "Must have at least one column to synthesize dataset")
    if isinstance(columns_list[0], pd.DataFrame) or isinstance(columns_list[0], pd.Series):
        df = pd.concat([c.reset_index(drop=True) for c in columns_list], axis=1)
        if index is not None:
            return df.set_index(index)
        else:
            return df
    elif isinstance(columns_list[0], np.ndarray):
        return np.column_stack(columns_list)
    else:
        raise InvalidDataException(
            columns_list, "Columns_list must come from a pandas dataframe or numpy arrays")


def conditional_permutations(data, n_bins):
    """
    Conditionally permute each feature in a dataset.

    Code appended to the PermutationImportance package by Montgomery Flora 2021. 
    
    Args:
    -------------------
        data : pd.DataFrame or np.ndarray shape=(n_examples, n_features,)
        n_bins : interger 
    
    Returns:
    -------------------
        permuted_data : a permuted version of data
    """
    permuted_data = data.copy()

    for i in range(np.shape(data)[1]):
        # Get the bin values of feature 
        if isinstance(data, pd.DataFrame):
            feature_values = data.iloc[:,i]
        elif isinstance(data, np.ndarray):
            feature_values = data[:,i]
        else:
            raise InvalidDataException(
                    data, "Data must be a pandas dataframe or numpy array")
        
        bin_edges = np.unique(
            np.percentile(
                feature_values,
                np.linspace(0, 100, n_bins + 1),
                interpolation="lower",
                )
            )
        
        bin_indices = np.clip(
                np.digitize(feature_values, bin_edges, right=True) - 1, 0, None
            )
        
        shuffled_indices = bin_indices.copy()
        unique_bin_values = np.unique(bin_indices)
        
        # indices is bin index for a corresponding value of feature  
        for bin_idx in unique_bin_values:
            # idx is the actual index of indices where the bin index == i 
            idx = np.where(bin_indices==bin_idx)[0]
            # Replace the bin indices with a permutation of the actual indices 
            shuffled_indices[idx] = np.random.RandomState(seed=42).permutation(idx)

        if isinstance(data, pd.DataFrame):
            permuted_data.iloc[:,i] = data.iloc[shuffled_indices,i]
        else:
            permuted_data[:,i] = data[shuffled_indices,i]
        
    return permuted_data

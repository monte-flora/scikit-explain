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


def make_data_from_columns(columns_list):
    """Synthesizes a dataset out of a list of columns

    :param columns_list: a list of either pandas series or numpy arrays
    :returns: a pandas dataframe or a numpy array
    """
    if len(columns_list) == 0:
        raise InvalidDataException(
            columns_list, "Must have at least one column to synthesize dataset")
    if isinstance(columns_list[0], pd.DataFrame):
        return pd.concat(columns_list, axis=1)
    elif isinstance(columns_list[0], np.ndarray):
        return np.column_stack(columns_list)
    else:
        raise InvalidDataException(
            columns_list, "Columns_list must come from a pandas dataframe or numpy arrays")

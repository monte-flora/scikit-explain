"""In a variable importance method, the ``scoring_strategy`` is a function which 
is used to determine which of the scores corresponding to a given variable 
indicates that the variable is "most important". This will be dependent on the
particular type of object which is returned as a score.

Here, we provide a few functions which can be used directly as scoring 
strategies as well as some utilities for construction scoring strategies. 
Moreover, we also provide a dictionary of aliases for several commonly used
strategies in ``VALID_SCORING_STRATEGIES``.
"""

import numpy as np

from .error_handling import InvalidStrategyException

__all__ = [
    "verify_scoring_strategy",
    "VALID_SCORING_STRATEGIES",
    "argmin_of_mean",
    "argmax_of_mean",
    "indexer_of_converter",
]


def verify_scoring_strategy(scoring_strategy):
    """Asserts that the scoring strategy is valid and interprets various strings

    :param scoring_strategy: a function to be used for determining optimal
        variables or a string. If a function, should be of the form
        ``([some value]) -> index``. If a string, must be one of the options in
        ``VALID_SCORING_STRATEGIES``
    :returns: a function to be used for determining optimal variables
    """
    if callable(scoring_strategy):
        return scoring_strategy
    elif scoring_strategy in VALID_SCORING_STRATEGIES:
        return VALID_SCORING_STRATEGIES[scoring_strategy]
    else:
        raise InvalidStrategyException(
            scoring_strategy, options=list(VALID_SCORING_STRATEGIES.keys())
        )


class indexer_of_converter(object):
    """This object is designed to help construct a scoring strategy by breaking
    the process of determining an optimal score into two pieces:
    First, each of the scores are converted to a simpler representation. For
    instance, an array of scores resulting from a bootstrapped evaluation method
    may be converted to just their mean.
    Second, each of the simpler representations are compared to determine the
    index of the one which is most optimal. This is typically just an ``argmin``
    or ``argmax`` call.
    """

    def __init__(self, indexer, converter):
        """Constructs a function which first converts all objects in a list to
        something simpler and then uses the indexer to determine the index of
        the most "optimal" one

        :param indexer: a function which converts a list of probably simply
            values (like numbers) to a single index
        :param converter: a function which converts a single more complex object
            to a simpler one (like a single number)
        """
        self.indexer = indexer
        self.converter = converter

    def __call__(self, scores):
        """Finds the index of the most "optimal" score in a list"""
        return self.indexer([self.converter(score) for score in scores])


argmin_of_mean = indexer_of_converter(np.argmin, np.mean)
argmax_of_mean = indexer_of_converter(np.argmax, np.mean)


VALID_SCORING_STRATEGIES = {
    "max": argmax_of_mean,
    "maximize": argmax_of_mean,
    "argmax": np.argmax,
    "min": argmin_of_mean,
    "minimize": argmin_of_mean,
    "argmin": np.argmin,
    "argmin_of_mean": argmin_of_mean,
    "argmax_of_mean": argmax_of_mean,
}

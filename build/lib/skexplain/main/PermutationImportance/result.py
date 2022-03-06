"""The ``ImportanceResult`` is an object which keeps track of the full context 
and scoring determined by a variable importance method. Because the variable 
importance methods iteratively determine the next most important variable, this
yields a sequence of pairs of "contexts" (i.e. the previous ranks/scores of 
variables) and "results" (i.e. the current ranks/scores of variables). This
object keeps track of those pairs and additionally provides methods for the easy
retrieve of both the results with empty context (singlepass, Breiman) and the
most complete context (multipass, Lakshmanan). Further, it enables iteration 
over the ``(context, results)`` pairs and for indexing into the list of pairs.
"""

import warnings

try:
    from itertools import izip as zip
except ImportError:  # python3
    pass

from .error_handling import FullImportanceResultWarning


class ImportanceResult(object):
    """Houses the result of any importance method, which consists of a
    sequence of contexts and results. An individual result can only be truly
    interpreted correctly in light of the corresponding context. This object
    allows for indexing into the contexts and results and also provides
    convenience methods for retrieving the results with no context and the
    most complete context"""

    def __init__(self, method, variable_names, original_score):
        """Initializes the results object with the method used and a list of
        variable names

        :param method: string for the type of variable importance used
        :param variable_names: a list of names for variables
        :param original_score: the score of the model when no variables are
            important
        """
        self.method = method
        self.variable_names = variable_names
        self.original_score = original_score
        # The initial context is "empty"
        self.contexts = [{}]
        self.results = list()

        self.complete = False

    def add_new_results(self, new_results, next_important_variable=None):
        """Adds a new round of results. Warns if the ImportanceResult is already
        complete

        :param new_results: a dictionary with keys of variable names and values
            of ``(rank, score)``
        :param next_important_variable: variable name of the next most important
            variable. If not given, will select the variable with the smallest
            rank
        """
        if not self.complete:
            if next_important_variable is None:
                next_important_variable = min(
                    new_results.keys(), key=lambda key: new_results[key][0]
                )
            self.results.append(new_results)
            new_context = self.contexts[-1].copy()
            self.contexts.append(new_context)
            __, score = new_results[next_important_variable]
            self.contexts[-1][next_important_variable] = (len(self.results) - 1, score)
            # Check to see if this result could constitute the last possible one
            if len(self.results) == len(self.variable_names):
                self.results.append(dict())
                self.complete = True
        else:
            warnings.warn(
                "Cannot add new result to full ImportanceResult",
                FullImportanceResultWarning,
            )

    def retrieve_singlepass(self):
        """Returns the singlepass results as a dictionary with keys of variable
        names and values of ``(rank, score)``."""
        return self.results[0]
    
    def retrieve_all_iterations(self):
        """Returns the singlepass results for all multipass iterations
        """
        return self.results

    def retrieve_multipass(self):
        """Returns the multipass results as a dictionary with keys of variable
        names and values of ``(rank, score)``."""
        return self.contexts[-1]

    def __iter__(self):
        """Iterates over pairs of contexts and results"""
        return zip(self.contexts, self.results)

    def __getitem__(self, index):
        """Retrieves the ith pair of ``(context, result)``"""
        if index < 0:
            index = len(self.results) + index
        return (self.contexts[index], self.results[index])

    def __len__(self):
        """Returns the total number of results computed"""
        return len(self.results)

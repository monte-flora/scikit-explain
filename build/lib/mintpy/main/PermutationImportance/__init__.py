"""PermutationImportance provides several methods for computing data-based
variable importance. All of the methods require data to be used for training 
and/or scoring and a function for scoring, as well as a method for converting 
the scores to relative rankings of the variables. Here, we provide functions at 
three different levels of abstraction:

1) Model-based: Typically, variable importance is computed with respect to a 
particular model. In this case, the function for scoring is in fact either a
performance metric or an error or loss function. All functions of this type are
prefixed with "sklearn_" because they are designed for use primarily with 
scikit-learn models.

2) Method-specific: In some cases, variable importance is computed either over 
the dataset itself (rather than a model) or for a model which is incompatible
with scikit-learn. Here, the function for scoring must be given in terms of the
training data and the scoring data. If you wish to score using a model which is
not compatible with scikit-learn, you may still find utility in the tools 
provided in PermutationImportance.sklearn_api. All functions of this type are 
named specifically for the method they employ

3) Method-agnostic: There are other data-based methods for computing variable
importance beyond the ones implemented here. If you wish to design your own
variable importance, you can still take advantage of the generalized algorithm
for computing data-based variable importances as well as the multithreaded 
functionality implemented in "abstract_variable_importance". In order to use
this function, you will need to design your own strategy for providing the 
datasets to be used at each iteration. Please see 
PermutationImportance.abstract_runner.abstract_variable_importance and
PermutationImportance.selection_strategies for more information.

In addition to these various data-based variable importance methods, we provide
some helpful metrics in PermutationImportance.metrics as well as some other 
tools for implementing your own custom scoring functions.

For more information on the various variable importance methods as well as 
examples of usage, please see the documentation at
https://github.com/gelijergensen/PermutationImportance

@author G. Eli Jergensen <gelijergensen@ou.edu>"""

from .abstract_runner import abstract_variable_importance
from . import metrics
from .permutation_importance import *
from .sequential_selection import *
from .result import ImportanceResult
from . import sklearn_api

__version__ = '1.2.1.5'

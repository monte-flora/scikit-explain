"""Permutation Importance determines which variables are important by comparing
performance on a dataset where some of the variables are permuted in their 
individual columns to performance on the dataset without any permutation. The
permutation of an individual variable in this manner has the effect of breaking
any relationship between the input variable and the target. The variable which,
when permuted, results in the worst performance is typically taken as the most
important variable.

Typically, when using a performance metric or skill score with Permutation 
Importance, the ``scoring_strategy`` should be to minimize the performance. On 
the other hand, when using an error or loss function, the ``scoring_strategy`` 
should be to maximize the error or loss function."""

import numpy as np
	
from .abstract_runner import abstract_variable_importance
from .selection_strategies import PermutationImportanceSelectionStrategy
from .sklearn_api import score_trained_sklearn_model, score_trained_sklearn_model_with_probabilities

__all__ = ["permutation_importance", "sklearn_permutation_importance"]

#fake_data = np.zeros((3,3))
#variable_names = ['' for i in range(len(fake_data))]
#from sklearn.metrics import roc_auc_score

def permutation_importance(scoring_data, scoring_fn, scoring_strategy, variable_names=None, nimportant_vars=None, njobs=1):
	"""Performs permutation importance over data given a particular
	set of functions for scoring and determining optimal variables

	:param scoring_data: a 2-tuple ``(inputs, outputs)`` for scoring in the
		``scoring_fn``
	:param scoring_fn: a function to be used for scoring. Should be of the form
		``(training_data, scoring_data) -> some_value``
	:param scoring_strategy: a function to be used for determining optimal
		variables. Should be of the form ``([some_value]) -> index``
	:param variable_names: an optional list for variable names. If not given,
		will use names of columns of data (if pandas dataframe) or column
		indices
	:param nimportant_vars: number of variables to compute multipass importance
		for. Defaults to all variables
	:param njobs: an integer for the number of threads to use. If negative, will
		use ``num_cpus + njobs``. Defaults to 1
	:returns: :class:`PermutationImportance.result.ImportanceResult` object 
		which contains the results for each run
	"""
	# We don't need the training data, so pass empty arrays to the abstract runner
	if scoring_data is None:
		raise ValueError("Must declare scoring data!")	 
	else:
		return abstract_variable_importance(training_data=(np.array([]), np.array([])), 
					scoring_data=scoring_data, 
					scoring_fn=scoring_fn, 
					scoring_strategy=scoring_strategy, 
					selection_strategy=PermutationImportanceSelectionStrategy, 
					variable_names=variable_names, 
					nimportant_vars=nimportant_vars, 
					njobs=njobs
	 	)

def sklearn_permutation_importance( model, 
									scoring_data, 
									evaluation_fn, 
									scoring_strategy, 
									variable_names=None, 
									nimportant_vars=None, njobs=1, nbootstrap=1, subsample=1, **kwargs):
	"""Performs permutation importance for a particular model, 
	``scoring_data``, ``evaluation_fn``, and strategy for determining optimal 
	variables

	:param model: a trained sklearn model
	:param scoring_data: a 2-tuple ``(inputs, outputs)`` for scoring in the
		``scoring_fn``
	:param evaluation_fn: a function which takes the deterministic or 
		probabilistic model predictions and scores them against the true 
		values. Must be of the form ``(truths, predictions) -> some_value``
		Probably one of the metrics in 
		:mod:`PermutationImportance.metrics` or 
		`sklearn.metrics <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
	:param scoring_strategy: a function to be used for determining optimal
		variables. Should be of the form ``([some_value]) -> index``
	:param variable_names: an optional list for variable names. If not given,
		will use names of columns of data (if pandas dataframe) or column
		indices
	:param nimportant_vars: number of variables to compute multipass importance
		for. Defaults to all variables
	:param njobs: an integer for the number of threads to use. If negative, will
		use ``num_cpus + njobs``. Defaults to 1
	:param nbootstrap: number of times to perform scoring on each variable.
		Results over different bootstrap iterations are averaged. Defaults to 1
	:param subsample: number of elements to sample (with replacement) per
		bootstrap round. If between 0 and 1, treated as a fraction of the number
		of total number of events (e.g. 0.5 means half the number of events).
		If not specified, subsampling will not be used and the entire data will
		be used (without replacement)
	:param kwargs: all other kwargs will be passed on to the ``evaluation_fn``
	:returns: :class:`PermutationImportance.result.ImportanceResult` object 
		which contains the results for each run
	"""
	# Check if the data is probabilistic
	#if len(scoring_data[1].shape) > 1 and scoring_data[1].shape[1] > 1:
	if len(np.unique(scoring_data[1])) == 2:	 
		scoring_fn = score_trained_sklearn_model_with_probabilities(
			model, evaluation_fn, nbootstrap=nbootstrap, subsample=subsample, **kwargs)
	else:
		scoring_fn = score_trained_sklearn_model(
			model, evaluation_fn, nbootstrap=nbootstrap, subsample=subsample, **kwargs)

	return permutation_importance(scoring_data=scoring_data, scoring_fn=scoring_fn, scoring_strategy=scoring_strategy, variable_names=variable_names, nimportant_vars=nimportant_vars, njobs=njobs)

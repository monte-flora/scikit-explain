"""While the various variable importance methods can, in general, for many 
different situations, such as evaluating the model-agnostic presence of 
information withing a dataset, the most typical application of the method is to
determine the importance of variables as evaluated by a particular model. The 
tools provide here are useful to assist in the training and evaluation of 
sklearn models. This is done by wrapping the training and evaluation of the 
model into a single function which is then used as the ``scoring_fn`` of a 
generalized variable importance method.

All of the variable importance methods with a ``sklearn_`` prefix use these 
tools to determine 1) whether to retrain a model at each step (as is necessary 
for Sequential Selection, but not for Permutation Importance) and 2) how to 
evaluate the resulting predictions of a model.

Here, the powerhouse is the ``model_scorer`` object, which handles all of the 
typical use-cases for any model by separately applying a training, prediction,
and evaluation function. Supplied with proper functions for each of these, the
``model_scorer`` object could also be implemented to score other types of 
models, such as Keras models."""

import numpy as np
from sklearn.base import clone

from .utils import get_data_subset, bootstrap_generator


__all__ = [
    "model_scorer",
    "score_untrained_sklearn_model",
    "score_untrained_sklearn_model_with_probabilities",
    "score_trained_sklearn_model",
    "score_trained_sklearn_model_with_probabilities",
    "train_model",
    "get_model",
    "predict_model",
    "predict_proba_model",
]


def train_model(model, training_inputs, training_outputs):
    """Trains a scikit-learn model and returns the trained model"""
    if training_inputs.shape[1] == 0:
        # No data to train over, so don't bother
        return None
    cloned_model = clone(model)
    return cloned_model.fit(training_inputs, training_outputs)


def get_model(model, training_inputs, training_outputs):
    """Just return the trained model"""
    return model


def predict_model(model, scoring_inputs):
    """Uses a trained scikit-learn model to predict over the scoring data"""
    return model.predict(scoring_inputs)


def predict_proba_model(model, scoring_inputs):
    """Uses a trained scikit-learn model to predict class probabilities for the
    scoring data"""
    return model.predict_proba(scoring_inputs)[:, 1]


class model_scorer(object):
    """General purpose scoring method which takes a particular model, trains the
    model over the given training data, uses the trained model to predict on the
    given scoring data, and then evaluates those predictions using some
    evaluation function. Additionally provides the tools for bootstrapping the
    scores and providing a distribution of scores to be used for statistics.
    """

    def __init__(
        self,
        model,
        training_fn,
        prediction_fn,
        evaluation_fn,
        nimportant_vars=1,
        default_score=0.0,
        n_permute=1,
        subsample=1,
        **kwargs
    ):
        """Initializes the scoring object by storing the training, predicting,
        and evaluation functions

        :param model: a scikit-learn model
        :param training_fn: a function for training a scikit-learn model. Must
            be of the form ``(model, training_inputs, training_outputs) ->
            trained_model | None``. If the function returns ``None``, then it is
            assumed that the model training failed.
            Probably :func:`PermutationImportance.sklearn_api.train_model` or
            :func:`PermutationImportance.sklearn_api.get_model`
        :param predicting_fn: a function for predicting on scoring data using a
            scikit-learn model. Must be of the form ``(model, scoring_inputs) ->
            predictions``. Predictions may be either deterministic or
            probabilistic, depending on what the evaluation_fn accepts.
            Probably :func:`PermutationImportance.sklearn_api.predict_model` or
            :func:`PermutationImportance.sklearn_api.predict_proba_model`
        :param evaluation_fn: a function which takes the deterministic or
            probabilistic model predictions and scores them against the true
            values. Must be of the form ``(truths, predictions) -> some_value``
            Probably one of the metrics in
            :mod:`PermutationImportance.metrics` or
            `sklearn.metrics <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
        :param default_score: value to return if the model cannot be trained
        :param nbootstrap: number of times to perform scoring on each variable.
            Results over different bootstrap iterations are averaged. Defaults
            to None, which will not perform bootstrapping
        :param subsample: number of elements to sample (with replacement) per
            bootstrap round. If between 0 and 1, treated as a fraction of the number
            of total number of events (e.g. 0.5 means half the number of events).
            If not specified, subsampling will not be used and the entire data will
            be used (without replacement)
        """

        self.model = model
        self.training_fn = training_fn
        self.prediction_fn = prediction_fn
        self.evaluation_fn = evaluation_fn
        self.default_score = default_score
        self.n_permute = n_permute
        self.subsample = subsample
        self.kwargs = kwargs
        self.random_seed = kwargs.get('random_seed', 42)
        
    def __call__(self, training_data, scoring_data, var_idx=None):
        """Uses the training, predicting, and evaluation functions to score the
        model given the training and scoring data

        :param training_data: (training_input, training_output)
        :param scoring_data: (scoring_input, scoring_output)
        :returns: either a single value or an array of values
        """
        (training_inputs, training_outputs) = training_data
        (scoring_inputs, scoring_outputs) = scoring_data

        subsample = (
            int(len(scoring_data[0]) * self.subsample)
            if self.subsample <= 1
            else self.subsample
        )

        # Try to train model
        trained_model = self.training_fn(self.model, training_inputs, training_outputs)
        # If we didn't succeed in training (probably because there weren't any
        # training predictors), return the default_score
        if trained_model is None:
            if self.n_permute == 1:
                return [self.default_score]
            else:
                return np.full((self.n_permute,), self.default_score)
            
        # Predict
        if var_idx is None:
            ###print('Getting the original score!') 
            predictions = self.prediction_fn(trained_model, scoring_inputs)
            return [self.evaluation_fn(scoring_outputs, predictions)]*self.n_permute
        
        random_states = bootstrap_generator(self.n_permute, seed=self.random_seed)
        if self.n_permute == 1 and subsample == int(len(scoring_data[0])):
            ###print('Only one permutation and no subsampling!')
            shuffled_indices = random_states[0].permutation(scoring_outputs.shape[0])
            permuted_scoring_inputs = scoring_inputs.copy()
            permuted_scoring_inputs[:,var_idx] = scoring_inputs[shuffled_indices,var_idx]
            permuted_predictions = self.prediction_fn(trained_model, permuted_scoring_inputs)
            return [self.evaluation_fn(scoring_outputs, permuted_predictions,)]
        
        elif self.n_permute == 1 and subsample != int(len(scoring_data[0])):
            ###print('Only one permutation, but with subsampling!')
            scores = []
            rows = random_states[0].choice(scoring_outputs.shape[0], subsample)
            subsampled_scoring_outputs = get_data_subset(scoring_outputs, rows)
            subsampled_scoring_inputs = get_data_subset(scoring_inputs, rows)
            
            shuffled_indices = random_states[0].permutation(subsampled_scoring_outputs.shape[0])
            permuted_scoring_inputs = subsampled_scoring_inputs.copy()
            permuted_scoring_inputs[:,var_idx] = subsampled_scoring_inputs[shuffled_indices,var_idx]
            permuted_predictions = self.prediction_fn(trained_model, permuted_scoring_inputs)
   
            scores.append(
                    self.evaluation_fn(
                        subsampled_scoring_outputs,
                        permuted_predictions,
                    )
                )
            return np.array(scores) 
        else:
            #print('Multiple permutations and subsampling!')
            # Bootstrap the scores
            scores = []
            # This function controls the randomness of the bootstrapping.
            # The samples are different per bootstrap iteration (but controlled) 
            # , but this same sampling is repeated per multipass iteration. 
            for random_state in random_states:
                rows = random_state.choice(scoring_outputs.shape[0], subsample)
                
                subsampled_scoring_outputs = get_data_subset(scoring_outputs, rows)
                subsampled_scoring_inputs = get_data_subset(scoring_inputs, rows)
            
                shuffled_indices = random_state.permutation(subsampled_scoring_outputs.shape[0])
                permuted_scoring_inputs = subsampled_scoring_inputs.copy()
                permuted_scoring_inputs[:,var_idx] = subsampled_scoring_inputs[shuffled_indices,var_idx]
                permuted_predictions = self.prediction_fn(trained_model, permuted_scoring_inputs)
                
                scores.append(
                    self.evaluation_fn(
                        subsampled_scoring_outputs,
                        permuted_predictions,
                    )
                )

            return np.array(scores)


def score_untrained_sklearn_model(
    model, evaluation_fn, nbootstrap=None, subsample=1, **kwargs
):
    """A convenience method which uses the default training and the
    deterministic prediction methods for scikit-learn to evaluate a model

    :param model: a scikit-learn model
    :param evaluation_fn: a function which takes the deterministic or
        probabilistic model predictions and scores them against the true
        values. Must be of the form ``(truths, predictions) -> some_value``
        Probably one of the metrics in
        :mod:`PermutationImportance.metrics` or
        `sklearn.metrics <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
    :param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    :param subsample: number of elements to sample (with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events (e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used (without replacement)
    :param kwargs: all other kwargs passed on to the evaluation_fn
    :returns: a callable which accepts ``(training_data, scoring_data)`` and
        returns some value (probably a float or an array of floats)
    """
    return model_scorer(
        model,
        training_fn=train_model,
        prediction_fn=predict_model,
        evaluation_fn=evaluation_fn,
        nbootstrap=nbootstrap,
        subsample=subsample,
        **kwargs
    )


def score_untrained_sklearn_model_with_probabilities(
    model, evaluation_fn, nbootstrap=None, subsample=1, **kwargs
):
    """A convenience method which uses the default training and the
    probabilistic prediction methods for scikit-learn to evaluate a model

    :param model: a scikit-learn model
    :param evaluation_fn: a function which takes the deterministic or
        probabilistic model predictions and scores them against the true
        values. Must be of the form ``(truths, predictions) -> some_value``
        Probably one of the metrics in
        :mod:`PermutationImportance.metrics` or
        `sklearn.metrics <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
    :param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    :param subsample: number of elements to sample (with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events (e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used (without replacement)
    :param kwargs: all other kwargs passed on to the evaluation_fn
    :returns: a callable which accepts ``(training_data, scoring_data)`` and
        returns some value (probably a float or an array of floats)
    """
    return model_scorer(
        model,
        training_fn=train_model,
        prediction_fn=predict_proba_model,
        evaluation_fn=evaluation_fn,
        nbootstrap=nbootstrap,
        subsample=subsample,
        **kwargs
    )


def score_trained_sklearn_model(
    model, evaluation_fn, nbootstrap=None, subsample=1, **kwargs
):
    """A convenience method which does not retrain a scikit-learn model and uses
    deterministic prediction methods to evaluate the model

    :param model: a scikit-learn model
    :param evaluation_fn: a function which takes the deterministic or
        probabilistic model predictions and scores them against the true
        values. Must be of the form ``(truths, predictions) -> some_value``
        Probably one of the metrics in
        :mod:`PermutationImportance.metrics` or
        `sklearn.metrics <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
    :param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    :param subsample: number of elements to sample (with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events (e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used (without replacement)
    :param kwargs: all other kwargs passed on to the evaluation_fn
    :returns: a callable which accepts ``(training_data, scoring_data)`` and
        returns some value (probably a float or an array of floats)
    """
    return model_scorer(
        model,
        training_fn=get_model,
        prediction_fn=predict_model,
        evaluation_fn=evaluation_fn,
        nbootstrap=nbootstrap,
        subsample=subsample,
        **kwargs
    )


def score_trained_sklearn_model_with_probabilities(
    model, evaluation_fn, nbootstrap=None, subsample=1, **kwargs
):
    """A convenience method which does not retrain a scikit-learn model and uses
    probabilistic prediction methods to evaluate the model

    :param model: a scikit-learn model
    :param evaluation_fn: a function which takes the deterministic or
        probabilistic model predictions and scores them against the true
        values. Must be of the form ``(truths, predictions) -> some_value``
        Probably one of the metrics in
        :mod:`PermutationImportance.metrics` or
        `sklearn.metrics <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
    :param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    :param subsample: number of elements to sample (with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events (e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used (without replacement)
    :param kwargs: all other kwargs passed on to the evaluation_fn
    :returns: a callable which accepts ``(training_data, scoring_data)`` and
        returns some value (probably a float or an array of floats)
    """
    return model_scorer(
        model,
        training_fn=get_model,
        prediction_fn=predict_proba_model,
        evaluation_fn=evaluation_fn,
        nbootstrap=nbootstrap,
        subsample=subsample,
        **kwargs
    )

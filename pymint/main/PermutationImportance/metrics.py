"""These are metric functions which can be used to score model predictions 
against the true values. They are designed to be used either as a component of
an ``scoring_fn`` of the method-specific variable importance methods or 
stand-alone as the ``evaluation_fn`` of a model-based variable importance 
method.

In addition to these metrics, all of the metrics and loss functions provided in
`sklearn.metrics <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
should also work."""


import numpy as np

from .error_handling import (
    AmbiguousProbabilisticForecastsException,
    UnmatchingProbabilisticForecastsException,
    UnmatchedLengthPredictionsException,
)


__all__ = ["gerrity_score", "peirce_skill_score", "heidke_skill_score"]


def gerrity_score(truths, predictions, classes=None):
    """Determines the Gerrity Score, returning a scalar. See `here <http://www.cawcr.gov.au/projects/verification/#Methods_for_multi-category_forecasts>`_
    for more details on the Gerrity Score

    :param truths: The true labels of these data
    :param predictions: The predictions of the model
    :param classes: an ordered set for the label possibilities. If not given,
        will be deduced from the truth values
    :returns: a single value for the gerrity score
    """
    table = _get_contingency_table(truths, predictions, classes)
    return _gerrity_score(table)


def peirce_skill_score(truths, predictions, classes=None):
    """Determines the Peirce Skill Score (True Skill Score), returning a scalar.
    See `here <http://www.cawcr.gov.au/projects/verification/#Methods_for_multi-category_forecasts>`_
    for more details on the Peirce Skill Score

    :param truths: The true labels of these data
    :param predictions: The predictions of the model
    :param classes: an ordered set for the label possibilities. If not given,
        will be deduced from the truth values
    :returns: a single value for the peirce skill score
    """
    table = _get_contingency_table(truths, predictions, classes)
    return _peirce_skill_score(table)


def heidke_skill_score(truths, predictions, classes=None):
    """Determines the Heidke Skill Score, returning a scalar. See
    `here <http://www.cawcr.gov.au/projects/verification/#Methods_for_multi-category_forecasts>`_
    for more details on the Peirce Skill Score

    :param truths: The true labels of these data
    :param predictions: The predictions of the model
    :param classes: an ordered set for the label possibilities. If not given,
        will be deduced from the truth values
    :returns: a single value for the heidke skill score
    """
    table = _get_contingency_table(truths, predictions, classes)
    return _heidke_skill_score(table)


def _get_contingency_table(truths, predictions, classes=None):
    """Uses the truths and predictions to compute a contingency matrix

    :param truths: The true labels of these data
    :param predictions: The predictions of the model
    :param classes: an ordered set for the label possibilities. If not given,
        will be deduced from the truth values if possible
    :returns: a numpy array of shape (num_classes, num_classes)
    """
    if len(truths) != len(predictions):
        raise UnmatchedLengthPredictionsException(truths, predictions)
    if len(truths.shape) == 2:
        # Fully probabilistic model
        if len(predictions.shape) != 2 or predictions.shape[1] != truths.shape[1]:
            raise UnmatchingProbabilisticForecastsException(truths, predictions)
        table = np.zeros((truths.shape[1], truths.shape[1]), dtype=np.float32)
        trues = np.argmax(truths, axis=1)
        preds = np.argmax(predictions, axis=1)
        for true, pred in zip(trues, preds):
            table[pred, true] += 1
    else:
        if len(predictions.shape) == 2:
            # in this case, we require the class listing
            if classes is None:
                raise AmbiguousProbabilisticForecastsException(truths, predictions)
            preds = np.take(classes, np.argmax(predictions, axis=1))
        else:
            preds = predictions
        # Truths and predictions are now both deterministic
        if classes is None:
            classes = np.unique(np.append(np.unique(truths), np.unique(preds)))
        table = np.zeros((len(classes), len(classes)), dtype=np.float32)
        for i, c1 in enumerate(classes):
            for j, c2 in enumerate(classes):
                table[i, j] = [
                    p == c1 and t == c2 for p, t in zip(preds, truths)
                ].count(True)
    return table


def _peirce_skill_score(table):
    """This function is borrowed with modification from the hagelslag repository
    MulticlassContingencyTable class. It is used here with permission of
    David John Gagne II <djgagne@ou.edu>

    Multiclass Peirce Skill Score (also Hanssen and Kuipers score, True Skill Score)
    """
    n = float(table.sum())
    nf = table.sum(axis=1)
    no = table.sum(axis=0)
    correct = float(table.trace())
    no_squared = (no * no).sum()
    if n ** 2 == no_squared:
        return correct / n
    else:
        return (n * correct - (nf * no).sum()) / (n ** 2 - no_squared)


def _heidke_skill_score(table):
    """This function is borrowed with modification from the hagelslag repository
    MulticlassContingencyTable class. It is used here with permission of
    David John Gagne II <djgagne@ou.edu>
    """
    n = float(table.sum())
    nf = table.sum(axis=1)
    no = table.sum(axis=0)
    correct = float(table.trace())
    return (correct / n - (nf * no).sum() / n ** 2) / (1 - (nf * no).sum() / n ** 2)


def _gerrity_score(table):
    """This function is borrowed with modification from the hagelslag repository
    MulticlassContingencyTable class. It is used here with permission of
    David John Gagne II <djgagne@ou.edu>

    Gerrity Score, which weights each cell in the contingency table by its
    observed relative frequency.
    """
    k = table.shape[0]
    n = float(table.sum())
    p_o = table.sum(axis=0) / n
    p_sum = np.cumsum(p_o)[:-1]
    a = (1.0 - p_sum) / p_sum
    s = np.zeros(table.shape, dtype=float)
    for (i, j) in np.ndindex(*s.shape):
        if i == j:
            s[i, j] = 1.0 / (k - 1.0) * (np.sum(1.0 / a[0:j]) + np.sum(a[j : k - 1]))
        elif i < j:
            s[i, j] = (
                1.0
                / (k - 1.0)
                * (np.sum(1.0 / a[0:i]) - (j - i) + np.sum(a[j : k - 1]))
            )
        else:
            s[i, j] = s[j, i]
    return np.sum(table / float(table.sum()) * s)


# This is unready for use, but is included here in case I decide to use it in future
# def categorical_metascorer(scoring_fn, category=None, selection_strategy=None):
#     """This is a meta-scorer which converts a binary-class scorer to a multi-class scorer for use above using a
#     one-versus-rest strategy or another specified strategy

#     :param scoring_fn: function which is a binary-class scorer (like bias). Must be of the form:
#         (new_predictions, truths, classes, particular class) -> float
#     :param category: the identity of the class to consider (if doing one-versus-rest)
#     :param selection_strategy: either "maximimum", "minimum", "average", or a callable
#         NOTE: if neither category or selection_strategy is specified, prints a warning and defaults to average
#         callable must be of the form (list of scores) -> float
#         category is ignored if selection_strategy is specified
#     :returns: scoring function which wraps correctly around scoring_fn
#     """
#     # First determine whether we are doing ovr or a specified strategy

#     if category is None and selection_strategy is None:
#         print("WARNING: categorical_metascorer defaulting to averaging")
#         selection_strategy = 'average'

#     if 'max' in selection_strategy:
#         selection_strategy = np.max
#     elif 'min' in selection_strategy:
#         selection_strategy = np.min
#     elif 'avg' in selection_strategy or 'average' in selection_strategy:
#         selection_strategy = np.average
#     else:
#         assert callable(
#             selection_strategy), "ERROR: strategy must be 'minimize', 'maximize', 'average' or a callable"

#     def cat_scorer(new_predictions, truths, classes):
#         if selection_strategy is None:  # then we know category isn't None
#             return scoring_fn(new_predictions, truths, classes, category)
#         else:
#             all_scores = [scoring_fn(
#                 new_predictions, truths, classes, bin_class) for bin_class in classes]
#             return selection_strategy(all_scores)

#     return cat_scorer

from functools import partial
from sklearn.metrics._base import _average_binary_score
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import (
    brier_score_loss,
    average_precision_score,
    precision_recall_curve,
)
import numpy as np


def brier_skill_score(y_values, forecast_probabilities):
    """Computes the brier skill score"""
    climo = np.mean((y_values - np.mean(y_values)) ** 2)
    return 1.0 - brier_score_loss(y_values, forecast_probabilities) / climo


def modified_precision(precision, known_skew, new_skew):
    """
    Modify the success ratio according to equation (3) from
    Lampert and Gancarski (2014).
    """
    precision[precision < 1e-5] = 1e-5

    term1 = new_skew / (1.0 - new_skew)
    term2 = (1 / precision) - 1.0

    denom = known_skew + ((1 - known_skew) * term1 * term2)

    return known_skew / denom


def calc_sr_min(skew):
    pod = np.linspace(0, 1, 100)
    sr_min = (skew * pod) / (1 - skew + (skew * pod))
    return sr_min


def _binary_uninterpolated_average_precision(
    y_true, y_score, known_skew, new_skew, pos_label=1, sample_weight=None
):
    precision, recall, _ = precision_recall_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )
    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    if known_skew is not None:
        precision = modified_precision(precision, known_skew, new_skew)
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def min_aupdc(
    y_true, pos_label, average, sample_weight=None, known_skew=None, new_skew=None
):
    """
    Compute the minimum possible area under the performance
    diagram curve. Essentially, a vote of NO for all predictions.
    """
    min_score = np.zeros((len(y_true)))
    average_precision = partial(
        _binary_uninterpolated_average_precision,
        known_skew=known_skew,
        new_skew=new_skew,
        pos_label=pos_label,
    )
    ap_min = _average_binary_score(
        average_precision, y_true, min_score, average, sample_weight=sample_weight
    )

    return ap_min


def norm_aupdc(
    y_true,
    y_score,
    known_skew=None,
    *,
    average="macro",
    pos_label=1,
    sample_weight=None,
    min_method="random",
):
    """
    Compute the normalized modified average precision. Normalization removes
    the no-skill region either based on skew or random classifier performance.
    Modification alters success ratio to be consistent with a known skew.

    Parameters:
    -------------------
        y_true, array of (n_samples,)
            Binary, truth labels (0,1)
        y_score, array of (n_samples,)
            Model predictions (either determinstic or probabilistic)
        known_skew, float between 0 and 1
            Known or reference skew (# of 1 / n_samples) for
            computing the modified success ratio.
        min_method, 'skew' or 'random'
            If 'skew', then the normalization is based on the minimum AUPDC
            formula presented in Boyd et al. (2012).

            If 'random', then the normalization is based on the
            minimum AUPDC for a random classifier, which is equal
            to the known skew.


    Boyd, 2012: Unachievable Region in Precision-Recall Space and Its Effect on Empirical Evaluation, ArXiv
    """
    new_skew = np.mean(y_true)
    if known_skew is None:
        known_skew = new_skew

    y_type = type_of_target(y_true)
    if y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError(
            "Parameter pos_label is fixed to 1 for "
            "multilabel-indicator y_true. Do not set "
            "pos_label or set pos_label to 1."
        )
    elif y_type == "binary":
        # Convert to Python primitive type to avoid NumPy type / Python str
        # comparison. See https://github.com/numpy/numpy/issues/6784
        present_labels = np.unique(y_true).tolist()
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError(
                f"pos_label={pos_label} is not a valid label. It should be "
                f"one of {present_labels}"
            )
    average_precision = partial(
        _binary_uninterpolated_average_precision,
        known_skew=known_skew,
        new_skew=new_skew,
        pos_label=pos_label,
    )

    ap = _average_binary_score(
        average_precision, y_true, y_score, average, sample_weight=sample_weight
    )

    if min_method == "random":
        ap_min = known_skew
    elif min_method == "skew":
        ap_min = min_aupdc(
            y_true,
            pos_label,
            average,
            sample_weight=sample_weight,
            known_skew=known_skew,
            new_skew=new_skew,
        )

    naupdc = (ap - ap_min) / (1.0 - ap_min)

    return naupdc

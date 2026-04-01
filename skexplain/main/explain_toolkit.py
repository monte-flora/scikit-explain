import numpy as np
import pandas as pd

from ..common.attributes import Attributes
from .local_explainer import LocalExplainer
from .global_explainer import GlobalExplainer

from ..common.utils import is_str, is_list, is_tuple
from ..plot.config import PlotConfig

from ._importance_mixin import ImportanceMixin
from ._curves_mixin import CurvesMixin
from ._interaction_mixin import InteractionMixin
from ._attribution_mixin import AttributionMixin
from ._plotting_mixin import PlottingMixin
from ._io_mixin import IOMixin


class ExplainToolkit(
    Attributes,
    ImportanceMixin,
    CurvesMixin,
    InteractionMixin,
    AttributionMixin,
    PlottingMixin,
    IOMixin,
):
    """

    ExplainToolkit is the primary interface of scikit-explain. The modules contained within compute several
    explainability machine learning methods such as

    Feature importance:

        * `permutation_importance`
        * `ale_variance`

    Feature Attributions:

        - `ale`
        - `pd`
        - `ice`
        - `local_attributions`

    Feature Interactions:

        - `interaction_strength`
        - `ale_variance`
        - `perm_based_interaction`
        - `friedman_h_stat`
        - `main_effect_complexity`
        - `ale`
        - `pd`

    Additionally, there are corresponding plotting modules for
    each method, which are designed to produce publication-quality graphics.

    .. note::
        ExplainToolkit is designed to work with estimators that implement predict or predict_proba.

    .. caution::
        ExplainToolkit is only designed to work with binary classification and regression problems.
        In future versions of skexplain, we hope to be compatible with multi-class classification.


    Parameters
    -----------

    estimators : list of tuples of (estimator name, fitted estimator)
        Tuple of (estimator name, fitted estimator object) or list thereof where the
        fitted estimator must implement ``predict`` or ``predict_proba``.
        Multioutput-multiclass classifiers are not supported.

    X : {array-like or dataframe} of shape (n_samples, n_features)
        Training or validation data used to compute the IML methods.
        If numpy.ndarray, must specify `feature_names`.

    y : {list or numpy.array} of shape (n_samples,)
        The target values (class labels in classification, real numbers in regression).

    estimator_output : ``"raw"`` or ``"probability"``
        What output of the estimator should be explained. Determined internally by
        ExplainToolkit. However, if using a classification model, the user
        can set to "raw" for non-probabilistic output.

    feature_names : array-like of shape (n_features,), dtype=str, default=None
        Name of each feature; ``feature_names[i]`` holds the name of the feature
        with index ``i``. By default, the name of the feature corresponds to their numerical
        index for NumPy array and their column name for pandas dataframe.
        Feature names are only required if ``X`` is a numpy.ndarray, as it will be
        converted to a pandas.DataFrame internally.

    seaborn_kws : dict, None, or False (default is None)
        Arguments for the seaborn.set_theme(). By default, we use the following settings.

        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        If False, then seaborn settings are not used.


    Raises
    ---------
    AssertError
        Number of estimator objects is not equal to the number of estimator names given!
    TypeError
        y variable must be numpy array or pandas.DataFrame.
    Exception
        Feature names must be specified if X is a numpy.ndarray.
    ValueError
        estimator_output is not an accepted option.

    """

    def __init__(
        self,
        estimators=None,
        X=None,
        y=None,
        estimator_output=None,
        feature_names=None,
        seaborn_kws=None,
    ):
        if X is None:
            X = pd.DataFrame(np.array([]))
        if y is None:
            y = np.array([])
        self.seaborn_kws = seaborn_kws
        if estimators is not None:
            if not is_list(estimators) and estimators:
                estimators = [estimators]

            # Check that the estimator name is provided!
            for e in estimators:
                if not is_tuple(e):
                    raise TypeError(
                        "The estimators arg must be a tuple of (estimator_name, estimator)!"
                    )
                else:
                    if not is_str(e[0]):
                        raise TypeError(
                            "Estimator name is supposed to be a string. Make sure that the tuple is (estimator_name, estimator)."
                        )

            estimator_names = [e[0] for e in estimators]
            estimators = [e[1] for e in estimators]
        else:
            estimator_names = None

        self.set_estimator_attribute(estimators, estimator_names)
        self.set_y_attribute(y)
        self.set_X_attribute(X, feature_names)
        self.set_estimator_output(estimator_output, estimators)
        self.checked_attributes = True

        # Initialize a global interpret object
        self.global_obj = GlobalExplainer(
            estimators=self.estimators,
            estimator_names=self.estimator_names,
            X=self.X,
            y=self.y,
            estimator_output=self.estimator_output,
            checked_attributes=self.checked_attributes,
        )

        # Initialize a local interpret object
        self.local_obj = LocalExplainer(
            estimators=self.estimators,
            estimator_names=self.estimator_names,
            X=self.X,
            y=self.y,
            estimator_output=self.estimator_output,
            checked_attributes=self.checked_attributes,
        )

        self.attrs_dict = {
            "estimator_output": self.estimator_output,
            "estimators used": self.estimator_names,
        }

        # Initialize plot configuration
        self._plot_config = PlotConfig()
        if seaborn_kws is not None and isinstance(seaborn_kws, dict):
            # Map legacy seaborn_kws to PlotConfig fields
            if "style" in seaborn_kws:
                self._plot_config.style = seaborn_kws["style"]
            if "palette" in seaborn_kws:
                self._plot_config.palette = seaborn_kws["palette"]
            if "font_scale" in seaborn_kws:
                self._plot_config.font_scale = seaborn_kws["font_scale"]
            if "rc" in seaborn_kws:
                self._plot_config.rc = seaborn_kws["rc"]

    def __repr__(self):
        return (
            "ExplainToolkit(estimator=%s \n \
                                 estimator_names=%s \n \
                                 X=%s length:%d \n \
                                 y=%s length:%d \n \
                                 estimator_output=%s \n \
                                 feature_names=%s length %d)"
            % (
                self.estimators,
                self.estimator_names,
                type(self.X),
                len(self.X),
                type(self.y),
                len(self.y),
                self.estimator_output,
                type(self.feature_names),
                len(self.feature_names),
            )
        )

    def set_plotting_config(self, **kwargs):
        """Set plot configuration. Works like seaborn.set_theme().

        Parameters set here become defaults for all subsequent plot calls.
        Per-call arguments override these defaults.

        Parameters
        ----------
        **kwargs
            Any attribute of :class:`~skexplain.plot.config.PlotConfig`.
            See ``PlotConfig`` for the full list.

        Examples
        --------
        >>> explainer.set_plotting_config(
        ...     figsize=(12, 8),
        ...     display_feature_names={"temp2m": "Temperature (2m)"},
        ...     display_units={"temp2m": "K"},
        ...     style="whitegrid",
        ...     base_font_size=14,
        ... )
        >>> # All subsequent plots use these settings:
        >>> explainer.plot_ale(ale_data)
        """
        valid_fields = PlotConfig.field_names()
        for key, val in kwargs.items():
            if key not in valid_fields:
                raise ValueError(
                    f"Unknown config key '{key}'. "
                    f"Available: {valid_fields}"
                )
            setattr(self._plot_config, key, val)

        # Update seaborn_kws for backward compatibility with plot classes
        self.seaborn_kws = self._plot_config.to_seaborn_kws()

    def get_plotting_config(self):
        """Return the current plot configuration.

        Returns
        -------
        PlotConfig
            The current plot configuration dataclass.
        """
        return self._plot_config

    def reset_plotting_config(self):
        """Reset all plot configuration to defaults.

        Restores the default PlotConfig (no custom display names,
        units, colors, or seaborn overrides). Equivalent to creating
        a fresh ExplainToolkit with no seaborn_kws.

        Examples
        --------
        >>> explainer.set_plotting_config(style="whitegrid", base_font_size=16)
        >>> # ... do some plotting ...
        >>> explainer.reset_plotting_config()  # back to defaults
        """
        self._plot_config = PlotConfig()
        self.seaborn_kws = None

    def _append_attributes(self, ds):
        """
        FOR INTERNAL PURPOSES ONLY.

        Append attributes to a xarray.Dataset or pandas.DataFrame

        Parameters
        ----------

        ds : xarray.Dataset or pandas.DataFrame
            Results data from the IML methods

        """

        for key in self.attrs_dict.keys():
            ds.attrs[key] = self.attrs_dict[key]

        return ds

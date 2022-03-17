import shap
import traceback
from .tree_interpreter import TreeInterpreter
import pandas as pd
import numpy as np

from lime.lime_tabular import LimeTabularExplainer

from ..common.attributes import Attributes
from ..common.importance_utils import retrieve_important_vars
from ..common.utils import to_dataframe, determine_feature_dtype
from ..common.metrics import brier_skill_score


from ..common.contrib_utils import (
    get_indices_based_on_performance,
    avg_and_sort_contributions,
)


list_of_acceptable_tree_estimators = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
]


class LocalExplainer(Attributes):

    """
    ExplainToolkit inherits functionality from LocalExplainer and is not meant to be
    instantiated by the end-user.


    LocalExplainer incorporates important methods for explaining local estimator behavior
    for select data instances. The calculations are primarily based on SHAP (source), but also
    includes treeinterpreter (source) for random forests and for other select tree-based methods in
    scikit-learn (see list_of_acceptable_tree_estimators).

    Attributes
    --------------
    estimators : object, list of objects
        A fitted estimator object or list thereof implementing `predict` or
        `predict_proba`.
        Multioutput-multiclass classifiers are not supported.

    estimator_names : string, list
        Names of the estimators (for internal and plotting purposes)

    X : {array-like or dataframe} of shape (n_samples, n_features)
        Training or validation data used to compute the IML methods.
        If ndnumpy array, make sure to specify the feature names

    y : {list or numpy.array} of shape = (n_samples,)
        y values.

    estimator_output : "raw" or "probability"
        What output of the estimator should be evaluated. default is None. If None,
        ExplainToolkit will determine internally what the estimator output is.

    feature_names : array-like of shape (n_features,), dtype=str, default=None
        Name of each feature; `feature_names[i]` holds the name of the feature
        with index `i`.
        By default, the name of the feature corresponds to their numerical
        index for NumPy array and their column name for pandas dataframe.
        Feature names are only required if X is an ndnumpy.array, it will be
        converted to a pandas.DataFrame internally.


    Reference:
        SHAP
        treeinterpreter
    """

    def __init__(
        self,
        estimators,
        estimator_names,
        X,
        y,
        estimator_output,
        feature_names=None,
        checked_attributes=False,
    ):
        # These functions come from the inherited Attributes class
        if not checked_attributes:
            self.set_estimator_attribute(estimators, estimator_names)
            self.set_X_attribute(X, feature_names)
            self.set_y_attribute(y)
            self.set_estimator_output(estimator_output, estimators)
        else:
            self.estimators = estimators
            self.estimator_names = estimator_names
            self.X = X
            self.y = y
            self.feature_names = list(X.columns)
            self.estimator_output = estimator_output

    def _get_local_prediction(
        self,
        method="shap",
        performance_based=True,
        n_samples=100,
        shap_kwargs=None,
        lime_kws=None, 
    ):
        """
        Explain individual predictions using SHAP (SHapley Additive exPlanations;
        https://github.com/slundberg/shap) or treeinterpreter
        (https://blog.datadive.net/random-forest-interpretation-with-scikit-learn/)

        Parameters
        ------------
        method : 'treeinterpreter', 'shap', or 'lime'
            Contributions can be computed using treeinterpreter for tree-based estimators
            or using SHAP for both tree- and non-tree-based estimators
        background_dataset : array (n_samples, n_features)
            A representative (often a K-means or random sample) subset of the
            data used to train the ML estimator. Used for the background dataset
            to compute the expected values for the SHAP calculations.
            Only required for non-tree based estimators.
        performance_based : boolean (default=False)
            If True, will average feature contributions over the best and worst
            performing of the given X. The number of X to average over
            is given by n_samples
        n_samples : interger (default=100)
            Number of samples to compute average over if performance_based = True

        shap_kwargs : dict
            Arguments passed to the shap.Explainer object. See
            https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer
            for details. The main two arguments supported in PyMint is the masker and
            algorithm options. By default, the masker option uses
            masker = shap.maskers.Partition(X, max_samples=100, clustering="correlation") for
            hierarchical clustering by correlations. You can also provide a background dataset
            e.g., background_dataset = shap.sample(X, 100).reset_index(drop=True). The algorithm
            option is set to "auto" by default.

            - masker
            - algorithm


        Returns
        ---------

        results_ds : pandas.DataFrame

        """
        if method not in ["tree_interpreter", "shap", "lime"]:
            raise ValueError(
                """
                             Declared method is not 'tree_interpreter','shap', or 'lime'. 
                             Check for spelling mistake or syntax error!
                             """
            )
        else:
            self.method = method

        # will be returned; a list of pandas dataframes, one for each performance dict key
        contributions_dict = {
            estimator_name: {} for estimator_name in self.estimator_names
        }
        feature_values_dict = {
            estimator_name: {} for estimator_name in self.estimator_names
        }

        for estimator_name, estimator in self.estimators.items():
            # create entry for current estimator
            # self.contributions_dict[estimator_name] = {}
            if performance_based:
                ### print('Computing performance-based contributions...')
                performance_dict = get_indices_based_on_performance(
                    estimator,
                    X=self.X,
                    y=self.y,
                    n_samples=n_samples,
                    estimator_output=self.estimator_output,
                )

                for key, indices in performance_dict.items():
                    cont_dict = self._get_feature_contributions(
                        estimator=estimator,
                        X=self.X.iloc[indices, :],
                        shap_kwargs=shap_kwargs,
                    )

                    contributions_dict[estimator_name][key] = cont_dict
                    feature_values_dict[estimator_name][key] = self.X.iloc[indices, :]

            else:
                cont_dict = self._get_feature_contributions(
                    estimator=estimator,
                    X=self.X,
                    shap_kwargs=shap_kwargs,
                    lime_kws=lime_kws, 
                )

                contributions_dict[estimator_name]["non_performance"] = cont_dict
                feature_values_dict[estimator_name]["non_performance"] = self.X

            # average out the contributions and sort based on contribution
            avg_contrib_dict, avg_feature_val_dict = avg_and_sort_contributions(
                contributions_dict[estimator_name], feature_values_dict[estimator_name]
            )

            contributions_dict[estimator_name] = avg_contrib_dict
            feature_values_dict[estimator_name] = avg_feature_val_dict

        results = (contributions_dict, feature_values_dict)

        results_df = to_dataframe(results, self.estimator_names, self.feature_names)

        return results_df

    def _get_shap_values(
        self,
        estimator,
        X,
        shap_kwargs=None,
    ):
        """
        FOR INTERNAL PURPOSES ONLY.

        """
        masker = shap_kwargs.get("masker", None)
        algorithm = shap_kwargs.get("algorithm", "auto")

        if masker is None:
            raise ValueError(
                """masker in shap_kwargs is None. 
                             This will cause issues with SHAP. We recommend starting with
                             shap_kwargs = {'masker' = shap.maskers.Partition(X, max_samples=100, clustering="correlation")}
                             where X is the original dataset and not the examples SHAP is being computed for. 
                             """
            )

        if self.estimator_output == "probability":
            model = estimator.predict_proba
        else:
            model = estimator.predict

        explainer = shap.Explainer(
            model=model,
            masker=masker,
            algorithm=algorithm,
        )

        shap_results = explainer(X)

        if self.estimator_output == "probability":
            shap_results = shap_results[..., 1]

        contributions = shap_results.values
        bias = shap_results.base_values
        
        return contributions, bias

    def _get_ti_values(self, estimator, X):
        """
        FOR INTERNAL PURPOSES ONLY.
        """
        # check to make sure estimator is of type Tree
        if type(estimator).__name__ not in list_of_acceptable_tree_estimators:
            raise TypeError(
                f""" Unfortunately, tree interpreter does not work on this type of estimator :
                                {type(estimator).__name__}
                            """
            )

        ti = TreeInterpreter(estimator, X)

        prediction, bias, contributions = ti.predict()

        if self.estimator_output == "probability":
            contributions = contributions[:, :, 1]
            bias = bias[:, 1]  # bias is all the same values for first index
        else:
            pass

        return contributions, bias

    
    def _get_lime_values(self, estimator, X, lime_kws):
        """
        Compute the Local Interpretable Model-Agnostic Explanations
        """
        if lime_kws is None:
            raise KeyError('lime_kws is None, but lime_kws must contain training_data!')
        
        lime_kws['feature_names'] = X.columns
        
        # Determine categorical features
        if lime_kws.get('categorical_names', None) is None and lime_kws.get('categorical_features', None) is None:
            features, cat_features = determine_feature_dtype(X, X.columns)
            lime_kws['categorical_names'] = cat_features
        
        
        # Determine whether its a classification or regression task.
        mode = 'regression' 
        if lime_kws.get('mode', None) is None:
            mode = 'classification' if hasattr(estimator, 'predict_proba') else 'regression'
            lime_kws['mode'] = mode
        
        # Set the random state for reproducible results. 
        if lime_kws.get('random_state', None) is None:
            lime_kws['random_state'] = 123 
        
        
        explainer = LimeTabularExplainer(**lime_kws)
        
        if mode == 'classification' and hasattr(estimator, 'predict_proba'):
            predict_fn = estimator.predict_proba 
        else:
            predict_fn = estimator.predict 
        
        # Using X.values[0] rather than X.values since it requires 
        # a (n_features) rather than (1,n_features).
        explanation = explainer.explain_instance(X.values[0], predict_fn, num_features=X.shape[1],)

        sorted_exp = sorted(explanation.local_exp[1], key=lambda x: x[0])
        contributions = np.array([[val[1] for val in sorted_exp]])
        bias = explanation.intercept[1] 
    
        return contributions, bias
        
    
    def _get_feature_contributions(self, estimator, X, shap_kwargs=None, lime_kws=None):
        """
        FOR INTERNAL PURPOSES ONLY.

        Compute the feature contributions either using treeinterpreter or SHAP.

        Parameters
        ------------
        estimator : callable
        X : pandas.DataFrame of shape (n_samples, n_features)

        """
        if self.method == "shap":
            contributions, bias = self._get_shap_values(estimator, X, shap_kwargs)
        elif self.method == "tree_interpreter":
            contributions, bias = self._get_ti_values(estimator, X)
        elif self.method == 'lime':
            contributions, bias = self._get_lime_values(estimator, X, lime_kws)

        n_samples = len(X)
        columns = self.feature_names + ["Bias"]

        if self.estimator_output == "probability":
            contributions *= 100.0
            bias *= 100.

        # A single example.
        if isinstance(bias, float):
            bias = [bias]

        bias_reshaped = np.reshape(bias, (len(bias), 1))
        data = np.concatenate((contributions, bias_reshaped), axis=1)

        return pd.DataFrame(data, columns=columns)

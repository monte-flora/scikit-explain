import shap
import traceback
from .tree_interpreter import TreeInterpreter
import pandas as pd
import numpy as np
from tqdm import tqdm 

from joblib import delayed, Parallel 
from joblib import wrap_non_picklable_objects
from joblib.externals.loky import set_loky_pickler

# This should allow for more shared memory for the LIME calcuations. 
import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

###from lime.lime_tabular import LimeTabularExplainer
from .lime_fast import FastLimeTabularExplainer

from ..common.attributes import Attributes
from ..common.importance_utils import retrieve_important_vars
from ..common.utils import to_dataframe, determine_feature_dtype
from ..common.metrics import brier_skill_score
from ..common.multiprocessing_utils import tqdm_joblib, delayed, Parallel 


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

def _ds_to_df(ds, method, estimator_name, feature_names):
    """Convert the Dataset to DataFrame"""
    df = pd.DataFrame(ds[f'{method}_values__{estimator_name}'].values, 
                                           columns = feature_names)
                   
    df['Bias'] = ds[f'{method}_bias__{estimator_name}'].values
              
    X = pd.DataFrame(ds.X.values, columns=feature_names)
    
    return df, X 
                    


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

    def _average_attributions(
        self,
        method,
        data=None, 
        performance_based=True,
        n_samples=100,
        shap_kws=None,
        lime_kws=None, 
        n_jobs=1
    ):
        """
        Explain individual predictions using SHAP (SHapley Additive exPlanations;
        https://github.com/slundberg/shap) or treeinterpreter
        (https://blog.datadive.net/random-forest-interpretation-with-scikit-learn/)

        Parameters
        ------------
        data: xarray.Dataset 
           Results of :func:`~ExplainToolkit.local_attributions`
        
        performance_based : boolean (default=False)
            If True, will average feature contributions over the best and worst
            performing of the given X. The number of X to average over
            is given by n_samples
        n_samples : interger (default=100)
            Number of samples to compute average over if performance_based = True

        shap_kws : dict
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
                    if data is not None:
                        cont_dict, X = _ds_to_df(data, method, estimator_name, self.feature_names)
                    else:
                        X = self.X.iloc[indices, :]
                        cont_dict = self._get_feature_contributions(
                        estimator=estimator,
                        X=X,
                        shap_kws=shap_kws,
                        lime_kws=lime_kws,
                        method=method, 
                        n_jobs=n_jobs
                        )

                    contributions_dict[estimator_name][key] = cont_dict
                    feature_values_dict[estimator_name][key] = X

            else:
                if data is not None:
                    cont_dict, X = _ds_to_df(data, method, estimator_name, self.feature_names)
                
                else:
                    cont_dict = self._get_feature_contributions(
                    estimator=estimator,
                    X=self.X,
                    shap_kws=shap_kws,
                    lime_kws=lime_kws,
                    method=method, 
                    n_jobs=n_jobs
                    )
                    X = self.X

                contributions_dict[estimator_name]["non_performance"] = cont_dict
                feature_values_dict[estimator_name]["non_performance"] = X

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
        shap_kws=None,
    ):
        """
        FOR INTERNAL PURPOSES ONLY.

        """
        if shap_kws is None:
            shap_kws = {}
        
        masker = shap_kws.get("masker", None)
        algorithm = shap_kws.get("algorithm", "auto")

        if masker is None:
            raise ValueError(
                """masker in shap_kws is None. 
                             This will cause issues with SHAP. We recommend starting with
                             shap_kws = {'masker' = shap.maskers.Partition(X, max_samples=100, clustering="correlation")}
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

        # TODO: generalize for the joint_contributions=True.
        ti = TreeInterpreter(estimator, X, n_jobs=self._n_jobs)

        prediction, bias, contributions = ti.predict()

        if self.estimator_output == "probability":
            contributions = contributions[:, :, 1]
            bias = bias[:, 1]  # bias is all the same values for first index
        else:
            pass

        return contributions, bias

    #@delayed
    #@wrap_non_picklable_objects
    def _explain_lime(self, explainer, predict_fn, X, label):
        """Get explanation from LIME"""
        
        num_features = len(X)
        contrib, bias = explainer.explain_instance(X, predict_fn, label=label, 
                                                 num_features=num_features, num_samples=2500)
        
        #if isinstance(explainer, LimeTabularExplainer):
        #    sorted_exp = sorted(explanation.local_exp[1], key=lambda x: x[0])
        #    contrib = np.array([[val[1] for val in sorted_exp]])[0,:]
        #    bias = explanation.intercept[1]
        #else:
        #contrib, bias = explanation

        return contrib, bias 
    
    
    def _get_lime_values(self, estimator, X, lime_kws):
        """
        Compute the Local Interpretable Model-Agnostic Explanations
        """
        lime_kws['fast_lime'] = True
        
        if lime_kws is None:
            raise KeyError('lime_kws is None, but lime_kws must contain training_data!')
        
        # Convert dataframe to array.
        if isinstance(lime_kws['training_data'], pd.DataFrame):
            lime_kws['training_data'] = lime_kws['training_data'].values
        
        lime_kws['feature_names'] = list(X.columns)
        
        # Determine categorical features
        if lime_kws.get('categorical_names', None) is None and lime_kws.get('categorical_features', None) is None:
            features, cat_features = determine_feature_dtype(X, X.columns)
            lime_kws['categorical_names'] = cat_features
        
        # Determine whether its a classification or regression task.
        if lime_kws.get('mode', None) is None:
            mode = 'classification' if hasattr(estimator, 'predict_proba') else 'regression'
            lime_kws['mode'] = mode
        
        # Set the random state for reproducible results. 
        if lime_kws.get('random_state', None) is None:
            lime_kws['random_state'] = 123 
        
        if lime_kws['fast_lime']:
            del lime_kws['fast_lime']
            lime_kws['feature_names'] = self.feature_names
            explainer = FastLimeTabularExplainer(**lime_kws)
        else:
            del lime_kws['fast_lime']
            explainer = LimeTabularExplainer(**lime_kws)
        
        if lime_kws['mode'] == 'classification' and hasattr(estimator, 'predict_proba'):
            predict_fn = estimator.predict_proba 
            label = 1 
        else:
            predict_fn = estimator.predict 
            label = 0 
        
        n_examples = X.shape[0]
        contributions = np.zeros(X.shape)
        bias = np.zeros((n_examples))
        
        X_values = X.values 
    
        # With the FAST-Lime code, parallelization is unneccsary. 
        if self._n_jobs == -1:
            parallel = Parallel(n_jobs=self._n_jobs, backend='loky')
            with tqdm_joblib(tqdm(desc="LIME", total=n_examples)) as progress_bar:
                results = parallel(delayed(self._explain_lime)(explainer, predict_fn, X_values[i,:], label) 
                       for i in range(n_examples))
       
            for j, (contrib, b) in enumerate(results):
                contributions[j,:] = contrib 
                bias[j] = b
        else:
            for i in tqdm(range(n_examples), desc='LIME'):
                contrib, b = self._explain_lime(explainer, predict_fn, X.values[i,:], label)
                contributions[i,:] = contrib
                bias[i] = b
        
        return contributions, bias
        
    def _get_feature_contributions(self, estimator, X, n_jobs=1, shap_kws=None, 
                                   lime_kws=None,  method=None, estimator_output=None):
        """
        FOR INTERNAL PURPOSES ONLY.

        Compute the feature contributions either using treeinterpreter or SHAP.

        Parameters
        ------------
        estimator : callable
        X : pandas.DataFrame of shape (n_samples, n_features)

        """
        self._n_jobs=n_jobs
        
        if method is not None:
            self.method = method
        
        if estimator_output is not None:
            self.estimator_output = estimator_output
        
        if self.method == "shap":
            contributions, bias = self._get_shap_values(estimator, X, shap_kws)
        elif self.method == "tree_interpreter":
            contributions, bias = self._get_ti_values(estimator, X)
        elif self.method == 'lime':
            contributions, bias = self._get_lime_values(estimator, X, lime_kws)

        n_samples = len(X)
        columns = self.feature_names + ["Bias"]

        # A single example.
        if isinstance(bias, float):
            bias = [bias]

        bias_reshaped = np.reshape(bias, (len(bias), 1))
        data = np.concatenate((contributions, bias_reshaped), axis=1)

        return pd.DataFrame(data, columns=columns)

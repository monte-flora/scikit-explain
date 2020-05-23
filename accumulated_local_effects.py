import numpy as np
import pandas as pd
import concurrent.futures

from utils import compute_bootstrap_samples

class AccumulatedLocalEffects:
    """
    Class for computing accumulated local effect.

    Args:
        model : a trained single scikit-learn model, or list of scikit-learn models, or
            dictionary of models where the key is a generic name and the value
            is a train model.
        examples : pandas DataFrame or ndnumpy array. If ndnumpy array, make sure
            to specify the feature names
        targets: list or numpy array of targets/labels. List converted to numpy array
        classification: defaults to True for classification problems.
            Set to false otherwise.
        feature_names : defaults to None. Should only be set if examples is a
            nd.numpy array. Make sure it's a list
    """

    def __init__(self, model=None, examples=None, targets=None, classification=True,
            feature_names=None):

        # if model is of type list or single objection, convert to dictionary
        if not isinstance(model, dict):
            if isinstance(model, list):
                self._models = {type(m).__name__ : m for m in model}
            else:
                self._models = {type(model).__name__ : model}
        # user provided a dict
        else:
            self._models = model

        self._examples = examples

        # check that targets are assigned correctly
        if isinstance(targets, list):
            self._targets = np.array(targets)
        elif isinstance(targets, np.ndarray):
            self._targets = targets
        else:
            raise TypeError('Target variable must be numpy array.')

        # make sure data is the form of a pandas dataframe regardless of input type
        if isinstance(self._examples, np.ndarray):
            if (feature_names is None):
                raise Exception('Feature names must be specified if using NumPy array.')
            else:
                self._feature_names = feature_names
                self._examples      = pd.DataFrame(data=examples, columns=feature_names)
        else:
            self._feature_names  = examples.columns.to_list()

        self._classification = classification
        self._ale            = None
        self._x1vals         = None
        self._x2vals         = None
        self._hist_vals      = None

        # dictionary containing information for all each feature and model
        self._dict_out = {}

    def get_final_dict(self):

        return self._dict_out

    def run_ale(self, features=None, njobs=None, subsample=1.0, nbootstrap=1, **kwargs):

        """
            Runs the accumulated local effect calculation and returns a dictionary with all
            necessary inputs for plotting.

            feature: List of strings for first-order partial dependence, or list of tuples
                     for second-order
            subsample: a float (between 0-1) for fraction of examples used in bootstrap
            nbootstrap: number of bootstrap iterations to perform. Defaults to 1 (no
                        bootstrapping).
        """


        self.subsample  = subsample
        self.nbootstrap = nbootstrap

        # get number of features we are processing
        n_feats = len(features)

        # check first element of feature and see if of type tuple; assume second-order calculations
        if isinstance(features[0], tuple):

            with concurrent.futures.ProcessPoolExecutor(max_workers=njobs) as executor:
                tdict = executor.map(self._parallelize_2d, features)

            #convert list of dicts to dict
            for elem in tdict:
                self._dict_out.update(elem)

        # else, single order calculations
        else:

            with concurrent.futures.ProcessPoolExecutor(max_workers=njobs) as executor:
                tdict = executor.map(self._parallelize_1d, features)

            #convert list of dicts to dict
            for elem in tdict:
                self._dict_out.update(elem)

    def _parallelize_1d(self, feature):

        temp_dict = {}
        temp_dict[feature] = {}

        print(f"Processing feature {feature}...")

        for model_name, model in self._models.items():

            temp_dict[feature][model_name] = {}

            self.calculate_first_order_ale(feature=feature,
                                            model=model,
                                            subsample=self.subsample,
                                            nbootstrap=self.nbootstrap)

            # add to a dict
            temp_dict[feature][model_name]['values']    = self._ale
            temp_dict[feature][model_name]['xdata1']    = 0.5 * (self._x1vals[1:] + self._x1vals[:-1])
            temp_dict[feature][model_name]['hist_data'] = self._hist_vals

        return temp_dict

    def _parallelize_2d(self, feature):

        temp_dict = {}
        temp_dict[feature] = {}

        print(f"Processing feature {feature}...")

        for model_name, model in self._models.items():

            #print(f"Processing model {model}...")

            temp_dict[feature][model_name] = {}

            self.compute_second_order_ale(feature=feature,
                                          model=model,
                                          subsample=self.subsample,
                                          nbootstrap=self.nbootstrap)

            #print(self._pdp_values)

            # add to a dict
            temp_dict[feature][model_name]['values'] = self._ale
            temp_dict[feature][model_name]['xdata1'] = 0.5 * (self._x1vals[1:] + self._x1vals[:-1])
            temp_dict[feature][model_name]['xdata2'] = 0.5 * (self._x2vals[1:] + self._x2vals[:-1])

        return temp_dict


    def calculate_first_order_ale(self, feature=None, **kwargs):

        """
            Computes first-order ALE function on single continuous feature data.

            Parameters
            ----------
            feature : string
                The name of the feature to consider.
            xdata : array
                Quantiles of feature.
        """

        model      = kwargs.get('model', "")
        subsample  = kwargs.get('subsample', 1.0)
        nbootstrap = kwargs.get('nbootstrap', 1)

        # make sure feature is set
        if feature is None:
            raise Exception("Specify a feature.")

        # Find the ranges to calculate the local effects over
        # Using xdata ensures each bin gets the same number of examples
        self._x1vals = np.percentile(self._examples[feature].values, np.arange(2.5, 97.5 + 5, 5))

        # get data in numpy format
        column_of_data = self._examples[feature].to_numpy()

        # append examples for histogram use
        self._hist_vals = column_of_data

        # get the bootstrap samples
        if nbootstrap > 1:
            bootstrap_examples = compute_bootstrap_samples(self._examples,
                                        subsample=subsample,
                                        nbootstrap=nbootstrap)
        else:
            bootstrap_examples = [self._examples.index.to_list()]

        # define ALE array
        self._ale = np.zeros((nbootstrap, self._x1vals.shape[0]-1))

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_examples):

            # get samples
            examples = self._examples.iloc[idx, :]

            # loop over all ranges
            for i in range(1, self._x1vals.shape[0]):

                # get subset of data
                df_subset = examples[(examples[feature] >= self._x1vals[i - 1]) &
                                     (examples[feature] < self._x1vals[i])]

                # Without any observation, local effect on splitted area is null
                if len(df_subset) != 0:
                    lower_bound = df_subset.copy()
                    upper_bound = df_subset.copy()

                    lower_bound[feature] = self._x1vals[i - 1]
                    upper_bound[feature] = self._x1vals[i]

                    upper_bound = upper_bound.values
                    lower_bound = lower_bound.values

                    if self._classification:
                        effect = 100.0 * ( model.predict_proba(upper_bound)[:, 1]
                                        -  model.predict_proba(lower_bound)[:, 1] )
                    else:
                        effect = model.predict(upper_bound) - model.predict(lower_bound)

                    self._ale[k,i - 1] = np.mean(effect)

            # The accumulated effect
            self._ale[k,:] = self._ale[k,:].cumsum()
            mean_ale       = self._ale[k,:].mean()

            # Now we have to center ALE function in order to obtain null expectation for ALE function
            self._ale[k,:] -= mean_ale


    def calculate_second_order_ale(self, feature=None, **kwargs):
        """
            Computes second-order ALE function on two continuous features data.

            Parameters
            ----------
            feature : string
                The name of the feature to consider.
            xdata : array
                Quantiles of feature.
        """
        # make sure there are two features...
        assert(len(feature) == 2), "Size of features must be equal to 2."

        # check to make sure both features are valid
        if (feature[0] not in self._feature_names):
            raise TypeError(f'Feature {features[0]} is not a valid feature')

        if (feature[1] not in self._feature_names):
            raise TypeError(f'Feature {features[1]} is not a valid feature')

        # create bins for computation for both features
        if self._x1vals is None:
            self._x1vals = np.percentile(self._examples[feature[0]].values, np.arange(2.5, 97.5 + 5, 5))

        if self._x2vals is None:
            self._x2vals = np.percentile(self._examples[feature[1]].values, np.arange(2.5, 97.5 + 5, 5))

        # get the bootstrap samples
        if nbootstrap > 1:
            bootstrap_examples = compute_bootstrap_samples(self._examples,
                                        subsample=subsample,
                                        nbootstrap=nbootstrap)
        else:
            bootstrap_examples = [self._examples.index.to_list()]

        # define ALE array as 3D
        self._ale = np.zeros((nbootstrap, self._x1vals.shape[1] - 1, self._x1vals.shape[1] - 1))

        # for each bootstrap set
        for k, idx in enumerate(bootstrap_examples):

            # get samples
            examples = self._examples.iloc[idx, :]

            # compute calculation over 2-d space
            for i in range(1, self._x1vals.shape[0]):
                for j in range(1, self._x1vals.shape[1]):

                    # Select subset of training data that falls within subset
                    df_subset = examples[ (examples[features[0]] >= self._x1vals[i - 1])
                                        & (examples[features[0]] <  self._x1vals[i])
                                        & (examples[features[1]] >= self._x2vals[j - 1])
                                        & (examples[features[1]] <  self._x2vals[j]) ]

                    # Without any observation, local effect on splitted area is null
                    if len(df_subset) != 0:

                        # get lower and upper bounds on accumulated grid
                        z_low = [df_subset.copy() for _ in range(2)]
                        z_up =  [df_subset.copy() for _ in range(2)]

                        # The main ALE idea that compute prediction difference between
                        # same data except feature's one
                        z_low[0][features[0]] = self._x1vals[i - 1]
                        z_low[0][features[1]] = self._x2vals[j - 1]
                        z_low[1][features[0]] = self._x1vals[i]
                        z_low[1][features[1]] = self._x2vals[j - 1]
                        z_up[0][features[0]]  = self._x1vals[i - 1]
                        z_up[0][features[1]]  = self._x2vals[j]
                        z_up[1][features[0]]  = self._x1vals[i]
                        z_up[1][features[1]]  = self._x2vals[j]

                        if self._classification is True:
                            effect = 100.0 * ( (model.predict_proba(z_up[1])[:, 1]
                                              - model.predict_proba(z_up[0])[:, 1])
                                             - (model.predict_proba(z_low[1])[:, 1]
                                              - model.predict_proba(z_low[0])[:, 1]) )
                        else:
                            effect = (model.predict(z_up[1]) - model.predict(z_up[0])
                                   - (model.predict(z_low[1]) - model.predict(z_low[0])))

                        self._ale[i - 1, j - 1] = np.mean(effect)

            # The accumulated effect
            self._ale[k,:,:] = np.cumsum(self._ale, axis=0)

            # Now we have to center ALE function in order to obtain null expectation for ALE function
            self._ale[k,:,:] -= self._ale.mean()

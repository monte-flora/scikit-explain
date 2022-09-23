####################################################################################
# Based on the Faster-LIME package (https://github.com/seansaito/Faster-LIME) 
# by author seansaito (https://github.com/seansaito)
# Slight modifications have been made to make it compatiable with scikit-explain.
####################################################################################

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OneHotEncoder

from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler

from ..common.utils import ridge_solve, kernel_fn, discretize, dict_disc_to_bin

class BaseTabularExplainer(ABC):

    def __init__(self, training_data, feature_names=None,
                 categorical_names=None, 
                 discretizer='quartile', **kwargs):
        """
        Args:
            training_data (np.ndarray): Training data to measure training data statistics
            feature_names (list): List of feature names
            categorical_feature_idxes (list): List of idxes of features that are categorical
            discretizer (str): Discretization resolution

        Assumptions:
            * Data only contains categorical and/or numerical data
            * Categorical data is already converted to ordinal labels (e.g. via scikit-learn's
                OrdinalEncoder)

        """
        self.training_data = training_data
        self.num_features = self.training_data.shape[1]

        # Parse columns
        if feature_names is not None:
            # TODO input validation
            self.feature_names = list(feature_names)
        else:
            self.feature_names = list(range(self.num_features))
            
        categorical_feature_idxes = [feature_names.index(f) for f in categorical_names]
            
        self.categorical_feature_idxes = categorical_feature_idxes
        if self.categorical_feature_idxes:
            self.categorical_features = [self.feature_names[i] for i in
                                         self.categorical_feature_idxes]
            self.numerical_features = [f for f in self.feature_names if
                                       f not in self.categorical_features]
            self.numerical_feature_idxes = [idx for idx in range(self.num_features) if
                                            idx not in self.categorical_feature_idxes]
        else:
            self.categorical_features = []
            self.numerical_features = self.feature_names
            self.numerical_feature_idxes = list(range(self.num_features))

        # Some book-keeping: keep track of the original indices of each feature
        self.dict_num_feature_to_idx = {feature: idx for (idx, feature) in
                                        enumerate(self.numerical_features)}
        self.dict_feature_to_idx = {feature: idx for (idx, feature) in
                                    enumerate(self.feature_names)}
        self.list_reorder = [self.dict_feature_to_idx[feature] for feature in
                             self.numerical_features + self.categorical_features]

        # Get training data statistics
        # Numerical feature statistics
        if self.numerical_features:
            training_data_num = self.training_data[:, self.numerical_feature_idxes]
            self.sc = StandardScaler(with_mean=False)
            self.sc.fit(training_data_num)
            self.percentiles = dict_disc_to_bin[discretizer]
            self.all_bins_num = np.percentile(training_data_num, self.percentiles, axis=0).T

        # Categorical feature statistics
        if self.categorical_features:
            training_data_cat = self.training_data[:, self.categorical_feature_idxes]
            training_data_cat = training_data_cat.astype(int)
            
            self.dict_categorical_hist = {
                feature: np.bincount(training_data_cat[:, idx]) / self.training_data.shape[0] for
                (idx, feature) in enumerate(self.categorical_features)
            }

        # Another mapping from feature to type
        self.dict_feature_to_type = {
            feature: 'categorical' if feature in self.categorical_features else 'numerical' for
            feature in self.feature_names}

    @abstractmethod
    def explain_instance(self, **kwargs):
        raise NotImplementedError



class FastLimeTabularExplainer(BaseTabularExplainer):
    """
    A basic tabular explainer
    """
    def explain_instance(self, data_row, predict_fn, label=0, num_samples=5000, num_features=10,
                         kernel_width=None, **kwargs):
        """
        Explain a prediction on a given instance

        Args:
            data_row (np.ndarray): Data instance to explain
            predict_fn (func): A function which provides predictions from the target model
            label (int): The class to explain
            num_samples (int): Number of synthetic samples to generate
            num_features (int): Number of top features to return
            kernel_width (Optional[float]): Width of the Gaussian kernel when weighting synthetic samples

        Returns:
            (list) Tuples of feature and score, sorted by the score
        """
        # Scale the data
        data_row = data_row.reshape((1, -1))

        # Split data into numerical and categorical data and process
        list_orig = []
        list_disc = []
        if self.numerical_features:
            data_num = data_row[:, self.numerical_feature_idxes]
            data_num = self.sc.transform(data_num)
            data_synthetic_num = np.tile(data_num, (num_samples, 1))
            # Add noise
            data_synthetic_num = data_synthetic_num + np.random.normal(
                size=(num_samples, data_num.shape[1]))
            data_synthetic_num[0] = data_num.ravel()
            # Convert back to original domain
            data_synthetic_num_original = self.sc.inverse_transform(data_synthetic_num)
            # Discretize
            data_synthetic_num_disc, _ = discretize(data_synthetic_num_original, self.percentiles,
                                                    self.all_bins_num)
            list_disc.append(data_synthetic_num_disc)
            list_orig.append(data_synthetic_num_original)

        if self.categorical_features:
            # Sample from training distribution for each categorical feature
            data_cat = data_row[:, self.categorical_feature_idxes]
            list_buf = []
            for feature in self.categorical_features:
                list_buf.append(np.random.choice(a=len(self.dict_categorical_hist[feature]),
                                                 size=(1, num_samples),
                                                 p=self.dict_categorical_hist[feature]))
            data_cat_original = data_cat_disc = np.concatenate(list_buf).T
            data_cat_original[0] = data_cat.ravel()
            data_cat_disc[0] = data_cat.ravel()
            list_disc.append(data_cat_disc)
            list_orig.append(data_cat_original)

        # Concatenate the data and reorder the columns
        data_synthetic_original = np.concatenate(list_orig, axis=1)
        data_synthetic_disc = np.concatenate(list_disc, axis=1)
        data_synthetic_original = data_synthetic_original[:, self.list_reorder]
        data_synthetic_disc = data_synthetic_disc[:, self.list_reorder]

        # Get model predictions (i.e. groundtruth)
        model_pred = predict_fn(data_synthetic_original)
        
        # For classification tasks.
        if np.ndim(model_pred)==2:
            model_pred=model_pred[:,label]


        # Get distances between original sample and neighbors
        if self.numerical_features:
            distances = cdist(data_synthetic_num[:1], data_synthetic_num).reshape(-1, 1)
        else:
            distances = cdist(data_synthetic_disc[:1], data_synthetic_disc).reshape(-1, 1)

        # Weight distances according to some kernel (e.g. Gaussian)
        if kernel_width is None:
            kernel_width = np.sqrt(data_row.shape[1]) * 0.75
        weights = kernel_fn(distances, kernel_width=kernel_width).ravel()

        # Turn discretized data into onehot
        data_synthetic_onehot = OneHotEncoder().fit_transform(data_synthetic_disc)

        # Solve
        tup = (data_synthetic_onehot, model_pred, weights)
        importances, bias = ridge_solve(tup)
        #print(importances.shape, bias.shape)
        
        #explanations = sorted(list(zip(self.feature_names, importances)),
        #                      key=lambda x: x[1], reverse=True)[:num_features]

        return importances, bias 
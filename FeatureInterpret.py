import numpy as np
import pandas as pd
from treeinterpreter import treeinterpreter as ti

list_of_acceptable_tree_models = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "DecisionTree",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
]


def indexs_based_on_performance(the_probabilities, targets, num_indices=10):
    """
    Determines the best 'hits' (forecast probabilties closest to 1)
    or false alarms (forecast probabilities furthest from 0 )
    or misses (forecast probabilties furthest from 1 )

    Note: This routine assumes you have already computed probabilities

    Args:
    ------------------
     model, sklearn RandomForestClassifier object
     examples, pandas dataframe of validation examples
     targets, a numpy array
     num_indices, 
    """
    # get indices for each binary class
    positive_idx = np.where(targets > 0)
    negative_idx = np.where(targets < 1)

    # get targets for each binary class
    positive_class = targets[positive_idx[0]]
    negative_class = targets[negative_idx[0]]

    # compute forecast probabilities for each binary class
    forecast_probabilities_on_pos_class = the_probabilities[positive_idx[0]]
    forecast_probabilities_on_neg_class = the_probabilities[negative_idx[0]]

    # compute the absolute difference
    diff_from_pos = abs(positive_class - forecast_probabilities_on_pos_class)
    diff_from_neg = abs(negative_class - forecast_probabilities_on_neg_class)

    # sort based on difference and store in array
    sorted_diff_for_hits = np.array(
        list(sorted(zip(diff_from_pos, positive_idx), key=lambda x: x[0]))
    )
    sorted_diff_for_misses = np.array(
        list(sorted(zip(diff_from_pos, positive_idx), key=lambda x: x[0], reverse=True))
    )
    sorted_diff_for_false_alarms = np.array(
        list(sorted(zip(diff_from_neg, negative_idx), key=lambda x: x[0], reverse=True))
    )

    # store all resulting indicies in one dictionary
    adict = {
        "hits": [sorted_diff_for_hits[i][1] for i in range(num_indices + 1)],
        "false_alarms": [
            sorted_diff_for_false_alarms[i][1] for i in range(num_indices + 1)
        ],
        "misses": [sorted_diff_for_misses[i][1] for i in range(num_indices + 1)],
    }

    # Converting the np.arrays to interger type so they
    # be used as indices
    for key in list(adict.keys()):
        adict[key] = np.array(adict[key]).astype(int)

    return adict


def interpert_RF(model, df_of_examples, feature_names=None):

    """
    Method for intrepreting a random forest prediction for a set of examples

    Args:
    --------------------
    model, sklearn RandomForestClassifier object
    examples, pandas dataframe of validation examples
    feature_names, list of features names corresponding to the columns of examples
    
    Return:
        pandas dataframe
    """
    # check to make sure model is of type Tree
    if type(model).__name__ not in list_of_acceptable_tree_models:
        raise Exception(f"{model_name} model is not accepted for this method.")

    # get columns from pandas DataFrame object
    feature_names = df_of_examples.columns

    # number of examples
    n_examples = df_of_examples.shape[0]

    print(f"Interpreting {n_examples} examples...")
    prediction, bias, contributions = ti.predict(model, df_of_examples)
    forecast_probabilities = model.predict_proba(df_of_examples)[:, 1] * 100.0
    positive_class_contributions = contributions[:, :, 1]

    tmp_data = []

    # loop over each case appending and append each feature and value to a dictionary
    for i in range(n_examples):

        key_list = []
        var_list = []

        for c, feature in zip(positive_class_contributions[i, :, 1], feature_names):

            key_list.append(feature)
            var_list.append(round(100.0 * c, 2))

        tmp_data.append(dict(zip(klist, vlist)))

    # return a pandas DataFrame to do analysis on
    contributions_dataframe = pd.DataFrame(data=tmp_data)

    return contributions_dataframe


def getStats(in_dataframe, n_best=10):

    """
    Routine to provide stats from tree interpreter. Needs work...
    """

    mean = in_dataframe.mean(axis=0).head(n_best)
    max = in_dataframe.max(axis=0).head(n_best)
    min = in_dataframe.min(axis=0).head(n_best)
    std = in_dataframe.std(skipna=True, ddof=1, axis=0).head(n_best)

    return mean, max, min, std


def compute_1d_partial_dependence(df_in, model, feature, **kwargs):
    """
    Calculate the partial dependence.
    # Friedman, J., 2001: Greedy function approximation: a gradient boosting machine.Annals of Statistics,29 (5), 1189–1232.
    ##########################################################################
    Partial dependence plots fix a value for one or more predictors
    # for examples, passing these new data through a trained model, 
    # and then averaging the resulting predictions. After repeating this process
    # for a range of values of X*, regions of non-zero slope indicates that
    # where the ML model is sensitive to X* (McGovern et al. 2019). Only disadvantage is
    # that PDP do not account for non-linear interactions between X and the other predictors.
    #########################################################################
    
    Args: 
        df: pandas dataframe of validation examples
        model: sklearn or similar model object with a predict_proba method
        feature: str of the feature to be evaluated
        kwargs: a dictionary contain 'mean' and 'std' to calculate the variable_range in normalized space 
                (functionality not currently included)
    Returns:
            pdp_values, numpy.array of partial dependence variables (size = len(variable_range))
            variable_range, numpy.array of values used in calculating the partial dependence
            all_values, all examples of feature (useful for plotting a histogram in the PDP plots)
    
    """

    # get data in numpy format
    column_of_data = df_in[feature].to_numpy()

    # define bins based on 10th and 90th percentiles
    variable_range = np.linspace(
        np.percentile(column_of_data, 10), np.percentile(column_of_data, 90), num=20
    )

    # define output array to store partial dependence values
    pdp_values = np.fill(variable_range.shape[0], np.nan)

    # for each value, set all indices to the value, make prediction, store mean prediction
    for i, value in enumerate(variable_range):

        copy_df = df_in.copy()
        copy_df.loc[:, feature] = value

        predictions = model.predict_proba(copy_df)[:, 1]
        pdp_values[i] = np.mean(predictions)

    return pdp_values, variable_range


def compute_2d_partial_dependence(df, model, features, **kwargs):

    """
    Calculate the partial dependence.
    # Friedman, J., 2001: Greedy function approximation: a gradient boosting machine.Annals of Statistics,29 (5), 1189–1232.
    ##########################################################################
    Partial dependence plots fix a value for one or more predictors
    # for examples, passing these new data through a trained model, 
    # and then averaging the resulting predictions. After repeating this process
    # for a range of values of X*, regions of non-zero slope indicates that
    # where the ML model is sensitive to X* (McGovern et al. 2019). Only disadvantage is
    # that PDP do not account for non-linear interactions between X and the other predictors.
    #######################################################################
     Args: 
        df: pandas dataframe of validation examples
        model: sklearn or similar model object with a predict_proba method
        features: 2-tuple of features to be evaluated
        
    Returns:
            pdp_values, 2D numpy.array of partial dependence variables (size = len(variable_range))
            var1_range, numpy.array of values used in calculating the partial dependence for the first feature
            var2_range, "..." for the second feature
    """

    # get data for both features
    values_for_var1 = df[features[0]].to_numpy()
    values_for_var2 = df[features[1]].to_numpy()

    # get ranges of data for both features
    var1_range = np.linspace(
        np.percentile(values_for_var1, 10), np.percentile(values_for_var1, 90), num=20
    )
    var2_range = np.linspace(
        np.percentile(values_for_var2, 10), np.percentile(values_for_var2, 90), num=20
    )

    # define 2-D grid
    pdp_values = np.fill((var1_range.shape[0], var2_range.shape[0]), np.nan)

    # similar concept as 1-D, but for 2-D
    for i, value1 in enumerate(var1_range):
        for k, value2 in enumerate(var2_range):
            copy_df = df.copy()
            copy_df.loc[features[0]] = value1
            copy_df.loc[features[1]] = value2
            predictions = model.predict_proba(copy_df)[:, 1]
            pdp_values[i, k] = np.mean(predictions)

    return pdp_values, var1_range, var2_range


def calculate_first_order_ALE(data, feature, quantiles=None, classification=True):

    """Computes first-order ALE function on single continuous feature data.

    Parameters
    ----------
    predictor : function
        A prediction function. In scikit-learn, can be predict, predict_proba, etc. 
        (default is predict_proba)
    train_set : pandas DataFrame
        Training set on which model was trained.
    feature : string
        Feature's name.
    quantiles : array-like
        Quantiles of feature.
    """

    if quantiles is None:
        quantiles = np.linspace(
            np.percentile(all_values, 10), np.percentile(all_values, 90), num=20
        )

    # define ALE function
    ALE = np.zeros(len(quantiles) - 1)

    # loop over all ranges
    for i in range(1, len(quantiles)):

        # get subset of data
        subset = train_set[
            (quantiles[i - 1] <= data[feature]) & (data[feature] < quantiles[i])
        ]

        # Without any observation, local effect on splitted area is null
        if len(subset) != 0:
            z_low = subset.copy()
            z_up = subset.copy()

            # The main ALE idea that compute prediction difference between same data except feature's one
            z_low[feature] = quantiles[i - 1]
            z_up[feature] = quantiles[i]

            if classification is True:
                ALE[i - 1] += (
                    model.predict_proba(z_up) - model.predict_proba(z_low)
                ).sum() / subset.shape[0]
            else:
                ALE[i - 1] += (
                    model.predict(z_up) - model.predict(z_low)
                ).sum() / subset.shape[0]

    # The accumulated effect
    ALE = ALE.cumsum()

    # Now we have to center ALE function in order to obtain null expectation for ALE function
    ALE -= ALE.mean()

    return ALE


def calculate_second_order_ALE(predictor, train_set, features, quantiles):

    """Computes second-order ALE function on two continuous features data.
    """

    quantiles = np.asarray(quantiles)
    ALE = np.zeros((quantiles.shape[1], quantiles.shape[1]))  # Final ALE functio
    print(quantiles)
    for i in range(1, len(quantiles[0])):
        for j in range(1, len(quantiles[1])):
            # Select subset of training data that falls within subset
            subset = train_set[
                (quantiles[0, i - 1] <= train_set[features[0]])
                & (quantiles[0, i] > train_set[features[0]])
                & (quantiles[1, j - 1] <= train_set[features[1]])
                & (quantiles[1, j] > train_set[features[1]])
            ]
            # Without any observation, local effect on splitted area is null
            if len(subset) != 0:
                z_low = [
                    subset.copy() for _ in range(2)
                ]  # The lower bounds on accumulated grid
                z_up = [
                    subset.copy() for _ in range(2)
                ]  # The upper bound on accumulated grid
                # The main ALE idea that compute prediction difference between same data except feature's one
                z_low[0][features[0]] = quantiles[0, i - 1]
                z_low[0][features[1]] = quantiles[1, j - 1]
                z_low[1][features[0]] = quantiles[0, i]
                z_low[1][features[1]] = quantiles[1, j - 1]
                z_up[0][features[0]] = quantiles[0, i - 1]
                z_up[0][features[1]] = quantiles[1, j]
                z_up[1][features[0]] = quantiles[0, i]
                z_up[1][features[1]] = quantiles[1, j]

                ALE[i, j] += (
                    predictor(z_up[1])
                    - predictor(z_up[0])
                    - (predictor(z_low[1]) - predictor(z_low[0]))
                ).sum() / subset.shape[0]

    ALE = np.cumsum(ALE, axis=0)  # The accumulated effect
    ALE -= (
        ALE.mean()
    )  # Now we have to center ALE function in order to obtain null expectation for ALE function
    return ALE


def calculate_first_order_ALE_categorical(
    predictor, train_set, feature, features_classes, feature_encoder=None
):

    """Computes first-order ALE function on single categorical feature data.

    Parameters
    ----------
    predictor : function
        Prediction function.
    train_set : pandas DataFrame
        Training set on which model was trained.
    feature : string
        Feature's name.
    features_classes : list or string
        Feature's classes.
    feature_encoder : function or list
        Encoder that was used to encode categorical feature. If features_classes is not None, this parameter is skipped.
    """
    num_cat = len(features_classes)
    ALE = np.zeros(num_cat)  # Final ALE function

    for i in range(num_cat):
        subset = train_set[train_set[feature] == features_classes[i]]

        # Without any observation, local effect on splitted area is null
        if len(subset) != 0:
            z_low = subset.copy()
            z_up = subset.copy()
            # The main ALE idea that compute prediction difference between same data except feature's one
            z_low[feature] = quantiles[i - 1]
            z_up[feature] = quantiles[i]
            ALE[i] += (
                predictor(z_up)[:, 1] - predictor(z_low)[:, 1]
            ).sum() / subset.shape[0]

    ALE = ALE.cumsum()  # The accumulated effect
    ALE -= (
        ALE.mean()
    )  # Now we have to center ALE function in order to obtain null expectation for ALE function
    return ALE

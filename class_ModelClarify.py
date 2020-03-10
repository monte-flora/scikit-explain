import numpy as np
import pandas as pd
from PermutationImportance.permutation_importance import sklearn_permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from treeinterpreter import treeinterpreter as ti

list_of_acceptable_tree_models = ['RandomForestClassifier', 'RandomForestRegressor',
    'DecisionTreeClassifier', 'ExtraTreesClassifier', 'ExtraTreesRegressor']

class ModelClarify():

    '''
    Class for computing various ML model interpretations...blah blah blah

    Args:
        model : a scikit-learn model
        examples_in : pandas DataFrame or ndnumpy array. If ndnumpy array, make sure
            to specify the feature names
        targets_in: numpy array of targets/labels
        classification: defaults to True for classification problems. 
            Set to false otherwise.
        feature_names : defaults to None. Should only be set if examples_in is a 
            nd.numpy array. Make sure it's a list
    '''

    def __init__(self, model, examples_in, targets_in, classification=True, 
            feature_names=None):

        self._model    = model
        self._examples = examples_in
        self._targets  = targets_in

        if isinstance(self._examples, np.ndarray): 
            self._feature_names  = feature_names
        else:
            self._feature_names  = examples_in.columns.to_list()

        self._classification = classification

    def get_indices_based_on_performance(self, num_indices=10):

        '''
        Determines the best 'hits' (forecast probabilties closest to 1)
        or false alarms (forecast probabilities furthest from 0 )
        or misses (forecast probabilties furthest from 1 )

        The returned dictionary below can be passed into interpert_tree_based_model()

        Args:
        ------------------
            num_indices : Integer representing the number of indices (examples) to return.
                          Default is 10
        '''
        
        if isinstance(self._examples, pd.DataFrame): examples_cp = self._examples.to_numpy()

        #get indices for each binary class
        positive_idx = np.where(self._targets > 0)
        negative_idx = np.where(self._targets < 1)

        #get targets for each binary class
        positive_class = self._targets[positive_idx[0]]
        negative_class = self._targets[negative_idx[0]]    
    
        #compute forecast probabilities for each binary class
        forecast_probabilities_on_pos_class = self._model.predict_proba(examples_cp[positive_idx[0], :])[:,1]
        forecast_probabilities_on_neg_class = self._model.predict_proba(examples_cp[negative_idx[0], :])[:,1]
    
        #compute the absolute difference
        diff_from_pos = abs(positive_class - forecast_probabilities_on_pos_class)
        diff_from_neg = abs(negative_class - forecast_probabilities_on_neg_class)
    
        #sort based on difference and store in array
        sorted_diff_for_hits = np.array( sorted( zip(diff_from_pos, positive_idx[0]), key = lambda x:x[0]))
        sorted_diff_for_misses = np.array( sorted( zip(diff_from_pos, positive_idx[0]), key = lambda x:x[0], reverse=True ))
        sorted_diff_for_false_alarms = np.array( sorted( zip(diff_from_neg, negative_idx[0]), key = lambda x:x[0], reverse=True )) 

        #store all resulting indicies in one dictionary
        adict =  { 
                    'hits': [ sorted_diff_for_hits[i][1] for i in range(num_indices+1) ],
                    'false_alarms': [ sorted_diff_for_false_alarms[i][1] for i in range(num_indices+1) ],
                    'misses': [ sorted_diff_for_misses[i][1] for i in range(num_indices+1) ]
                    } 

        for key in list(adict.keys()):
            adict[key] = np.array(adict[key]).astype(int)

        return adict  

    def tree_interpreter_performance_based(self, performance_dict=None):

        '''
        Method for intrepreting tree based ML models using treeInterpreter. 
        Uses indices from dictionary returned by get_indices_based_on_performance()

        ADD REFERENCE HERE SOMEWHERE

        '''

        # check to make sure model is of type Tree
        if (type(self._model).__name__ not in list_of_acceptable_tree_models):
            raise Exception(f'{model_name} model is not accepted for this method.')                
        
        if (performance_dict is None): 
            performance_dict = self.get_indices_based_on_performance()

        # will be returned; a list of pandas dataframes, one for each performance dict key
        list_of_dfs = []

        for key,values in zip(performance_dict.keys(), performance_dict.values()):

            # number of examples
            n_examples = values.shape[0]

            # get examples for key
            tmp_examples = self._examples.loc[values,:]

            print( f'Interpreting {n_examples} examples from {key}')
            prediction, bias, contributions = ti.predict( self._model, tmp_examples)

            forecast_probabilities = self._model.predict_proba(tmp_examples)[:,1]*100.
            positive_class_contributions = contributions[:,:,1]

            tmp_data = []
    
            #loop over each case appending each feature and value to a dictionary
            for i in range(n_examples):

                key_list = []
                var_list = []

                for c, feature in zip(positive_class_contributions[i,:], self._feature_names):
    
                    key_list.append(feature)
                    var_list.append(round(100.0*c,2))
     
                tmp_data.append(dict(zip(key_list,var_list))) 
    
            #return a pandas DataFrame to do analysis on
            contributions_dataframe = pd.DataFrame(data=tmp_data)

            list_of_dfs.append(contributions_dataframe)

        return list_of_dfs 

    def tree_interpreter_simple(self):

        '''
        Method for intrepreting tree based ML models using treeInterpreter.
        Uses all data passed in to constructor
 
        ADD REFERENCE HERE SOMEWHERE

        '''

        #check to make sure model is of type Tree
        if (type(self._model).__name__ not in list_of_acceptable_tree_models):
            raise Exception(f'{model_name} model is not accepted for this method.')                

        #number of examples
        n_examples = self._examples.shape[0]

        print( f'Interpreting {n_examples} examples...')
        prediction, bias, contributions = ti.predict( self._model, self._examples)

        forecast_probabilities = self._model.predict_proba(self._examples)[:,1]*100.
        positive_class_contributions = contributions[:,:,1]

        tmp_data = []
    
        #loop over each case appending each feature and value to a dictionary
        for i in range(n_examples):

            key_list = []
            var_list = []

            for c, feature in zip(positive_class_contributions[i,:], self._feature_names):
    
                key_list.append(feature)
                var_list.append(round(100.0*c,2))
     
            tmp_data.append(dict(zip(key_list,var_list))) 
    
        #return a pandas DataFrame to do analysis on
        contributions_dataframe = pd.DataFrame(data=tmp_data)

        return contributions_dataframe 

    def compute_1d_partial_dependence(self, feature=None, **kwargs):

        '''
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
            feature : name of feature to compute PD for (string) 
        '''
    
        # check to make sure a feature is present...
        if (feature is None): raise Exception('Specify a feature')

        #check to make sure feature is valid
        if (feature not in self._feature_names): 
            raise Exception(f'Feature {feature} is not a valid feature')

        print("Computing 1-D partial dependence...")

        # get data in numpy format
        column_of_data = self._examples[feature].to_numpy()

        # define bins based on 10th and 90th percentiles
        variable_range = np.linspace(np.percentile(column_of_data, 10), 
                                     np.percentile(column_of_data, 90), num = 20)

        # define output array to store partial dependence values
        pdp_values = np.full(variable_range.shape[0], np.nan)

        # for each value, set all indices to the value, make prediction, store mean prediction
        for i, value in enumerate(variable_range):
        
            copy_df = self._examples.copy()
            copy_df.loc[:,feature] = value

            if (self._classification is True): 
                predictions = self._model.predict_proba( copy_df)[:,1]
            else:
                predictions = self._model.predict( copy_df)
            
            pdp_values[i] = np.mean(predictions)

        return pdp_values, variable_range

    def compute_2d_partial_dependence(self, features, **kwargs):

        '''
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
            feature : tuple of type string of predictor names

        '''
    
        # make sure there are two features...
        if (len(features) > 2): raise Exception(f'tuple of size {len(features)} is greater than 2')
        if (len(features) < 2): raise Exception(f'tuple of size {len(features)} is less than 2')

        # make sure both features are valid...
        if (feature[0] is None or feature[1] is None): 
            raise Exception('One or more features is of type None.')

        #check to make sure feature is valid
        if (feature[0] not in self._feature_names or feature[1] not in self._feature_names): 
            raise Exception(f'Feature {feature} is not a valid feature')


        # get data for both features
        values_for_var1 = self._examples[features[0]].to_numpy()
        values_for_var2 = self._examples[features[1]].to_numpy()

        # get ranges of data for both features
        var1_range = np.linspace(np.percentile(values_for_var1, 10), 
                                 np.percentile(values_for_var1, 90), num = 20 )
        var2_range = np.linspace(np.percentile(values_for_var2, 10), 
                                 np.percentile(values_for_var2, 90), num = 20 )

        # define 2-D grid
        pdp_values = np.full((var1_range.shape[0], var2_range.shape[0]), np.nan)

        # similar concept as 1-D, but for 2-D
        for i, value1 in enumerate(var1_range):
            for k, value2 in enumerate(var2_range):
                copy_df = self._examples.copy()
                copy_df.loc[features[0]] = value1
                copy_df.loc[features[1]] = value2

                if (self._classification is True): 
                    predictions = self._model.predict_proba( copy_df)[:,1]
                else:
                    predictions = self._model.predict( copy_df)

                pdp_values[i,k] = np.mean(predictions)

        return pdp_values, var1_range, var2_range

    def calculate_first_order_ale(self, feature=None, quantiles=None):

        """
            Computes first-order ALE function on single continuous feature data.

            Parameters
            ----------
            feature : string
                The name of the feature to consider.
            quantiles : array
                Quantiles of feature.
        """
        
        #TODO: incorporate the monte carlo aspect into these routines in a clean way...
        nbins = 15
        # make sure feature is set
        if (feature is None): raise Exception('Specify a feature.')

        # convert quantiles to array if list
        if isinstance(quantiles, list): quantiles = np.array(quantiles)

        if (quantiles is None):
            # Find the ranges to calculate the local effects over
            # Since the local effect is a deriative, it best to keep 
            # the bins equally spaced. 
            percentiles = np.percentile(df[feature].values, [5,95])
            quantiles = np.linspace(percentiles[0], percentiles[1], num=nbins)
        
        # define ALE function
        ale = np.zeros(len(quantiles) - 1)

        # loop over all ranges
        for i in range(1, len(quantiles)):
    
            # get subset of data
            df_subset = self._examples[ (self._examples[feature]>= quantiles[i - 1]) & 
                                    (self._examples[feature] < quantiles[i])]

            # Without any observation, local effect on splitted area is null
            if len(subset) != 0:
                lower_bound = df_subset.copy()
                upper_bound = df_subset.copy()
                
                # The main ALE idea that compute prediction difference between same data except feature's one
                lower_bound[feature] = quantiles[i - 1]
                upper_bound[feature]  = quantiles[i]
            
                # The main ALE idea that compute prediction difference between same data except feature's one
                lower_bound[feature] = quantiles[i - 1]
                upper_bound[feature]  = quantiles[i]
            
                if self._classification:
                    effect = 100.*(model.predict_proba(upper_bound)[:,1] - model.predict_proba(lower_bound)[:,1])
                else:
                    effect = model.predict(upper_bound) - model.predict(lower_bound)
              ale[i-1] = np.mean(effect)  
        
        # The accumulated effect      
        ale = ale.cumsum()
        mean_ale = ale.mean()

        # Now we have to center ALE function in order to obtain null expectation for ALE function
        ALE -= mean_ale

        return ALE, mean_ale, quantiles

    def calculate_second_order_ALE(self, feature=None, quantiles=None):

        """
            Computes second-order ALE function on two continuous features data.

            Parameters
            ----------
            feature : string
                The name of the feature to consider.
            quantiles : array
                Quantiles of feature.
        """

        # make sure feature is set
        if (feature is None): raise Exception('Specify a feature.')

        # convert quantiles to array if list
        if isinstance(quantiles, list): quantiles = np.array(quantiles)

        if (quantiles is None):
            quantiles = np.linspace(np.percentile(self._examples,10), 
                                    np.percentile(self._examples,90), num = 20)

        # define ALE function
        ALE = np.zeros((quantiles.shape[1], quantiles.shape[1]))

        for i in range(1, len(quantiles[0])):
            for j in range(1, len(quantiles[1])):
                # Select subset of training data that falls within subset
                subset = train_set[(quantiles[0,i-1] <= self._examples[features[0]]) &
                                   (quantiles[0,i] > self._examples[features[0]]) &
                                   (quantiles[1,j-1] <= self._examples[features[1]]) &
                                   (quantiles[1,j] > self._examples[features[1]])]
                # Without any observation, local effect on splitted area is null
                if (len(subset) != 0):
                    
                    #get lower and upper bounds on accumulated grid
                    z_low = [subset.copy() for _ in range(2)]
                    z_up  = [subset.copy() for _ in range(2)]

                    # The main ALE idea that compute prediction difference between 
                    # same data except feature's one
                    z_low[0][features[0]] = quantiles[0, i - 1]
                    z_low[0][features[1]] = quantiles[1, j - 1]
                    z_low[1][features[0]] = quantiles[0, i]
                    z_low[1][features[1]] = quantiles[1, j - 1]
                    z_up[0][features[0]] = quantiles[0, i - 1]
                    z_up[0][features[1]] = quantiles[1, j]
                    z_up[1][features[0]] = quantiles[0, i]
                    z_up[1][features[1]] = quantiles[1, j]

                    if (self._classification is True):
                        ALE[i,j] += (self._model.predict_proba(z_up[1])[:,1] - 
                                     self._model.predict_proba(z_up[0])[:,1] - 
                                    (self._model.predict_proba(z_low[1])[:,1] -
                                     self._model.predict_proba(z_low[0])[:,1])).sum() / subset.shape[0]

                    else:
                        ALE[i,j] += (self._model.predict(z_up[1]) - 
                                     self._model.predict(z_up[0]) - 
                                    (self._model.predict(z_low[1]) -
                                     self._model.predict(z_low[0]))).sum() / subset.shape[0]

        # The accumulated effect
        ALE = np.cumsum(ALE, axis=0)

        # Now we have to center ALE function in order to obtain null expectation for ALE function
        ALE -= ALE.mean()  

        return ALE, quantiles

    def calculate_first_order_ALE_categorical(self, feature=None, 
                features_classes=None):

        """
            Computes first-order ALE function on single categorical feature data.

            Parameters
            ----------
            feature : string
                The name of the feature to consider.
            features_classes : list or string
                The values the feature can take.
        """

        # make sure feature is set
        if (feature is None): raise Exception('Specify a feature.')

        # get range of values of feature class if not set
        if (feature_classes is None): 
            features_classes = self._examples[feature].unique().to_list()

        num_cat = len(features_classes)
        ALE = np.zeros(num_cat)  # Final ALE function

        for i in range(num_cat):
            subset = self._examples[self._examples[feature] == features_classes[i]]

            # Without any observation, local effect on splitted area is null
            if len(subset) != 0:
                z_low = subset.copy()
                z_up = subset.copy()

                # The main ALE idea that compute prediction difference between same data except feature's one
                z_low[feature] = quantiles[i - 1]
                z_up[feature] = quantiles[i]

                if (self._classification is True):
                    ALE[i] += (self._model.predict_proba(z_up)[:,1] - 
                                   self._model.predict_proba(z_low)[:,1]).sum() / subset.shape[0]
                else:
                    ALE[i] += (self._model.predict(z_up) - 
                                   self._model.predict(z_low)).sum() / subset.shape[0]


        # The accumulated effect
        ALE = np.cumsum(ALE, axis=0)

        # Now we have to center ALE function in order to obtain null expectation for ALE function
        ALE -= ALE.mean()  

        return ALE

    def permutation_importance(
                               self, 
                               n_multipass_vars, 
                               evaluation_fn='auprc', 
                               subsample = 1.0, 
                               njobs=1,
                               nbootstrap = 1 
                               ):
        """
        Perform single or multipass permutation importance using Eli's code.

            Parameters:
            -----------
            n_multipass_vars : integer
                number of variables to calculate the multipass permutation importance for.
            evaluation_fn : string or callable
                evaluation function
            subsample: float 
                value of between 0-1 to subsample examples (useful for speedier results)
            njobs : interger or float
                if integer, interpreted as the number of processors to use for multiprocessing
                if float, interpreted as the fraction of proceesors to use
            nbootstrap: integer
                number of bootstrapp resamples 
        """
        if evaluation_fn.lower() == 'auc':
            evaluation_fn = roc_auc_score
            scoring_strategy = 'argmin_of_mean'
        elif evaluation_fn.lower() == 'auprc':
            evaluation_fn = average_precision_score
            scoring_strategy = 'argmin_of_mean'

        result = sklearn_permutation_importance( model = self._model,
                                                 scoring_data = (self._examples, self._targets),
                                                 evaluation_fn = evaluation_fn,
                                                 variable_names = self._feature_names,
                                                 scoring_strategy = scoring_strategy,
                                                 subsample=subsample,
                                                 nimportant_vars = n_multipass_vars,
                                                 njobs = njobs,
                                                 nbootstrap = nbootstrap)   
        return result 




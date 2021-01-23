import shap
import traceback
from .tree_interpreter import TreeInterpreter
import pandas as pd

from ..common.attributes import Attributes
from ..common.utils import (
       get_indices_based_on_performance, 
     avg_and_sort_contributions, 
     retrieve_important_vars,
     brier_skill_score)

list_of_acceptable_tree_models = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
]

class LocalInterpret(Attributes):
    
    """
    LocalInterpret incorporates important methods for explaining local model behavior 
    for select examples. The calculations are primarily based on SHAP (source), but also 
    includes treeinterpreter (source) for random forests and for other select tree-based methods in 
    scikit-learn (see list_of_acceptable_tree_models). 
    
    Attributes:
        model : pre-fit scikit-learn model object, or list of 
            Provide a list of model objects to compute the global methods for.
        
        model_names : str, list 
            List of model names for the model objects in model. 
            For internal and plotting purposes. 
            
        examples : pandas.DataFrame or ndnumpy.array. 
            Examples to explain. Typically, one should use the training dataset for 
            the global methods. 
            
        feature_names : list of strs
            If examples are ndnumpy.array, then provide the feature_names 
            (default is None; assumes examples are pandas.DataFrame).
            
        model_output : 'probability' or 'regression' 
            What is the expected model output. 'probability' uses the positive class of 
            the .predict_proba() method while 'regression' uses .predict(). 
            
        checked_attributes : boolean 
            Ignore this parameter; For internal purposes only
    
    Reference: 
        SHAP
        treeinterpreter
    """
    
    def __init__(self, models, model_names, examples, targets, model_output, 
                 feature_names=None, checked_attributes=False):
        # These functions come from the inherited Attributes class  
        if not checked_attributes:
            self.set_model_attribute(models, model_names)
            self.set_examples_attribute(examples, feature_names)
            self.set_target_attribute(targets)
            self.set_model_output(model_output, models)
        else:
            self.models = models
            self.model_names = model_names
            self.examples = examples
            self.targets = targets
            self.feature_names = list(examples.columns)
            self.model_output = model_output
        
    def _get_local_prediction(self, method='shap', background_dataset=None,
                              performance_based=True, n_examples=100,):
        """
        Explain individual predictions using SHAP (SHapley Additive exPlanations;
        https://github.com/slundberg/shap) or treeinterpreter 
        (https://blog.datadive.net/random-forest-interpretation-with-scikit-learn/)
        
        Args:
            model: scikit-learn tree-based model object 
                see list_of_acceptable_tree_models for examples of models 
            examples : pandas.DataFrame
                data to perform the contribution break-down on.
            method : 'treeinterpreter' or 'shap' 
        """
        if background_dataset is None and method == 'shap':
                raise ValueError("""
                                 background_dataset is None!, but the user set method='shap'.
                                 If using SHAP, then you must provide data to train the explainer.
                                 """
                                )
        
        self.background_dataset = background_dataset
        
        if method not in ['tree_interpreter', 'shap']:
            raise ValueError("""
                             Declared method is not 'tree_interpreter' or 'shap'. 
                             Check for spelling mistake or syntax error!
                             """
                            )
        else:
            self.method = method 

        # will be returned; a list of pandas dataframes, one for each performance dict key
        contributions_dict = {model_name:{} for model_name in self.model_names}
        feature_values_dict = {model_name:{} for model_name in self.model_names}
        
        for model_name, model in self.models.items():
            # create entry for current model
            #self.contributions_dict[model_name] = {}
            if performance_based:
                ### print('Computing performance-based contributions...') 
                performance_dict = get_indices_based_on_performance(model, 
                                                                examples=self.examples, 
                                                                targets=self.targets, 
                                                                n_examples=n_examples,
                                                                 model_output=self.model_output)
            
                for key, indices in performance_dict.items():
                    cont_dict = self._get_feature_contributions(model=model, 
                                                           examples = self.examples.iloc[indices,:], 
                                                          )
                    
                    contributions_dict[model_name][key] = cont_dict
                    feature_values_dict[model_name][key] = self.examples.iloc[indices,:]

            else:
                cont_dict = self._get_feature_contributions(model=model, 
                                                           examples = self.examples, 
                                                          )
                
                contributions_dict[model_name]['non_performance'] = cont_dict
                feature_values_dict[model_name]['non_performance'] = self.examples
                    
            # average out the contributions and sort based on contribution
            avg_contrib_dict, avg_feature_val_dict = avg_and_sort_contributions(
                contributions_dict[model_name], feature_values_dict[model_name])

            contributions_dict[model_name] = avg_contrib_dict
            feature_values_dict[model_name] = avg_feature_val_dict
                    
        return contributions_dict, feature_values_dict
    
    
    def _get_shap_values(self, model, examples,):
        """
        FOR INTERNAL PURPOSES ONLY.
        
        """
        #if subsample_method=='kmeans':
            ### print(f'Performing K-means clustering (K={subsample_size}) to subset the data for the background dataset...')
         #   data_for_shap = shap.kmeans(self.data_for_shap, subsample_size)
        #elif subsample_method=='random':
            ### print(f'Performing random sampling (N={subsample_size}) to subset the data for the background dataset...')
         #   data_for_shap = shap.sample(self.data_for_shap, subsample_size)

        try: 
            print('trying TreeExplainer...')
            explainer = shap.TreeExplainer(model, 
                                       data=self.background_dataset,
                                       feature_perturbation = "interventional",
                                       model_output=self.model_output
                                          )
            contributions = explainer.shap_values(examples)
            
        except Exception as e:
            ### traceback.print_exc()
            if self.model_output == 'probability':
                func = model.predict_proba
            else:
                func = model.predict
                
            print('TreeExplainer failed, starting KernelExplainer...')
            explainer = shap.KernelExplainer(func, 
                                             self.background_dataset, 
                                             link='identity'
                                            )
            
            contributions = explainer.shap_values(examples, l1_reg="num_features(30)")
        
        if self.model_output == 'probability':
            # Neccesary for XGBoost, which only outputs a scalar, not a list like scikit-learn 
            try:
                bias = explainer.expected_value[1]
                contributions = contributions[1]
            except IndexError:
                bias = explainer.expected_value
                contributions = contributions
        else:
            bias = explainer.expected_value
        
        return contributions, bias 
    
    def _get_ti_values(self, model, examples):
        """
         FOR INTERNAL PURPOSES ONLY.
        """
        # check to make sure model is of type Tree
        if type(model).__name__ not in list_of_acceptable_tree_models:
            raise TypeError(f""" Unfortunately, tree interpreter does not work on this type of model :
                                {type(model).__name__}
                            """
                            )
            
        ti = TreeInterpreter(model, examples)
        
        prediction, bias, contributions = ti.predict()

        if self.model_output == 'probability':
            contributions = contributions[:, :, 1]
            bias = bias[0,1]   #bias is all the same values for first index
        else:
            pass
       
        return contributions, bias 
        
    def _get_feature_contributions(self, model, examples):
        """
        FOR INTERNAL PURPOSES ONLY.
        
        Compute the feature contributions either using treeinterpreter or SHAP. 
        
        Args:
            model : 
            examples : 
            subsample_size : 
            model_output
        """
        if self.method == 'shap':
            contributions, bias = self._get_shap_values(model, examples)
        elif self.method == 'tree_interpreter':
            contributions, bias = self._get_ti_values(model, examples)     

        n_examples = len(examples)    

        tmp_data = []
        for i in range(n_examples):
            key_list=[]
            var_list=[]
            for c, feature in zip(contributions[i,:], self.feature_names):
                key_list.append(feature)
                if self.model_output == 'probability':
                    var_list.append(100.*c)
                else:
                    var_list.append(c)
            
            key_list.append('Bias')
            if self.model_output == 'probability':
                var_list.append(100.*bias)
            else:
                var_list.append(bias)

            tmp_data.append(dict(zip(key_list, var_list)))   
                
       # return a pandas DataFrame to do analysis on
        contributions_dataframe = pd.DataFrame(data=tmp_data)
       
        return contributions_dataframe
        

        
        
        


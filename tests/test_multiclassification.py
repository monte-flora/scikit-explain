#===================================================
# Unit test for the multiclassification 
# code in Scikit-Explain.
#===================================================
# Create a multiclassification model from the Iris dataset in Sklearn. 
import sys, os 
sys.path.insert(0, os.path.dirname(os.getcwd()))

import unittest

import skexplain 
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import shap 
from skexplain.plot.base_plotting import PlotStructure
import seaborn as sns


class TestMultiClass(unittest.TestCase):
    def setUp(self):
        # Create a multiclassification model from the Iris dataset in Sklearn. 
        self.X, self.y = load_iris(return_X_y=True, as_frame=True)
        self.lr = LogisticRegression().fit(self.X, self.y)
        self.name = 'LogisticRegression'
        self.explainer = skexplain.ExplainToolkit((self.name, self.lr), X=self.X, y=self.y,)
    
    
    def test_pi(self):
        """Check that the multiclassification permutation importance works"""
        direction = 'backward'
        results = self.explainer.permutation_importance(
                                           n_vars=self.X.shape[1], 
                                           evaluation_fn="rpss",
                                           scoring_strategy='minimize',
                                           n_permute=10, 
                                           subsample=1.0,
                                           n_jobs=2,
                                           verbose=True,
                                           random_seed=42, 
                                           direction=direction,
                                           )
        #Correct rank
        TRUE_RANKINGS = ['sepal width (cm)', 'sepal length (cm)', 'petal width (cm)',
       'petal length (cm)']
        
        np.testing.assert_array_equal(
                results[f"{direction}_singlepass_rankings__{self.name}"].values,
                TRUE_RANKINGS,
            )
    def test_ale(self):
        """Test the multiclass ALE computation and plotting."""
        
        
        ales = [self.explainer.ale(features='all', n_bootstrap=1, 
                                   n_jobs=1, n_bins=20, class_index=class_idx)
            for class_idx in np.unique(self.y)]
        
        features = self.X.columns
        n_panels=len(features)
        plotter= PlotStructure(BASE_FONT_SIZE = 16)
        fig, axes = plotter.create_subplots(n_panels=len(features), n_columns=2, figsize=(8,8), dpi=300, 
                                      wspace=0.4, hspace=0.35)

        colors = list(sns.color_palette("Set2"))
        for ax, feature in zip(axes.flat, features):
            for i, ale in enumerate(ales): 
                self.explainer.plot_ale(ale = ale, 
                           features=feature, 
                           ax=ax, line_kws = {'line_colors' : [colors[i]],
                                              'linewidth': 2.0}, 
                          )
        
        # Add legend 
        plotter.set_legend(n_panels, fig, ax, labels=['Setosa', 'Versicolour', 'Virginica'])
        
    def test_shap(self):
        """Test the multiclass SHAP computation and plotting. """
        results = []
        for class_idx in np.unique(self.y):
            shap_kws={'masker' : shap.maskers.Partition(self.X, max_samples=10, clustering="correlation"), 
           'algorithm' : 'permutation', 'class_idx' : class_idx}

            results.append( self.explainer.local_attributions(method='shap', 
                                       shap_kws=shap_kws,
                                          )
                  )
        
        features = self.X.columns
        n_panels=len(features)
        plotter= PlotStructure(BASE_FONT_SIZE = 16)
        fig, axes = plotter.create_subplots(n_panels=len(features), n_columns=2, figsize=(8,8), dpi=300, 
                                      wspace=0.4, hspace=0.35)

        colors = list(sns.color_palette("Set2"))
        for ax, feature in zip(axes.flat, features):
            for i, shap_vals in enumerate(results): 
                self.explainer.scatter_plot(features=[feature],
                    plot_type = 'dependence',
                    dataset=shap_vals,
                    method = ['shap'], 
                    estimator_name = 'LogisticRegression',
                    color = colors[i],
                    interaction_index=None, 
                    ax=ax, 
                 )
        
        
        # Add legend 
        plotter.set_legend(n_panels, fig, ax, labels=['Setosa', 'Versicolour', 'Virginica'])
        

        
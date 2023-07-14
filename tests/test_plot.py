#===================================================
# Unit test for the plotting 
# code in Scikit-Explain.
#===================================================
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import sys, os 
sys.path.insert(0, os.path.dirname(os.getcwd()))
import skexplain

from tests import TestLR, TestSciKitExplainData, TestRF

# TODO: 
# 1. Check that each plotting script works!
#    To be fair, some of that testing is performed
#    in other test scripts.
# 2. Check the plotting flexibility works
#   - Chaning line colors, figsize, w/hspace, etc. 


class Test1DPlotting(TestLR):
    def test_1d_plot(self):
        # Make sure the plot data is correct.
        feature = "X_1"
        explainer = skexplain.ExplainToolkit(
            estimators=self.lr_estimator, X=self.X, y=self.y
        )
        results = explainer.ale(features=feature, n_bins=30, n_bootstrap=1)
        ydata = results[f"{feature}__{self.lr_estimator_name}__ale"].values[0, :]
        xdata = results[f"{feature}__bin_values"].values

        fig, ax = explainer.plot_ale(ale=results, features=feature)

        ## effect line
        eff_plt_data = ax.lines[0].get_xydata()
        # the x values should be the bins
        np.testing.assert_array_equal(eff_plt_data[:, 0], xdata)
        # the y values should be the effect
        np.testing.assert_array_equal(eff_plt_data[:, 1], ydata)

    def test_ice_and_ale(self):
        
        explainer = skexplain.ExplainToolkit(self.lr_estimator, X=self.X, y=self.y,)

        important_vars = ['X_1', 'X_2']
        
        ale_1d_ds = explainer.ale(features=important_vars, n_bootstrap=1, subsample=0.25, n_jobs=1, n_bins=20)
        ice_ds = explainer.ice(features=important_vars,  subsample=200, n_jobs=1, n_bins=20, random_seed=50)

        fig, axes = explainer.plot_ale(
                               ale=ale_1d_ds,
                                features = important_vars,
                               ice_curves=ice_ds,
                               color_by = 'X_2', 
                                figsize=(10,6),
                                wspace=0.25, 
                                  )
        
        

class Test2DPlotting(TestSciKitExplainData):
    def test_2d_plot(self):
        ### Loading the training data and pre-fit models within the scikit-explain package

        explainer = skexplain.ExplainToolkit(self.estimators, X=self.X, y=self.y,)
        features=[('Feature 2', 'Feature 3'), ('Feature 2', 'Feature 5') ]

        ale_2d_ds = explainer.ale(features=features, 
                                 n_bootstrap=1, 
                                 subsample=0.5,
                                 n_jobs=2,
                                 n_bins=10
                                )
        
        from matplotlib.ticker import MaxNLocator
        cbar_kwargs = {'extend' : 'neither', 'ticks': MaxNLocator(3)}

        fig, axes = explainer.plot_ale(ale=ale_2d_ds,
                                   kde_curves=False,
                                   scatter=False,
                               figsize=(10,8), fontsize=10,
                               cbar_kwargs=cbar_kwargs
                                  ) 
       
    
class TestRankPlots(TestLR):

    # Make sure the ranking is plotted correctly
    def test_rank_plot(self):
        explainer = skexplain.ExplainToolkit(
            estimators=self.lr_estimator, X=self.X, y=self.y
        )
        
        results = explainer.permutation_importance(
            n_vars=5, evaluation_fn="mse", n_permute=10
        )
        
        explainer.plot_importance(data=results, 
                                panels=[('backward_singlepass', self.lr_estimator_name)], 
                                num_vars_to_plot=15, 
                                 )
        
        # Changing the x-labels.
        explainer.plot_importance(data=results, 
                                panels=[('backward_singlepass', self.lr_estimator_name)], 
                                num_vars_to_plot=15,
                                xlabels = ['Single-Pass']  
                                 )
        # Plot single- and multi-pass permutation importance.
        explainer.plot_importance(data=[results]*2, 
                                panels=[('backward_singlepass', self.lr_estimator_name),
                                        ('backward_multipass', self.lr_estimator_name)
                                       ], 
                                num_vars_to_plot=15,
                                 )
        
        # Check error when len(results) != len(panels)
        explainer.plot_importance(data=results, 
                                panels=[('backward_singlepass', self.lr_estimator_name),
                                        ('backward_multipass', self.lr_estimator_name)
                                       ], 
                                num_vars_to_plot=15,
                                 )
        
        # Using feature_colors.
        explainer.plot_importance(
                                data=results, 
                                panels=[('backward_singlepass', self.lr_estimator_name), 
                                       ],
                                num_vars_to_plot=15,
                                feature_colors = 'xkcd:medium green'
                                    )
        
        
        # Plotting connections between correlated features. 
        explainer.plot_importance(
                                data=results, 
                                panels=[('backward_singlepass', self.lr_estimator_name), 
                                       ],
                                num_vars_to_plot=15,
                                feature_colors = 'xkcd:medium green',
                                plot_correlated_features=True, 
                                rho_threshold = 0.01, 
                                    )
        
        # Test Sobol indices 
        sobol_results = explainer.sobol_indices(n_bootstrap=5000, class_index=1)
        explainer.plot_importance(data=[sobol_results], 
                          panels=[('sobol_total', self.lr_estimator[0])],
                          figsize=(12,4)
                         )
        
class TestContributionPlots(TestLR):

    # Make sure the ranking is plotted correctly
    def test_contrib_plot(self):
        # For the LIME, we must provide the training dataset. We also denote any categorical features. 
        lime_kws = {'training_data' : self.X.values}
        explainer = skexplain.ExplainToolkit(
            estimators=self.lr_estimator,
            X=self.X,
            y=self.y,
        )
        
        
        results = explainer.local_attributions(method='lime', lime_kws=lime_kws)
        
        # Create the summary plot. 
        explainer.scatter_plot(
                    plot_type = 'summary',
                    dataset=results,
                    method = 'lime',
                    estimator_name = self.lr_estimator[0],
        )  
        
        # Create the summary plot with a custom axes
        fig, ax = plt.subplots()
        explainer.scatter_plot(
                    plot_type = 'summary',
                    dataset=results,
                    method = 'lime',
                    estimator_name = self.lr_estimator[0],
                    ax=ax
        )  
        
        # Note: If the plots are looking wonky, you can change the figsize. 
        features = ['X_1', 'X_3']
        explainer.scatter_plot(features=features,
                    plot_type = 'dependence',
                    dataset=results,
                    method = 'lime', 
                    estimator_name = self.lr_estimator[0],
                    interaction_index=None, 
        )
        
        


if __name__ == "__main__":
    unittest.main()

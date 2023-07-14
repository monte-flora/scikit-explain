#===================================================
# Unit test for the plotting 
# code in Scikit-Explain.
#===================================================
from os.path import join
import numpy as np
import pandas as pd

import skexplain

from tests import TestLR

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


class Test2DPlotting(TestLR):
    def test_2d_plot(self):
        pass
        ### Loading the training data and pre-fit models within the scikit-explain package
        estimators = skexplain.load_models()
        X,y = skexplain.load_data()

        explainer = skexplain.ExplainToolkit(estimators[:2], X=X, y=y,)
        features=[('temp2m', 'sfc_temp'), ('dwpt2m', 'sfc_temp'), ('dwpt2m', 'sfc_temp') ]

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
        pass


if __name__ == "__main__":
    unittest.main()

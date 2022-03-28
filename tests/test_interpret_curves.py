#===================================================
# Unit test for the ALE and PD 
# code in Scikit-Explain.
#===================================================

import shap
import numpy as np
import skexplain
from skexplain.common.importance_utils import to_skexplain_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import itertools

from tests import TestLR, TestRF, TestSciKitExplainData


class TestALECat(TestSciKitExplainData):
     def test_cat_ale_simple(self):
        # Make sure the categorical ALE is working correctly
        explainer = skexplain.ExplainToolkit(
            estimators=self.estimators[0], X=self.X, y=self.y
        )
        explainer.ale(features='urban') 
        
        
class TestInterpretCurves(TestLR):
    def test_bad_feature_names_exception(self):
        feature = "bad_feature"
        explainer = skexplain.ExplainToolkit(
            estimators=self.lr_estimator, X=self.X, y=self.y
        )
        with self.assertRaises(KeyError) as ex:
            explainer.ale(features=feature)

        except_msg = f"'{feature}' is not a valid feature."
        self.assertEqual(ex.exception.args[0], except_msg)

    def test_too_many_bins(self):
        explainer = skexplain.ExplainToolkit(
            estimators=self.lr_estimator, X=self.X, y=self.y
        )

        n_bins = 100
        with self.assertRaises(ValueError) as ex:
            explainer.ale(
                features=["X_1"],
                subsample=100,
                n_bins=n_bins,
            )
        except_msg = f"""
                                 Broadcast error!
                                 The value of n_bins ({n_bins}) is likely too 
                                 high relative to the sample size of the data. Either increase
                                 the data size (if using subsample) or use less bins. 
                                 """
        self.assertMultiLineEqual(ex.exception.args[0], except_msg)
        

    def test_results_shape(self):
        # Bootstrap has correct shape
        feature = "X_1"
        explainer = skexplain.ExplainToolkit(
            estimators=(self.lr_estimator_name, self.lr), X=self.X, y=self.y
        )
        results = explainer.ale(features=feature, n_bins=10, n_bootstrap=5)
        ydata = results[f"{feature}__{self.lr_estimator_name}__ale"].values

        self.assertEqual(ydata.shape, (5, 10))

        
    def test_xdata(self):
        # Bin values are correct.
        explainer = skexplain.ExplainToolkit(
            estimators=(self.lr_estimator_name, self.lr), X=self.X, y=self.y
        )
        feature = "X_1"
        results = explainer.ale(features=feature, n_bins=5)
        xdata = results[f"{feature}__bin_values"].values

        self.assertCountEqual(
            np.round(xdata, 8),
            [-2.03681146, -0.53458709,  0.01499989,  0.55125525,  2.38096491,],
        )

    def test_ale_simple(self):
        # ALE is correct for a simple case
        # The coefficient of the ALE curves must
        # match that of the actual coefficient.
        explainer = skexplain.ExplainToolkit(
            estimators=(self.lr_estimator_name, self.lr), X=self.X, y=self.y
        )
        feature = "X_1"
        results = explainer.ale(features=feature, n_bins=5)
        lr = LinearRegression()
        lr.fit(
            results[f"{feature}__bin_values"].values.reshape(-1, 1),
            results[f"{feature}__{self.lr_estimator_name}__ale"].values[0, :],
        )

        self.assertAlmostEqual(lr.coef_[0], self.WEIGHTS[0])

    def test_pd_simple(self):
        # PD is correct for a simple case
        # The coefficient of the PD curves must
        # match that of the actual coefficient.
        explainer = skexplain.ExplainToolkit(
            estimators=(self.lr_estimator_name, self.lr), X=self.X, y=self.y
        )
        feature = "X_1"
        results = explainer.pd(features=feature, n_bins=5)
        lr = LinearRegression()
        lr.fit(
            results[f"{feature}__bin_values"].values.reshape(-1, 1),
            results[f"{feature}__{self.lr_estimator_name}__pd"].values[0, :],
        )

        self.assertAlmostEqual(lr.coef_[0], self.WEIGHTS[0])


    def test_ale_complex(self):
        # ALE is correct for a more complex case!
        """ Test that ALE is correct for y = X^2 """
        X = np.linspace(-3, 3, 10000)
        X = X.reshape(-1, 1)
        y = X**2
        rf = RandomForestRegressor()
        rf.fit(X, y)
        
        explainer = skexplain.ExplainToolkit(('RF', rf), X=X, y=y, feature_names=['X_1'])
        ale = explainer.ale(features='all', n_bins=50)
        
        # The idea is that for y = x^2 
        # ALE(x) = x^2 - mean(y) 
        
        val = (ale['X_1__bin_values'].values)**2 - np.mean(y)
        np.testing.assert_allclose(val, ale['X_1__RF__ale'].values[0,:], rtol=0.1)
        

    def test_ice_curves(self):
        """ Test the ICE curves """
        ice_ds = self.explainer_interact.ice(features='all', n_jobs=1)
        ale = self.explainer_interact.ale(features='all')
        self.explainer_interact.plot_ale(ale=ale, ice_curves=ice_ds, )
        
        # Test the color-by functionality. 
        self.explainer_interact.plot_ale(ale=ale, ice_curves=ice_ds, color_by='X_1')
    
        
    def test_2d_ale(self):
        """ Test the 2D ALE """
        # From TestInteractions
        ale_2d = self.explainer_interact.ale(features='all_2d', n_bins=10)
        
        self.explainer_interact.plot_ale(ale_2d) 
        
        # The 2nd order ALE is remaining effect after the 
        # first order effects have been removed. The first order 
        # effects having already had the average effect removed 
        # as that the average(first order effect) == 0. 
        
        # Since the coefficients for each feature is 1, then 
        # the first order effect is just X - mean(self.y). 
        #var1_effect = ale_2d['X_1__bin_values'].values - np.mean(self.y)
        #var2_effect = ale_2d['X_2__bin_values'].values - np.mean(self.y)
        
        #first_order = var1_effect[:, np.newaxis] + var2_effect[np.newaxis, :]
        #xx,yy = np.meshgrid(ale_2d_ds['X_1__bin_values'].values,
        #            ale_2d_ds['X_2__bin_values'].values
        #           )

        #y_pred = self.rf_interact.predict(np.c_[xx.ravel(), yy.ravel()])
        #y_pred = y_pred.reshape(xx.shape)

        # The second order effect is the prediction minus the first order effects. 
        #second_order = y_pred - first_order
        
        #ale_second_order = ale_2d[f'X_1__X_2__{self.rf_estimator_name}__ale'].values
        
        #diff = np.absolute(ale_second_order - second_order) 
        # This should be approximately zero, but is unlikely to be precisely zero.
        #self.assertAlmostEqual(diff, 0., places=1)


    def test_ale_var(self):
        """ Test the 1st and 2nd ALE variance """
        ale_1d = self.explainer_interact.ale(features='all') 
        ale_2d = self.explainer_interact.ale(features='all_2d') 
        
        ale_var_2d = self.explainer_interact.ale_variance(ale=ale_2d,  interaction=True)
        ale_var = self.explainer_interact.ale_variance(ale=ale_1d) 
        
        # Check the dimension!
        with self.assertRaises(Exception) as ex:
            self.explainer_interact.ale_variance(ale=ale_1d,  interaction=True)
            
        except_msg = "ale must be second-order if interaction == True"
        self.assertEqual(ex.exception.args[0], except_msg)    
        

    def test_2d_pd(self):
        """ Test the 2D PD compute and plot."""
        pd_2d = self.explainer_interact.pd(features = 'all_2d')
        self.explainer_interact.plot_pd(pd_2d) 
        
    def test_ias(self):
        """ Test the interaction strength statistics """
        # Test for a linear model IAS ~ 0. 
        explainer = skexplain.ExplainToolkit(
            estimators=(self.lr_estimator_name, self.lr), X=self.X, y=self.y
        )
        
        ale = explainer.ale(features='all') 
        ias = explainer.interaction_strength(ale)[f'{self.lr_estimator_name}_ias'].values[0]
        self.assertAlmostEqual(ias, 0., 4)
        
    def test_hstat(self):
        explainer = skexplain.ExplainToolkit(
            estimators=(self.lr_estimator_name, self.lr), X=self.X, y=self.y
        )
        
        pd_1d = explainer.pd(features='all', n_bins=10) 
        pd_2d = explainer.pd(features='all_2d', n_bins=10,)
        
        # Compute the H-statistic 
        features = list(itertools.combinations(self.X.columns, r=2))
        
        hstat_results= explainer.friedman_h_stat(pd_1d=pd_1d, pd_2d=pd_2d, features=features)
        
    def test_mec(self):
        """ Test the main effect complexity """
        explainer = skexplain.ExplainToolkit(
            estimators=self.lr_estimator, X=self.X, y=self.y
        )
        ale_1d = explainer.ale(features='all')
        mec = explainer.main_effect_complexity(ale=ale_1d)[self.lr_estimator_name]
        self.assertAlmostEqual(mec, 1., 4)


if __name__ == "__main__":
    unittest.main()

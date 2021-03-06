# Unit test for the SHAP code in scikit-explain

import shap
import skexplain
import numpy as np

from tests import TestMultiExampleContributions


class TestSHAP(TestMultiExampleContributions):
    # Simple test to see that the shap contributions + bias = predictions
    """
    The accuracy can vary and thus is not a good test for scikit-explain package.
    def test_compute_shap_equals_prediction(self):
        shap_values = self.results[f"shap_values__{self.rf_estimator_name}"].values
        bias = self.results[f"bias__{self.rf_estimator_name}"].values

        contrib = np.concatenate((shap_values, bias.reshape(len(bias), 1)), axis=1)
        shap_predictions = np.sum(contrib, axis=1)
        predictions = self.rf.predict(self.X_sub)

        np.testing.assert_allclose(predictions, shap_predictions, rtol=0.1)
    """
    def test_compute_shap_with_no_masker(self):
        # Computing SHAP values in scikit-explain.
        explainer = skexplain.ExplainToolkit(
            estimators=self.rf_estimator, X=self.X_sub, y=self.y_sub
        )
        with self.assertRaises(Exception) as ex:
            results = explainer.shap(
                shap_kwargs={"masker": None, "algorithm": "permutation"}
            )

        except_msg = """masker in shap_kwargs is None. 
                             This will cause issues with SHAP. We recommend starting with
                             shap_kwargs = {'masker' = shap.maskers.Partition(X, max_samples=100, clustering="correlation")}
                             where X is the original dataset and not the examples SHAP is being computed for. 
                             """
        self.assertMultiLineEqual(ex.exception.args[0], except_msg)

    def test_plot_shap_summary(self):
        # Just checking that the plot is created!
        self.explainer.plot_shap(
            plot_type="summary",
            shap_values=self.results,
            estimator_name=self.rf_estimator_name,
        )

        # Checking error when estimator name is not declared.
        with self.assertRaises(Exception) as ex:
            self.explainer.plot_shap(
                plot_type="summary",
                shap_values=self.results,
            )

        estimator_name = None
        key = f"shap_values__{estimator_name}"
        except_msg = f"""{key} is not an available variable for this dataset! 
                      Check that SHAP values were compute for estimator : {estimator_name}
                      """
        # Checking that plot_type is correct.
        with self.assertRaises(Exception) as ex:
            self.explainer.plot_shap(
                plot_type="summar",
                shap_values=self.results,
                estimator_name=self.rf_estimator_name,
            )

        except_msg = "Invalid plot_type! Must be 'summary' or 'dependence'"
        self.assertEqual(ex.exception.args[0], except_msg)

    def test_plot_shap_dependence(self):
        features = ["X_1", "X_2"]

        histdata = self.X.copy()
        histdata["target"] = self.y

        # Plot no histdata.
        self.explainer.plot_shap(
            features=features,
            plot_type="dependence",
            shap_values=self.results,
            estimator_name=self.rf_estimator_name,
        )

        # Plot 'auto' interaction index.
        self.explainer.plot_shap(
            features=features,
            plot_type="dependence",
            shap_values=self.results,
            estimator_name=self.rf_estimator_name,
            histdata=histdata,
            interaction_index="auto",
        )

        # Plot interaction index == 'X_3'
        self.explainer.plot_shap(
            features=features,
            plot_type="dependence",
            shap_values=self.results,
            estimator_name=self.rf_estimator_name,
            histdata=histdata,
            interaction_index="X_3",
        )

        # Plot interaction index == None
        self.explainer.plot_shap(
            features=features,
            plot_type="dependence",
            shap_values=self.results,
            estimator_name=self.rf_estimator_name,
            histdata=histdata,
        )

        # Plot interaction index == None, but with target_values
        self.explainer.plot_shap(
            features=features,
            plot_type="dependence",
            shap_values=self.results,
            estimator_name=self.rf_estimator_name,
            histdata=histdata,
            target_values=self.y_sub.values,
            interaction_index=None,
        )


if __name__ == "__main__":
    unittest.main()

<p>
  <img src="https://github.com/monte-flora/py-mint/blob/master/images/mintpy_logo.png?raw=true" align="right" width="400" height="400" />
</p>


<a href="https://app.travis-ci.com/monte-flora/py-mint"><img src="https://app.travis-ci.com/monte-flora/py-mint.svg?branch=master"></a>
[![codecov](https://codecov.io/gh/monte-flora/py-mint/branch/master/graph/badge.svg?token=GG9NRQOZ0N)](https://codecov.io/gh/monte-flora/py-mint)
[![Updates](https://pyup.io/repos/github/monte-flora/py-mint/shield.svg)](https://pyup.io/repos/github/monte-flora/py-mint/)
[![Python 3](https://pyup.io/repos/github/monte-flora/py-mint/python-3-shield.svg)](https://pyup.io/repos/github/monte-flora/py-mint/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI](https://img.shields.io/pypi/v/py-mint)
[![Documentation Status](https://readthedocs.org/projects/py-mint/badge/?version=latest)](https://py-mint.readthedocs.io/en/latest/?badge=latest)


__PyMint__ (__Python-based Model INTerpretations__) is designed to be a user-friendly package for computing and plotting machine learning interpretability/explainability output in Python. Current computation includes
* Feature importance: 
  * Single- and Multi-pass Permutation Importance ([Brieman et al. 2001](https://link.springer.com/article/10.1023/A:1010933404324)], [Lakshmanan et al. 2015](https://journals.ametsoc.org/view/journals/atot/32/6/jtech-d-13-00205_1.xml?rskey=hlSyXu&result=2))
  * SHAP 
  * First-order PD/ALE Variance ([Greenwell et al. 2018](https://arxiv.org/abs/1805.04755))    

* Feature Effects/Attributions: 
  * Partial Dependence (PD), 
  * Accumulated local effects (ALE), 
  * Random forest-based feature contributions (treeinterpreter)
  * SHAP 
  * Main Effect Complexity (MEC; [Molnar et al. 2019](https://arxiv.org/abs/1904.03867))

* Feature Interactions:
  * Second-order PD/ALE 
  * Interaction Strength and Main Effect Complexity (IAS; [Molnar et al. 2019](https://arxiv.org/abs/1904.03867))
  * Second-order PD/ALE Variance ([Greenwell et al. 2018](https://arxiv.org/abs/1805.04755)) 
  * Second-order Permutation Importance ([Oh et al. 2019](https://www.mdpi.com/2076-3417/9/23/5191))
  * Friedman H-statistic ([Friedman and Popescu](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-2/issue-3/Predictive-learning-via-rule-ensembles/10.1214/07-AOAS148.full))

All of these methods are discussed at length in [Christoph Molnar's interpretable ML book](https://christophm.github.io/interpretable-ml-book/). Most calculations can be performed in parallel when multi-core processing is available. The primary feature of this package is the accompanying built-in plotting methods, which are desgined to be easy to use while producing publication-level quality figures. Documentation for PyMint can be found at https://py-mint.readthedocs.io/en/master/. 

The package is under active development and will likely contain bugs or errors. Feel free to raise issues!

This package is largely original code, but also includes snippets or chunks of code from preexisting packages. Our goal is not take credit from other code authors, but to make a single source for computing several machine learning interpretation methods. Here is a list of packages used in PyMint: 
[**PyALE**](https://github.com/DanaJomar/PyALE),
[**PermutationImportance**](https://github.com/gelijergensen/PermutationImportance),
[**ALEPython**](https://github.com/blent-ai/ALEPython),
[**SHAP**](https://github.com/slundberg/shap/), 
[**Scikit-Learn**](https://github.com/scikit-learn/scikit-learn)

If you employ PyMint in your research, please cite this github and the relevant packages listed above. 

If you are experiencing issues with loading the tutorial jupyter notebooks, you can enter the URL/location of the notebooks into the following address: https://nbviewer.jupyter.org/. 

## Install

PyMint can be installed through pip, but we are working on uploading to conda-forge. 
```
pip install py-mint
```

## Dependencies 

PyMint is compatible with Python 3.6 or newer.  PyMint requires the following packages:
```
numpy 
pandas
scikit-learn
matplotlib
shap>=0.30.0
xarray>=0.16.0
tqdm
statsmodels
seaborn>=0.11.0
```

### Initializing PyMint
The interface of PyMint is the ```InterpretToolkit```, which houses the computations and plotting methods
for all the interpretability methods contained within. Once initialized ```InterpretToolkit``` can 
compute a variety of interpretability methods and plot them. See the tutorial notebooks for examples. 

```python
import pymint

# Loads three ML models (random forest, gradient-boosted tree, and logistic regression)
# trained on a subset of the road surface temperature data from Handler et al. (2020).
estimators = pymint.load_models()
X,y = pymint.load_data()

explainer = pymint.InterpretToolkit(estimators=estimators,X=X,y=y,)
```
## Permutation Importance

For predictor ranking, PyMint uses both single-pass and multiple-pass permutation importance method ([Breiman 2001]; Lakshmanan et al. 2015; McGovern et al. 2019).
We can calculate the permutation importance and then plot the results. In the tutorial it discusses options to make the figure publication-quality giving the plotting method
additional argument to convert the feature names to a more readable format or color coding by feature type. 
```python
perm_results = explainer.permutation_importance(n_vars=10, evaluation_fn='auc')
explainer.plot_importance(data=perm_results)
```

<p align="center">
  <img width="811" src="https://github.com/monte-flora/py-mint/blob/master/images/multi_pass_perm_imp.png?raw=true"  />
</p>

Sample notebook can be found here: [**Permutation Importance**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/permutation_importance_tutorial.ipynb) 


## Partial dependence and Accumulated Local Effects 

To compute the expected functional relationship between a feature and an ML model's prediction, you can use partial dependence or accumulated local effects. There is also an option for second-order interaction effects. For the choice of feature, you can manually select or can run the permutation importance and a built-in method will retrieve those features. It is also possible to configure the plot for readable feature names. 
```python 
# Assumes the .permutation_importance has already been run.
important_vars = explainer.get_important_vars(results, multipass=True, nvars=7)

ale = explainer.ale(features=important_vars, n_bins=20)
explainer.plot_ale(ale)
```
<p align="center">
  <img width="811" src="https://github.com/monte-flora/py-mint/blob/master/images/ale_1d.png?raw=true"  />
</p>

Additionally, you can use the same code snippet to compute the second-order ALE (see the notebook for more details). 

<p align="center">
  <img width="811" src="https://github.com/monte-flora/py-mint/blob/master/images/ale_2d.png?raw=true"  />
</p>

Sample notebook can be found here: 
- [**Accumulated Local effects**](https://github.com/monte-flora/pymint/blob/master/tutorial_notebooks/accumulated_local_effect_tutorial.ipynb) 
- [**Partial Dependence**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/partial_dependence_tutorial.ipynb) 


## Feature Contributions 

To explain specific examples, you can use SHAP values. PyMint employs both KernelSHAP for any model and TreeSHAP for tree-based methods. In future work, PyMint will also include DeepSHAP for convolution neural network-based models. PyMint can create the summary and dependence plots from the shap python package, but is adapted for multiple predictors and an easier user interface. It is also possible to plot contributions for a single example or summarized by model performance. 

```python
import shap
single_example = examples.iloc[[0]]
explainer = pymint.InterpretToolkit(estimators=estimators[0], X=single_example,)

background_dataset = shap.sample(examples, 100)
results = explainer.local_contributions(method='shap', background_dataset=background_dataset)
fig = explainer.plot_contributions(results)
```
<p align="center">
  <img width="811" src="https://github.com/monte-flora/py-mint/blob/master/images/feature_contribution_single.png?raw=true" />
</p>

```python
explainer = pymint.InterpretToolkit(estimators=estimators[0],X=X, y=y)

background_dataset = shap.sample(examples, 100)
results = explainer.local_contributions(method='shap', background_dataset=background_dataset, performance_based=True,)
fig = myInterpreter.plot_contributions(results)
```

<p align="center">
  <img width="811" src="https://github.com/monte-flora/py-mint/blob/master/images/feature_contributions_perform.png?raw=true"  />
</p>

```python
explainer = pymint.InterpretToolkit(estimators=estimators[0],X=X, y=y)
                                
background_dataset = shap.sample(examples, 100)
results = explainer.shap(background_dataset=background_dataset)
shap_values, bias = results['Random Forest']
explainer.plot_shap(plot_type = 'summary', shap_values=shap_values,) 
```

<p align="center">
  <img width="811" src="https://github.com/monte-flora/py-mint/blob/master/images/shap_dependence.png?raw=true"  />
</p>

```python
from pymint.common import plotting_config

features = ['tmp2m_hrs_bl_frez', 'sat_irbt', 'sfcT_hrs_ab_frez', 'tmp2m_hrs_ab_frez', 'd_rad_d']
explainer.plot_shap(features=features,
                        plot_type = 'dependence',
                        shap_values=shap_values,
                        display_feature_names=plotting_config.display_feature_names,
                        display_units = plotting_config.display_units,
                        to_probability=True)
```

<p align="center">
  <img width="811" src="https://github.com/monte-flora/py-mint/blob/master/images/shap_summary.png?raw=true" />
</p>

Sample notebook can be found here: 
- [**Feature Contributions**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/feature_contributions.ipynb) 
- [**SHAP-Style Plots**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/shap_style_plots.ipynb) 


## Tutorial notebooks

The notebooks provides the package documentation and demonstrate PyMint API, which was used to create the above figures. If you are experiencing issues with loading the jupyter notebooks, you can enter the URL/location of the notebooks into the following address: https://nbviewer.jupyter.org/. 

- [**Permutation Importance**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/permutation_importance_tutorial.ipynb) 
- [**Accumulated Local effects**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/accumulated_local_effect_tutorial.ipynb) 
- [**Partial Dependence**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/partial_dependence_tutorial.ipynb) 
- [**Feature Contributions**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/feature_contributions.ipynb) 
- [**SHAP-Style Plots**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/shap_style_plots.ipynb) 



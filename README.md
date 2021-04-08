<p>
  <img src="https://github.com/monte-flora/py-mint/blob/master/images/mintpy_logo.png?raw=true" align="right" width="400" height="400" />
</p>


<a href="https://travis-ci.com/monte-flora/mintpy"><img src="https://travis-ci.com/monte-flora/mintpy.svg?branch=master"></a>
[![codecov](https://codecov.io/gh/monte-flora/mintpy/branch/master/graph/badge.svg?token=GG9NRQOZ0N)](https://codecov.io/gh/monte-flora/mintpy)
[![Updates](https://pyup.io/repos/github/monte-flora/mintpy/shield.svg)](https://pyup.io/repos/github/monte-flora/mintpy/)
[![Python 3](https://pyup.io/repos/github/monte-flora/mintpy/python-3-shield.svg)](https://pyup.io/repos/github/monte-flora/mintpy/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI](https://img.shields.io/pypi/v/py-mint)

__PyMint__ (__Python-based Model INTerpretations__) is designed to be a user-friendly package for computing and plotting machine learning interpretation output in Python. Current computation includes partial dependence (PD), accumulated local effects (ALE), random forest-based feature contributions (treeinterpreter), single- and multiple-pass permutation importance, and Shapley Additive Explanations (SHAP). All of these methods are discussed at length in [Christoph Molnar's interpretable ML book](https://christophm.github.io/interpretable-ml-book/). Most calculations can be performed in parallel when multi-core processing is available. The primary feature of this package is the accompanying built-in plotting methods, which are desgined to be easy to use while producing publication-level quality figures. 

The package is under active development and will likely contain bugs or errors. Feel free to raise issues!

This package is largely original code, but also includes snippets from preexisting packages. Our goal is not take credit from other code authors, but to
make a single source for computing several machine learning interpretation methods. Here is a list of packages used in PyMint: 
[**PyALE**](https://github.com/DanaJomar/PyALE),
[**PermutationImportance**](https://github.com/gelijergensen/PermutationImportance),
[**ALEPython**](https://github.com/blent-ai/ALEPython),
[**SHAP**](https://github.com/slundberg/shap/), 
[**Scikit-Learn**](https://github.com/scikit-learn/scikit-learn)

If you employ PyMint in your research, please cite this github and the relevant packages listed above. 

## Install

PyMint can be installed through pip or conda-forge. 
```
pip install py-mint
or 
conda install -c conda-forge py-mint
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
```

### Initializing PyMint
The interface of PyMint is the ```InterpretToolkit```, which houses the computations and plotting methods
for all the interpretability methods contained within. Once initialized ```InterpretToolkit``` can 
compute a variety of interpretability methods and plot them. See the tutorial notebooks for examples. 

```python
import pymint

# Loads three ML models (random forest, gradient-boosted tree, and logistic regression)
# trained on a subset of the road surface temperature data from Handler et al. (2020).
model_objs, model_names = pymint.load_models()
examples, targets = pymint.load_data()

myInterpreter = pymint.InterpretToolkit(model=model_objs,
                                 model_names=model_names,
                                 examples=examples,
                                 targets=targets,
                                )
```
## Permutation Importance

For predictor ranking, PyMint uses both single-pass and multiple-pass permutation importance method (Breiman 2001; Lakshmanan et al. 2015; McGovern et al. 2019).
We can calculate the permutation importance and then plot the results. In the tutorial it discusses options to make the figure publication-quality giving the plotting method
additional argument to convert the feature names to a more readable format or color coding by feature type. 
```python
myInterpreter.calc_permutation_importance(n_vars=10, evaluation_fn='auc')
myInterpreter.plot_importance(method='multipass')
```

<p align="center">
  <img width="811" src="https://github.com/monte-flora/py-mint/blob/master/images/multi_pass_perm_imp.png?raw=true"  />
</p>

Sample notebook can be found here: [**Permutation Importance**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/permutation_importance_tutorial.ipynb) 


## Partial dependence and Accumulated Local Effects 

To compute the expected functional relationship between a feature and an ML model's prediction, you can use partial dependence or accumulated local effects. There is also an option for second-order interaction effects. For the choice of feature, you can manually select or can run the permutation importance and a built-in method will retrieve those features. It is also possible to configure the plot for readable feature names. 
```python 
# Assumes the calc_permutation_importance has already been run.
important_vars = myInterpreter.get_important_vars(results, multipass=True, nvars=7)

myInterpreter.calc_ale(features=important_vars, n_bins=20)
myInterpreter.plot_ale()
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
single_example = examples.iloc[[0]]
myInterpreter = pymint.InterpretToolkit(models=model_objs[0],
                                 model_names=model_names[0],
                                 examples=single_example,
                                 targets=targets,
                                )

background_dataset = shap.sample(examples, 100)
results = myInterpreter.calc_contributions(method='shap', background_dataset=background_dataset)
fig = myInterpreter.plot_contributions()
```
<p align="center">
  <img width="811" src="https://github.com/monte-flora/py-mint/blob/master/images/feature_contribution_single.png?raw=true" />
</p>

```python
myInterpreter = pymint.InterpretToolkit(models=model_objs[0],
                                 model_names=model_names[0],
                                 examples=examples,
                                 targets=targets,
                                )

background_dataset = shap.sample(examples, 100)
results = myInterpreter.calc_contributions(method='shap', background_dataset=background_dataset, performance_based=True,)
fig = myInterpreter.plot_contributions()
```

<p align="center">
  <img width="811" src="https://github.com/monte-flora/py-mint/blob/master/images/feature_contributions_perform.png?raw=true"  />
</p>

```python
myInterpreter = pymint.InterpretToolkit(models=model_objs[0],
                                 model_names=model_names[0],
                                 examples=examples,
                                 targets=targets,
                                )
                                
background_dataset = shap.sample(examples, 100)
results = myInterpreter.calc_shap(background_dataset=background_dataset)
shap_values, bias = results['Random Forest']
myInterpreter.plot_shap(plot_type = 'summary', shap_values=shap_values,) 
```

<p align="center">
  <img width="811" src="https://github.com/monte-flora/py-mint/blob/master/images/shap_dependence.png?raw=true"  />
</p>

```python
features = ['tmp2m_hrs_bl_frez', 'sat_irbt', 'sfcT_hrs_ab_frez', 'tmp2m_hrs_ab_frez', 'd_rad_d']
myInterpreter.plot_shap(features=features,
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

The notebooks provides the package documentation and demonstrate PyMint API, which was used to create the above figures. 

- [**Permutation Importance**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/permutation_importance_tutorial.ipynb) 
- [**Accumulated Local effects**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/accumulated_local_effect_tutorial.ipynb) 
- [**Partial Dependence**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/partial_dependence_tutorial.ipynb) 
- [**Feature Contributions**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/feature_contributions.ipynb) 
- [**SHAP-Style Plots**](https://github.com/monte-flora/py-mint/blob/master/tutorial_notebooks/shap_style_plots.ipynb) 



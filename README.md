<a href="url"><img src="images/mintpy_logo.png" align="right" height="400" width="400" ></a>

<a href="https://travis-ci.org/monte-flora/mintpy"><img src="https://travis-ci.com/monte-flora/mintpy.svg?branch=master"></a>
[![codecov](https://codecov.io/gh/monte-flora/mintpy/branch/master/graph/badge.svg?token=GG9NRQOZ0N)](https://codecov.io/gh/monte-flora/mintpy)


__MintPy__ (__Model INTerpretability in Python__) is designed to be a user-friendly package for computing and plotting machine learning interpretation output in Python. Current computation includes partial dependence (PD), accumulated local effects (ALE), random forest-based feature contributions (treeinterpreter), single- and multiple-pass permutation importance, and Shapley Additive Explanations (SHAP). All of these methods are discussed at length in Christoph Molnar's interpretable ML book (https://christophm.github.io/interpretable-ml-book/). Most calculations can be performed in parallel when multi-core processing is available. The primary feature of this package is the accompanying built-in plotting methods, which are desgined to be easy to use while producing publication-level quality figures. 

The package is under active development and will likely contain bugs or errors. Feel free to raise issues!

This package is largely original code, but also includes snippets from preexisting packages. Our goal is not take credit from other code authors, but to
make a single source for computing several machine learning interpretation methods. 

### Install
MintPy can be installed through pip or conda-forge. 
```
pip install mintpy
or 
conda install -c conda-forge mintpy
```

### Dependencies 
MintPy is compatible with Python 3.6 or newer.  MintPy requires the following packages:
```
numpy 
pandas
scikit-learn
matplotlib
shap>=0.30.0
xarray>=0.16.0
tqdm
```

### Initializing MintPy
The interface of MintPy is the ```InterpretToolkit```, which houses the computations and plotting methods
for all the interpretability methods contained within. Once initialized ```InterpretToolkit``` can 
compute a variety of interpretability methods and plot them. See the tutorial notebooks for examples. 

```python
import mintpy

# Loads three ML models (random forest, gradient-boosted tree, and logistic regression)
# trained on a subset of the road surface temperature data from Handler et al. (2020).
model_objs, model_names = mintpy.load_models()
examples, targets = mintpy.load_data()

myInterpreter = mintpy.InterpretToolkit(model=model_objs,
                                 model_names=model_names,
                                 examples=examples,
                                 targets=targets,
                                )
```
### Permutation Importance
For predictor ranking, MintPy uses both single-pass and multiple-pass permutation importance method (Breiman 2001; Lakshmanan et al. 2015; McGovern et al. 2019).
We can calculate the permutation importance and then plot the results. In the tutorial it discusses options to make the figure publication-quality giving the plotting method
additional argument to convert the feature names to a more readable format or color coding by feature type. 
```
myInterpreter.calc_permutation_importance(n_vars=10, evaluation_fn='auc')
myInterpreter.plot_importance(multipass=True, metric = "Training AUC")
```
<a href="url"><img src="images/multi_pass_perm_imp.png" align="center" height="250" width="500" ></a>

### Partial dependence and Accumulated Local Effects 
To compute the expected functional relationship between a feature and an ML model's prediction, you can use partial dependence or accumulated local effects. There is also an option for second-order interaction effects. For the choice of feature, you can manually select or can run the permutation importance and a built-in method will retrieve those features. It is also possible to configure the plot for readable feature names. 
```
# Assumes the calc_permutation_importance has already been run.
important_vars = myInterpreter.get_important_vars(results, multipass=True, nvars=7)

myInterpreter.calc_ale(features=important_vars, nbins=20)
myInterpreter.plot_ale()
```
<a href="url"><img src="images/ale_1d.png" align="center" height="500" width="500" ></a>
Additionally, you can use the same code snippet to compute the second-order ALE (see the notebook for more details). 

<a href="url"><img src="images/ale_2d.png" align="center" height="500" width="500" ></a>

### Feature Contributions 
To explain specific examples, you can use SHAP values. MintPy employs both KernelSHAP for any model and TreeSHAP for tree-based methods. In future work, MintPy will also include DeepSHAP for convolution neural network-based models. MintPy can create the summary and dependence plots from the shap python package, but is adapted for multiple predictors and an easier user interface. It is also possible to plot contributions for a single example or summarized by model performance. 

<a href="url"><img src="images/feature_contribution_single.png" align="center" height="500" width="700" ></a>

<a href="url"><img src="images/feature_contributions_perform.png" align="center" height="500" width="700" ></a>

<a href="url"><img src="images/shap_summary.png" align="center" height="500" width="700" ></a>

<a href="url"><img src="images/shap_dependence.png" align="center" height="500" width="700" ></a>

### Tutorial notebooks

The notebooks provides the package documentation and demonstrate MintPy API, which was used to create the above figures. 



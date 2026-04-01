<p>
  <img src="https://github.com/monte-flora/scikit-explain/blob/master/images/mintpy_logo.png?raw=true" align="right" width="400" height="400" />
</p>


![Unit Tests](https://github.com/monte-flora/scikit-explain/actions/workflows/continuous_intergration.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI](https://img.shields.io/pypi/v/scikit-explain)
[![Documentation Status](https://readthedocs.org/projects/scikit-explain/badge/?version=latest)](https://scikit-explain.readthedocs.io/en/latest/?badge=latest)


# scikit-explain

A user-friendly Python module for tabular machine learning explainability. For a comprehensive tutorial, see [Flora et al. (2024)](https://journals.ametsoc.org/view/journals/aies/3/1/AIES-D-23-0018.1.xml).

## Explainability Methods

### Feature Importance
  * Single- and Multi-pass Permutation Importance ([Breiman et al. 2001](https://link.springer.com/article/10.1023/A:1010933404324); [Lakshmanan et al. 2015](https://journals.ametsoc.org/view/journals/atot/32/6/jtech-d-13-00205_1.xml); [McGovern et al. 2019](https://journals.ametsoc.org/view/journals/bams/100/11/bams-d-18-0195.1.xml))
  * First-order PD/ALE Variance ([Greenwell et al. 2018](https://arxiv.org/abs/1805.04755))
  * Grouped Permutation Importance ([Au et al. 2021](https://arxiv.org/abs/2104.11688))

### Feature Effects/Attributions
  * [Partial Dependence](https://christophm.github.io/interpretable-ml-book/pdp.html) (PD)
  * [Accumulated Local Effects](https://christophm.github.io/interpretable-ml-book/ale.html) (ALE)
  * Individual Conditional Expectations (ICE)
  * [SHapley Additive Explanations](https://christophm.github.io/interpretable-ml-book/shap.html) (SHAP)
  * [Local Interpretable Model-Agnostic Explanations](https://christophm.github.io/interpretable-ml-book/lime.html) (LIME)
  * [TreeInterpreter](http://blog.datadive.net/interpreting-random-forests/) (tree-based feature contributions)

### Feature Interactions
  * Second-order PD/ALE
  * Interaction Strength (IAS) and Main Effect Complexity (MEC) ([Molnar et al. 2019](https://arxiv.org/abs/1904.03867))
  * Second-order PD/ALE Variance ([Greenwell et al. 2018](https://arxiv.org/abs/1805.04755))
  * Second-order Permutation Importance ([Oh et al. 2019](https://www.mdpi.com/2076-3417/9/23/5191))
  * Friedman H-statistic ([Friedman and Popescu 2008](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-2/issue-3/Predictive-learning-via-rule-ensembles/10.1214/07-AOAS148.full))
  * [Sobol Indices](https://towardsdatascience.com/sobol-indices-to-measure-feature-importance-54cedc3281bc)

These methods are discussed in Christoph Molnar's [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/). A primary feature of scikit-explain is the built-in plotting methods, designed to be easy to use while producing publication-quality figures. Documentation is available at [Read the Docs](https://scikit-explain.readthedocs.io/en/latest/).

## Installation

**pip** (PyPI):
```bash
pip install scikit-explain
```

**conda** (conda-forge):
```bash
conda install -c conda-forge scikit-explain
```

**Development version** (most up-to-date):
```bash
git clone https://github.com/monte-flora/scikit-explain.git
cd scikit-explain
pip install -e .
```

## Dependencies

scikit-explain is compatible with Python 3.8 or newer and requires:
```
numpy, scipy, pandas, scikit-learn, matplotlib, shap>=0.30.0,
xarray>=0.16.0, tqdm, statsmodels, seaborn>=0.11.0
```

## Quick Start

```python
import skexplain

# Load pre-trained models and data
estimators = skexplain.load_models()
X, y = skexplain.load_data()

# Create the explainer
explainer = skexplain.ExplainToolkit(estimators=estimators, X=X, y=y)

# Configure plot display settings once (optional)
explainer.set_plotting_config(
    display_feature_names={"sfc_temp": "$T_{sfc}$", "temp2m": "$T_{2m}$"},
    display_units={"sfc_temp": "$^\\circ$C", "temp2m": "$^\\circ$C"},
)
```

### Permutation Importance

```python
perm_results = explainer.permutation_importance(n_vars=10, evaluation_fn='norm_aupdc')
explainer.plot_importance(data=perm_results, panels=[('multipass', 'Random Forest')])
```

<p align="center">
  <img width="811" src="https://github.com/monte-flora/scikit-explain/blob/master/images/multi_pass_perm_imp.png?raw=true"  />
</p>

### Accumulated Local Effects

```python
important_vars = explainer.get_important_vars(perm_results, multipass=True, nvars=7)
ale = explainer.ale(features=important_vars, n_bins=20)
explainer.plot_ale(ale=ale)
```
<p align="center">
  <img width="811" src="https://github.com/monte-flora/scikit-explain/blob/master/images/ale_1d.png?raw=true"  />
</p>

### Feature Attributions

```python
import shap

single_example = X.iloc[[0]]
explainer = skexplain.ExplainToolkit(estimators=estimators, X=single_example)

shap_kws = {
    'masker': shap.maskers.Partition(X, max_samples=100, clustering="correlation"),
    'algorithm': 'auto',
}
attr_results = explainer.local_attributions(
    method=['shap', 'lime', 'tree_interpreter'],
    shap_kws=shap_kws,
)
explainer.plot_contributions(attr_results)
```
<p align="center">
  <img width="811" src="https://github.com/monte-flora/scikit-explain/blob/master/images/feature_contribution_single.png?raw=true" />
</p>

## Tutorial Notebooks

| Notebook | Description |
|----------|-------------|
| [01 Quickstart](tutorial_notebooks/01_quickstart.ipynb) | Minimal workflow from model to explanation |
| [02 Permutation Importance](tutorial_notebooks/02_permutation_importance.ipynb) | Single/multi-pass permutation importance |
| [03 Grouped Importance](tutorial_notebooks/03_grouped_importance.ipynb) | Grouped PI and comparing ranking methods |
| [04 ALE](tutorial_notebooks/04_ale.ipynb) | 1D Accumulated Local Effects |
| [05 Partial Dependence](tutorial_notebooks/05_partial_dependence.ipynb) | 1D Partial Dependence |
| [06 ICE Curves](tutorial_notebooks/06_ice_curves.ipynb) | Individual Conditional Expectations |
| [07 2D Effects](tutorial_notebooks/07_2d_effects.ipynb) | 2D ALE and Partial Dependence |
| [08 Local Attributions](tutorial_notebooks/08_local_attributions.ipynb) | SHAP, LIME, and TreeInterpreter |
| [09 SHAP Plots](tutorial_notebooks/09_shap_plots.ipynb) | Summary and dependence plots |
| [10 Interactions](tutorial_notebooks/10_interactions.ipynb) | H-statistic, IAS, MEC, Sobol indices |
| [11 Multiclass](tutorial_notebooks/11_multiclass.ipynb) | Multiclass classification support |
| [12 Plot Configuration](tutorial_notebooks/12_plot_configuration.ipynb) | Customizing plots with PlotConfig |

## Citation

If you use scikit-explain in your research, please cite:

```bibtex
@article{Flora_2024,
  author  = {Flora, Montgomery L. and McGovern, Amy and Handler, Shawn},
  title   = {A Machine Learning Explainability Tutorial for Atmospheric Sciences},
  journal = {Artificial Intelligence for the Earth Systems},
  volume  = {3},
  number  = {1},
  pages   = {e230018},
  year    = {2024},
  doi     = {10.1175/AIES-D-23-0018.1},
}
```

## Acknowledgments

This package includes adapted code from:
[PyALE](https://github.com/DanaJomar/PyALE),
[PermutationImportance](https://github.com/gelijergensen/PermutationImportance),
[ALEPython](https://github.com/blent-ai/ALEPython),
[SHAP](https://github.com/slundberg/shap/),
[scikit-learn](https://github.com/scikit-learn/scikit-learn),
[LIME](https://github.com/marcotcr/lime),
[Faster-LIME](https://github.com/seansaito/Faster-LIME),
[treeinterpreter](https://github.com/andosa/treeinterpreter)

## Contributing

- Issue Tracker: https://github.com/monte-flora/scikit-explain/issues
- Source Code: https://github.com/monte-flora/scikit-explain

## License

BSD license.

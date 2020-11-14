__MintPy__ (__Model INTerpretability in Python__) is designed to be a user-friendly package for computing and plotting machine learning interpretation output in Python. Current computation includes partial dependence (PD), accumulated local effects (ALE), random forest-based feature contributions (treeinterpreter), single- and multiple-pass permutation importance, and Shapley Additive Explanations (SHAP). All of these methods are discussed at length in Christoph Molnar's interpretable ML book (https://christophm.github.io/interpretable-ml-book/). Most calculations can be performed in parallel when multi-core processing is available. The primary feature of this package is the accompanying built-in plotting methods, which are desgined to be easy to use while producing publication-level quality figures. 

The package is under active development and will likely contain bugs or errors. Feel free to raise issues!

This package is largely original code, but also includes snippets from preexisting packages. Our goal is not take credit from other code authors, but to
make a single source for computing several machine learning interpretation methods. 

### Install
MintPy can be installed from XXX
```
pip install mintpy
or 
conda install -c conda-forge mintpy
```

### Dependencies 
```
numpy 
pandas
scikit-learn
matplotlib
shap
```


### Initializing MintPy
The interface of MintPy is the ```InterpretToolkit```, which houses the computations and plotting methods
for all the interpretability methods contained within. See permutation_importance_tutorial notebook 
for initializing ```InterpretToolkit``` (set a link!). Once initialized ```InterpretToolkit``` can 
compute a variety of interpretability methods and plot them.

```python
import mintpy

myInterpreter = mintpy.InterpretToolkit(model=model_objs,
                                 model_names=model_names,
                                 examples=examples,
                                 targets=targets,
                                )
```
### Permutation Importance
For predictor ranking, we use the permutation importance method. 

![](images/perm_imp_fig.png?raw=true)

### Partial dependence 

### Accumulated local effect 

### Feature Contributions 

### Tutorial notebooks

The notebooks provide package documentation and demonstrate MintPy API. 



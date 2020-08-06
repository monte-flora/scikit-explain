# MintPy (Model INTerpretability in Python) 

MintPy is designed to be a user-friendly package for computing and plotting machine learning interpretation output. Current computation includes partial dependence (PD), accumulated local effects (ALE), random forest-based feature contributions (treeinterpreter), permutation importance, shapley values (SHAP). All of these methods are discussed at length in Christoph Molnar's interpretable ML book (https://christophm.github.io/interpretable-ml-book/). The calculations for PDP and ALE can be performed in parallel when multi-core processing is available. The accompanying built-in plotting methods are desgined to be easy to use and publication-level quality. 

The package is under active development and will likely contain bugs or errors. Feel free to raise issues!

This package is largely original code, but also includes preexisting packages into a single source. The goal is make a user-friendly python class for computing several machine learning interpretation methods. 

### Dependencies 
MintPy is designed to be use 
```
numpy 
pandas
scikit-learn
matplotlib
shap
```



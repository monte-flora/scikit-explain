# MintPy (Model INTerpretability in Python) 

![MintPyLogo](https://github.com/monte-flora/MintPy/blob/master/MintPyLogo.png)

MintPy is designed to be a user-friendly package for computing and plotting machine learning interpretation output. Current computation includes partial dependence (PD), accumulated local effects (ALE), feature contributions (random forest only), and permutation importance. All of these methods are discussed at length in Christoph Molnar's interpretable ML book (https://christophm.github.io/interpretable-ml-book/). The calculations for PDP and ALE can be performed in parallel when multi-core processing is available. The accompanying built-in plotting methods are desgined to be easy to use and publication-level quality 

The package is under active development and will likely contain bugs or errors. Feel free to raise issues!

This package is largely original code, but also includes preexisting packages into a single source. The goal is make a user-friendly python class for computing several machine learning interpretation methods. 
### Dependencies 
MintPy is designed to be use 
```
numpy 
pandas
scikit-learn
matplotlib
```


### Initializing ModelClarifier
Let's show an example of ModelClarifier in action using the RandomForestClassifier on the scikit-learn breast cancer dataset. 
```
from model_clarify import ModelClarifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

mc = ModelClarifier(model=clf, examples=X_train, targets=y_train)
```
### Partial dependence 

### Accumulated local effect 

### Permutation Importance 

### Feature Contributions (Tree-based Method Only)



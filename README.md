# MintPy (Model INTerpretability in Python) 

MintPy is designed to be a user-friendly package for computing and plotting machine learning interpretation output in Python. Current computation includes partial dependence (PD), accumulated local effects (ALE), random forest-based feature contributions (treeinterpreter), single- and multiple-pass permutation importance, and shapley values (SHAP). All of these methods are discussed at length in Christoph Molnar's interpretable ML book (https://christophm.github.io/interpretable-ml-book/). Most calculations can be performed in parallel when multi-core processing is available. A primary feature is the accompanying built-in plotting methods, which are desgined to be easy to use while producing publication-level quality. 

The package is under active development and will likely contain bugs or errors. Feel free to raise issues!

This package is largely original code, but also includes snippets from preexisting packages. Our goal is not take credit from other code authors, but to
make a single source for computing several machine learning interpretation methods. 

### Dependencies 
```
numpy 
pandas
scikit-learn
matplotlib
shap
```


### Initializing MintPy
```ruby
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```

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



# MintPy (Model INTerpretability in Python) 
Interpretability methods for traditional machine learning methods.
Computations include partial dependence, accumulate local effect, feature contributions (random forest only),
permutation importance. There are accompanying built-in plotting methods for each interpretability techinque desgined to be easy to use for end-users with limited experience with python and interpretability techniques. 

The package is under active development and will likely contain bugs or errors. Feel free to raise issues!

This package is largely original code, but also includes preexisting packages into a single source. The goal is make a user-friendly python class for computing several machine learning interpretation methods. 

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



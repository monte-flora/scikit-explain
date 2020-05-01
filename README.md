# ModelClarifier
Interpretable methods for traditional and deep learning machine model methods.
Methods include partial dependence, accumulated local effect, and treeinterpreter.

The package is under active development and will likely contain bugs or errors. Feel free to raise issues!

This package has original code, but is largely a compilation of preexisting packages into a single source. The goal is make a user-friendly python class for computing several machine learning interpretation methods. 

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

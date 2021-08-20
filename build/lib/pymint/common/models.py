import os

current_dir = os.getcwd()
from os.path import join
from joblib import load

path = os.path.dirname(os.path.realpath(__file__))

def load_models():
    """Loads models trained on the road surface temperature dataset from Handler et al. (2020)"""

    # Load the model objects. In this case, we are using two popular scikit-learn tree-based methods.
    # model_filepath = join(current_dir, 'models')
    model_fname = [
        "RandomForestClassifier.pkl",
        "GradientBoostingClassifier.pkl",
        "LogisticRegression.pkl",
    ]
    model_names = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
    model_objs = [load(join(path, "models", fname)) for fname in model_fname]

    return list(zip(model_names, model_objs))

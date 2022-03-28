
# Import the main package
from .main.explain_toolkit import ExplainToolkit
from .main.global_explainer import GlobalExplainer
from .main.local_explainer import LocalExplainer

# Import data for notebooks
from .common.models import load_models
from .common.dataset import load_data

__version__ = '0.0.4' 



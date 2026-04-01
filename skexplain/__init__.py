# Import the main package
from .main.explain_toolkit import ExplainToolkit
from .main.global_explainer import GlobalExplainer
from .main.local_explainer import LocalExplainer

# Import plot configuration
from .plot.config import PlotConfig

# Import data for notebooks
from .common.models import load_models
from .common.dataset import load_data

# Import utilities for advanced workflows
from .common.importance_utils import to_skexplain_importance, group_sage
from .common.contrib_utils import group_local_values, group_feature_values

__version__ = "1.0.0"

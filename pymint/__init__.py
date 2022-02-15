
# Import the main package
from .main.interpret_toolkit import InterpretToolkit
from .main.global_interpret import GlobalInterpret
from .main.local_interpret import LocalInterpret

# Import data for notebooks
from .common.models import load_models
from .common.dataset import load_data

__version__ = '0.2.6' 



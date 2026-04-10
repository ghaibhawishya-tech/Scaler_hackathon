from .environment import RouterEnvironment as RouterEnv

__version__ = "2.0.0"
__env_id__ = "RouterEnv-v1"

# Explicitly expose graders for OpenEnv parsing engine
from . import graders


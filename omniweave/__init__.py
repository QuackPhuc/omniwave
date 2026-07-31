"""OmniWeave — a hierarchical vision backbone built on two-sided tiled GEMM."""

__version__ = "0.1.0"

from omniweave.models.registry import create_model, create_model_from_config
from omniweave.ops.bigemm import bigemm

__all__ = ["create_model", "create_model_from_config", "bigemm"]

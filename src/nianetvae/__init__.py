"""NiaNetVAE: controlled MetroPT architecture-search and PdM experiments."""

from .config import StudyConfig, load_study_config
from .contracts import ArchitectureSpec, WorkflowSpec

__all__ = [
    "ArchitectureSpec",
    "StudyConfig",
    "WorkflowSpec",
    "load_study_config",
]

__version__ = "2.0.0"

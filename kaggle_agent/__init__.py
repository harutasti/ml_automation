"""Kaggle Automation Agent - A fully automated ML pipeline for Kaggle competitions."""

__version__ = "0.1.0"

from .pipeline import KagglePipeline
from .core import KaggleAuth, ProjectManager, CompetitionManager
from .core.experiment_tracker import ExperimentTracker
from .core.state_manager import StateManager
from .core.code_injector import CodeInjector
from .hooks.hook_manager import HookManager

__all__ = [
    'KagglePipeline',
    'KaggleAuth',
    'ProjectManager', 
    'CompetitionManager',
    'ExperimentTracker',
    'StateManager',
    'CodeInjector',
    'HookManager'
]
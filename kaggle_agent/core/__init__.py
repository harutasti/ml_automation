"""Core modules for Kaggle Agent."""

from .auth import KaggleAuth
from .project import ProjectManager
from .competition import CompetitionManager
from .competition_auth import CompetitionAuth

__all__ = ["KaggleAuth", "ProjectManager", "CompetitionManager", "CompetitionAuth"]
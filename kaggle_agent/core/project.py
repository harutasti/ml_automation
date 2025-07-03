"""Project initialization and management module."""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ProjectManager:
    """Manages Kaggle project initialization and structure."""
    
    def __init__(self, project_name: str, base_dir: Optional[str] = None):
        self.project_name = project_name
        self.base_dir = Path(base_dir or os.getcwd()) / project_name
        self.config_path = self.base_dir / "config.yaml"
        self.state_path = self.base_dir / ".kaggle_agent_state.yaml"
        
    def initialize(self, competition_name: str) -> bool:
        """Initialize a new Kaggle project."""
        try:
            # Create project structure
            self._create_directory_structure()
            
            # Create initial configuration
            self._create_initial_config(competition_name)
            
            # Initialize state tracking
            self._initialize_state()
            
            print(f"✓ Project '{self.project_name}' initialized successfully")
            return True
            
        except Exception as e:
            print(f"✗ Project initialization failed: {e}")
            return False
    
    def _create_directory_structure(self):
        """Create the project directory structure."""
        directories = [
            "data/raw",
            "data/processed",
            "data/submissions",
            "notebooks",
            "models",
            "features",
            "logs",
            "custom_modules",
            "experiments"
        ]
        
        for dir_path in directories:
            (self.base_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create .gitignore
        gitignore_content = """
# Data files
data/raw/*
data/processed/*
*.csv
*.json
*.pkl
*.h5

# Models
models/*
*.pth
*.joblib

# Logs
logs/*
*.log

# Kaggle
.kaggle/
kaggle.json

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
.kaggle_agent_state.yaml
experiments/*/
"""
        
        with open(self.base_dir / ".gitignore", "w") as f:
            f.write(gitignore_content.strip())
    
    def _create_initial_config(self, competition_name: str):
        """Create initial project configuration."""
        config = {
            "project": {
                "name": self.project_name,
                "competition": competition_name,
                "created_at": datetime.now().isoformat(),
                "version": "0.1.0"
            },
            "pipeline": {
                "auto_mode": True,
                "stages": {
                    "data_download": {"enabled": True},
                    "eda": {"enabled": True, "generate_report": True},
                    "feature_engineering": {
                        "enabled": True, 
                        "auto_generate": True,
                        "advanced_features": True
                    },
                    "modeling": {
                        "enabled": True,
                        "algorithms": ["lgbm", "xgboost", "catboost"],
                        "cv_folds": 5,
                        "create_ensemble": True,
                        "ensemble_methods": ["voting", "stacking", "advanced_stacking"]
                    },
                    "ensemble": {"enabled": True, "methods": ["voting", "stacking", "advanced_stacking"]},
                    "submission": {"enabled": True, "auto_submit": False}
                }
            },
            "hooks": {
                "after_data_download": None,
                "after_eda": None,
                "after_feature_engineering": None,
                "after_modeling": None,
                "before_submission": None
            },
            "optimization": {
                "hyperparameter_tuning": {
                    "enabled": True,
                    "method": "optuna",
                    "n_trials": 100,
                    "timeout": 3600
                }
            },
            "tracking": {
                "experiments": True,
                "mlflow": False,
                "save_all_models": False
            }
        }
        
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def _initialize_state(self):
        """Initialize pipeline state tracking."""
        initial_state = {
            "pipeline_status": "initialized",
            "current_stage": None,
            "completed_stages": [],
            "last_updated": datetime.now().isoformat(),
            "checkpoints": {},
            "metrics": {
                "best_cv_score": None,
                "best_model": None,
                "submission_scores": []
            }
        }
        
        with open(self.state_path, "w") as f:
            yaml.dump(initial_state, f, default_flow_style=False)
    
    def load_config(self) -> Dict[str, Any]:
        """Load project configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update project configuration."""
        config = self.load_config()
        
        # Deep merge updates
        def deep_merge(base: dict, update: dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(config, updates)
        
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current pipeline state."""
        if not self.state_path.exists():
            return {}
        
        with open(self.state_path, "r") as f:
            return yaml.safe_load(f)
    
    def update_state(self, updates: Dict[str, Any]):
        """Update pipeline state."""
        state = self.get_state()
        state.update(updates)
        state["last_updated"] = datetime.now().isoformat()
        
        with open(self.state_path, "w") as f:
            yaml.dump(state, f, default_flow_style=False)
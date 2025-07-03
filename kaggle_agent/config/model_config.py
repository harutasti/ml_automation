"""Centralized configuration for model training."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    
    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = 10
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_random_state: int = 42
    
    # Logistic Regression / Linear Models
    lr_max_iter: int = 1000
    lr_random_state: int = 42
    lr_solver: str = 'lbfgs'
    
    # LightGBM
    lgbm_n_estimators: int = 100
    lgbm_max_depth: int = 7
    lgbm_learning_rate: float = 0.1
    lgbm_num_leaves: int = 31
    lgbm_subsample: float = 0.8
    lgbm_colsample_bytree: float = 0.8
    lgbm_random_state: int = 42
    lgbm_verbosity: int = -1
    
    # XGBoost
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 7
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_random_state: int = 42
    
    # CatBoost
    catboost_iterations: int = 100
    catboost_depth: int = 7
    catboost_learning_rate: float = 0.1
    catboost_random_state: int = 42
    catboost_verbose: bool = False
    
    # General settings
    n_jobs: int = -1
    thread_count: int = -1


@dataclass
class ModelVariantConfig:
    """Configuration for model variants."""
    
    lgbm_variants: list = field(default_factory=lambda: [
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
        {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.03},
        {'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.01}
    ])
    
    xgboost_variants: list = field(default_factory=lambda: [
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
        {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.03},
        {'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.01}
    ])
    
    catboost_variants: list = field(default_factory=lambda: [
        {'iterations': 200, 'depth': 5, 'learning_rate': 0.05},
        {'iterations': 300, 'depth': 7, 'learning_rate': 0.03},
        {'iterations': 500, 'depth': 9, 'learning_rate': 0.01}
    ])


@dataclass
class TrainingConfig:
    """Configuration for model training process."""
    
    # Cross-validation
    cv_folds: int = 5
    cv_shuffle: bool = True
    cv_random_state: int = 42
    
    # Task detection
    classification_threshold: int = 10
    binary_classification_classes: set = field(default_factory=lambda: {0, 1})
    
    # Feature importance
    max_features_to_plot: int = 20
    feature_importance_threshold: float = 0.01
    
    # Ensemble
    ensemble_min_models: int = 2
    ensemble_score_threshold: float = 0.9  # Use models with score >= best_score * threshold
    
    # Optimization
    default_optimization_trials: int = 50
    default_optimization_timeout: int = 3600  # 1 hour
    
    # Visualization
    figure_size: tuple = (10, 6)
    figure_dpi: int = 100
    
    # Performance
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 50
    
    # Memory optimization
    reduce_memory_usage: bool = True
    chunk_size: Optional[int] = None  # For large datasets


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    
    model_config: ModelConfig = field(default_factory=ModelConfig)
    variant_config: ModelVariantConfig = field(default_factory=ModelVariantConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get parameters for a specific model type."""
        params = {}
        
        if model_type == 'random_forest':
            params = {
                'n_estimators': self.model_config.rf_n_estimators,
                'max_depth': self.model_config.rf_max_depth,
                'min_samples_split': self.model_config.rf_min_samples_split,
                'min_samples_leaf': self.model_config.rf_min_samples_leaf,
                'random_state': self.model_config.rf_random_state,
                'n_jobs': self.model_config.n_jobs
            }
        elif model_type == 'logistic_regression':
            params = {
                'max_iter': self.model_config.lr_max_iter,
                'random_state': self.model_config.lr_random_state,
                'solver': self.model_config.lr_solver
            }
        elif model_type == 'lgbm':
            params = {
                'n_estimators': self.model_config.lgbm_n_estimators,
                'max_depth': self.model_config.lgbm_max_depth,
                'learning_rate': self.model_config.lgbm_learning_rate,
                'num_leaves': self.model_config.lgbm_num_leaves,
                'subsample': self.model_config.lgbm_subsample,
                'colsample_bytree': self.model_config.lgbm_colsample_bytree,
                'random_state': self.model_config.lgbm_random_state,
                'verbosity': self.model_config.lgbm_verbosity,
                'n_jobs': self.model_config.n_jobs
            }
        elif model_type == 'xgboost':
            params = {
                'n_estimators': self.model_config.xgb_n_estimators,
                'max_depth': self.model_config.xgb_max_depth,
                'learning_rate': self.model_config.xgb_learning_rate,
                'subsample': self.model_config.xgb_subsample,
                'colsample_bytree': self.model_config.xgb_colsample_bytree,
                'random_state': self.model_config.xgb_random_state,
                'n_jobs': self.model_config.n_jobs
            }
        elif model_type == 'catboost':
            params = {
                'iterations': self.model_config.catboost_iterations,
                'depth': self.model_config.catboost_depth,
                'learning_rate': self.model_config.catboost_learning_rate,
                'random_state': self.model_config.catboost_random_state,
                'verbose': self.model_config.catboost_verbose,
                'thread_count': self.model_config.thread_count
            }
        
        return params
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create PipelineConfig from dictionary."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        variant_config = ModelVariantConfig(**config_dict.get('variants', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        return cls(
            model_config=model_config,
            variant_config=variant_config,
            training_config=training_config
        )
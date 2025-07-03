"""Automated Model Selection and Training module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import joblib
from datetime import datetime
import warnings
import logging
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, precision_score, 
                           recall_score, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from .hyperparameter_optimization import HyperparameterOptimizer
from .ensemble import AutoEnsemble
from ..utils.validation import DataValidator
from ..config import PipelineConfig

# Import advanced optimization if available
try:
    from .advanced_optimization import AdvancedOptimizer
    ADVANCED_OPTIMIZATION_AVAILABLE = True
except ImportError:
    ADVANCED_OPTIMIZATION_AVAILABLE = False

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Set up logging
logger = logging.getLogger(__name__)


class AutoModeling:
    """Automated model selection and training for Kaggle competitions."""
    
    def __init__(self, output_dir: Path, task_type: Optional[str] = None, config: Optional[PipelineConfig] = None, competition_info: Optional[Dict] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_type = task_type  # 'classification' or 'regression'
        self.competition_info = competition_info  # Competition metadata from Kaggle API
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = None
        self.predictions = {}
        self.config = config or PipelineConfig()
        self.optimizer = None
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: Optional[pd.DataFrame] = None,
                    cv_folds: int = 5, 
                    algorithms: Optional[List[str]] = None,
                    optimize_hyperparameters: bool = False,
                    optimization_trials: int = 50,
                    optimization_timeout: Optional[int] = None,
                    create_ensemble: bool = True,
                    ensemble_methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train multiple models and select the best one."""
        logger.info("🤖 Starting Automated Model Training...")
        
        # Validate input data
        try:
            DataValidator.validate_dataframe(X_train, "X_train")
            DataValidator.validate_target(y_train, "y_train")
            DataValidator.validate_features_target_match(X_train, y_train)
            
            if X_test is not None:
                DataValidator.validate_dataframe(X_test, "X_test")
                DataValidator.validate_test_data(X_train, X_test)
            
            # Check data quality
            quality_report = DataValidator.check_data_quality(X_train)
            if quality_report['warnings']:
                logger.warning("Data quality issues detected:")
                for warning in quality_report['warnings']:
                    logger.warning(f"  - {warning}")
        
        except ValueError as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise
        
        # Detect task type if not specified
        if not self.task_type:
            # First try to get from competition info
            if self.competition_info and self.competition_info.get('task_type'):
                self.task_type = self.competition_info['task_type']
                logger.info(f"  Task type from competition info: {self.task_type}")
            else:
                # Fall back to detection from data
                self.task_type = self._detect_task_type(y_train)
                logger.info(f"  Task type detected from data: {self.task_type}")
        
        # Set default algorithms if not specified
        if not algorithms:
            algorithms = self._get_default_algorithms()
        
        # Setup cross-validation
        cv = self._setup_cv(y_train, cv_folds)
        
        # Get evaluation metric
        scoring = self._get_scoring_metric()
        
        # Initialize hyperparameter optimizer if requested
        if optimize_hyperparameters:
            self.optimizer = HyperparameterOptimizer(self.output_dir / "hyperopt")
        
        # Train each algorithm
        for algo in algorithms:
            logger.info(f"\n  Training {algo}...")
            try:
                # Get model instance
                model = self._get_model(algo)
                
                # Hyperparameter optimization if enabled
                if optimize_hyperparameters and algo in ['lgbm', 'xgboost', 'catboost', 'random_forest']:
                    if ADVANCED_OPTIMIZATION_AVAILABLE and algo in ['lgbm', 'xgboost', 'catboost']:
                        # Use advanced optimization with Optuna
                        logger.info(f"    Using advanced optimization with Optuna...")
                        advanced_opt = AdvancedOptimizer(self.task_type)
                        
                        if algo == 'lgbm':
                            best_params, best_score = advanced_opt.optimize_lgbm(
                                X_train, y_train, n_trials=optimization_trials, cv_folds=cv_folds
                            )
                            model = advanced_opt.get_model_with_best_params('lgbm')
                        else:
                            # Fall back to standard optimization for now
                            best_params, best_score = self.optimizer.optimize(
                                X_train, y_train, algo, self.task_type,
                                n_trials=optimization_trials,
                                timeout=optimization_timeout,
                                cv_folds=cv_folds,
                                metric=scoring
                            )
                            model = self.optimizer.get_optimized_model(algo, self.task_type)
                    else:
                        # Standard optimization
                        best_params, best_score = self.optimizer.optimize(
                            X_train, y_train, algo, self.task_type,
                            n_trials=optimization_trials,
                            timeout=optimization_timeout,
                            cv_folds=cv_folds,
                            metric=scoring
                        )
                        model = self.optimizer.get_optimized_model(algo, self.task_type)
                    
                    mean_score = best_score
                    std_score = 0.0  # Not available from optimization
                    scores = [best_score] * cv_folds  # Placeholder
                else:
                    # Standard cross-validation
                    scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                           scoring=scoring, n_jobs=1)
                    mean_score = scores.mean()
                    std_score = scores.std()
                    logger.info(f"    CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
                
                # Train on full data
                model.fit(X_train, y_train)
                
                # Make predictions on test set if available
                test_predictions = None
                if X_test is not None:
                    if self.task_type == 'classification':
                        test_predictions = model.predict_proba(X_test)[:, 1]
                    else:
                        test_predictions = model.predict(X_test)
                
                # Store results
                self.models[algo] = model
                self.results[algo] = {
                    'cv_scores': scores if isinstance(scores, list) else scores.tolist(),
                    'mean_cv_score': mean_score,
                    'std_cv_score': std_score,
                    'test_predictions': test_predictions,
                    'hyperparameters_optimized': optimize_hyperparameters and algo in ['lgbm', 'xgboost', 'catboost', 'random_forest']
                }
                
                # Update best model
                if self.best_score is None or mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = algo
                    
            except ValueError as e:
                logger.error(f"    ValueError training {algo}: {str(e)}")
                continue
            except MemoryError as e:
                logger.error(f"    MemoryError training {algo}: {str(e)}")
                logger.warning(f"    Consider reducing data size or using fewer features")
                continue
            except Exception as e:
                logger.error(f"    Unexpected error training {algo}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info(f"\n✓ Best model: {self.best_model} (CV Score: {self.best_score:.4f})")
        
        # Train best model variants if applicable
        # Temporarily disabled for performance - can be enabled via config
        # if self.best_model in ['lgbm', 'xgboost', 'catboost']:
        #     self._train_model_variants(X_train, y_train, X_test, cv)
        
        # Generate predictions ensemble
        if X_test is not None:
            self._generate_ensemble_predictions(X_test)
        
        # Create advanced ensemble models if requested
        if create_ensemble and len(self.models) >= 2:
            ensemble_methods = ensemble_methods or ['voting', 'stacking', 'blending']
            ensemble = AutoEnsemble(self.output_dir / "ensemble")
            ensemble_results = ensemble.create_ensemble(
                models=self.models,
                model_results=self.results,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                task_type=self.task_type,
                methods=ensemble_methods
            )
            
            # Update predictions with ensemble results
            for method, result in ensemble_results.items():
                if 'predictions' in result and result['predictions'] is not None:
                    self.predictions[f'ensemble_{method}'] = result['predictions']
                if 'cv_score' in result:
                    self.results[f'ensemble_{method}'] = {
                        'mean_cv_score': result['cv_score'],
                        'std_cv_score': 0.0,  # Ensemble methods don't provide std, set to 0
                        'test_predictions': result.get('predictions'),
                        'ensemble_type': method
                    }
            
            # Update best model if ensemble is better
            for method, result in self.results.items():
                if method.startswith('ensemble_') and result['mean_cv_score'] > self.best_score:
                    self.best_score = result['mean_cv_score']
                    self.best_model = method
        
        # Save results
        self._save_results()
        
        # Save hyperparameter optimization report if applicable
        if optimize_hyperparameters and self.optimizer:
            self.optimizer.save_summary_report()
        
        return self.results
    
    def _detect_task_type(self, y: pd.Series) -> str:
        """Detect if it's a classification or regression task."""
        # Check data type and unique values
        if y.dtype == 'object' or y.nunique() < 10 or set(y.unique()) == {0, 1}:
            return 'classification'
        else:
            return 'regression'
    
    def _get_default_algorithms(self) -> List[str]:
        """Get default algorithms based on task type."""
        if self.task_type == 'classification':
            return ['logistic_regression', 'random_forest', 'lgbm', 'xgboost', 'catboost']
        else:
            return ['ridge', 'lasso', 'random_forest', 'lgbm', 'xgboost', 'catboost']
    
    def _setup_cv(self, y: pd.Series, n_folds: int):
        """Setup cross-validation strategy."""
        if self.task_type == 'classification':
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        else:
            return KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    def _get_scoring_metric(self) -> str:
        """Get scoring metric based on task type."""
        if self.task_type == 'classification':
            # For binary classification, use ROC-AUC
            return 'roc_auc'
        else:
            # For regression, use negative MSE
            return 'neg_mean_squared_error'
    
    def _get_model(self, algorithm: str):
        """Get model instance based on algorithm name."""
        if self.task_type == 'classification':
            models = {
                'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
                'random_forest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=2
                ),
                'lgbm': lgb.LGBMClassifier(
                    n_estimators=100, max_depth=7, random_state=42, 
                    verbosity=-1, n_jobs=2
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100, max_depth=7, random_state=42,
                    use_label_encoder=False, eval_metric='logloss', n_jobs=2
                ),
                'catboost': CatBoostClassifier(
                    iterations=100, depth=7, random_state=42,
                    verbose=False, thread_count=2
                )
            }
        else:
            models = {
                'ridge': Ridge(alpha=1.0, random_state=42),
                'lasso': Lasso(alpha=1.0, random_state=42),
                'random_forest': RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=2
                ),
                'lgbm': lgb.LGBMRegressor(
                    n_estimators=100, max_depth=7, random_state=42,
                    verbosity=-1, n_jobs=2
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100, max_depth=7, random_state=42,
                    n_jobs=2
                ),
                'catboost': CatBoostRegressor(
                    iterations=100, depth=7, random_state=42,
                    verbose=False, thread_count=2
                )
            }
        
        return models.get(algorithm)
    
    def _train_model_variants(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: Optional[pd.DataFrame], cv):
        """Train different variants of the best model."""
        print(f"\n  Training {self.best_model} variants...")
        
        # Different parameter sets to try
        param_variants = {
            'lgbm': [
                {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
                {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.03},
                {'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.01}
            ],
            'xgboost': [
                {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
                {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.03},
                {'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.01}
            ],
            'catboost': [
                {'iterations': 200, 'depth': 5, 'learning_rate': 0.05},
                {'iterations': 300, 'depth': 7, 'learning_rate': 0.03},
                {'iterations': 500, 'depth': 9, 'learning_rate': 0.01}
            ]
        }
        
        if self.best_model not in param_variants:
            return
        
        variants = param_variants[self.best_model]
        scoring = self._get_scoring_metric()
        
        for i, params in enumerate(variants):
            variant_name = f"{self.best_model}_v{i+1}"
            
            try:
                # Get base model
                model = self._get_model(self.best_model)
                
                if model is None:
                    continue
                
                # Update parameters
                for param, value in params.items():
                    setattr(model, param, value)
            
                # Cross-validation
                scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                       scoring=scoring, n_jobs=-1)
                
                mean_score = scores.mean()
                print(f"    {variant_name}: {mean_score:.4f}")
                
                # Train on full data
                model.fit(X_train, y_train)
                
                # Make predictions
                test_predictions = None
                if X_test is not None:
                    if self.task_type == 'classification':
                        test_predictions = model.predict_proba(X_test)[:, 1]
                    else:
                        test_predictions = model.predict(X_test)
                
                # Store results
                self.models[variant_name] = model
                self.results[variant_name] = {
                    'cv_scores': scores.tolist(),
                    'mean_cv_score': mean_score,
                    'std_cv_score': scores.std(),
                    'test_predictions': test_predictions,
                    'params': params
                }
                
                # Update best model if better
                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = variant_name
                    
            except ValueError as e:
                print(f"    ValueError training {variant_name}: {str(e)}")
                continue
            except MemoryError as e:
                print(f"    MemoryError training {variant_name}: {str(e)}")
                continue
            except Exception as e:
                print(f"    Unexpected error training {variant_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    def _generate_ensemble_predictions(self, X_test: pd.DataFrame):
        """Generate ensemble predictions from multiple models."""
        print("\n  Generating ensemble predictions...")
        
        # Collect predictions from top models
        model_scores = [(name, res['mean_cv_score']) 
                       for name, res in self.results.items() 
                       if res['test_predictions'] is not None]
        
        # Sort by score and take top 5
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = model_scores[:5]
        
        if len(top_models) >= 3:
            # Simple average ensemble
            predictions = []
            weights = []
            
            for model_name, score in top_models:
                predictions.append(self.results[model_name]['test_predictions'])
                # Use CV score as weight
                weights.append(score)
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Weighted average
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            
            self.predictions['ensemble'] = ensemble_pred
            self.predictions['best_single'] = self.results[self.best_model]['test_predictions']
            
            print(f"    Ensemble created from {len(top_models)} models")
    
    def _save_results(self):
        """Save training results and models."""
        # Save results summary
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'task_type': self.task_type,
            'best_model': self.best_model,
            'best_cv_score': float(self.best_score),
            'model_results': {}
        }
        
        # Add individual model results
        for name, result in self.results.items():
            results_summary['model_results'][name] = {
                'mean_cv_score': float(result.get('mean_cv_score', 0.0)),
                'std_cv_score': float(result.get('std_cv_score', 0.0)),
                'params': result.get('params', {})
            }
        
        # Save JSON summary
        try:
            with open(self.output_dir / 'model_results.json', 'w') as f:
                json.dump(results_summary, f, indent=2)
        except IOError as e:
            print(f"❌ Failed to save model results JSON: {str(e)}")
            raise
        
        # Save best model
        if self.best_model and self.best_model in self.models:
            try:
                model_path = self.output_dir / f'best_model_{self.best_model}.pkl'
                joblib.dump(self.models[self.best_model], model_path)
                print(f"✓ Best model saved to: {model_path}")
            except IOError as e:
                print(f"❌ Failed to save best model: {str(e)}")
                raise
        else:
            print(f"⚠️ Warning: Best model '{self.best_model}' not found in models dict")
        
        # Save all predictions if available
        if self.predictions:
            predictions_df = pd.DataFrame(self.predictions)
            predictions_df.to_csv(self.output_dir / 'test_predictions.csv', index=False)
        
        # Create visualization of results
        self._create_results_visualization()
        
        print(f"  Results saved to: {self.output_dir}")
    
    def _create_results_visualization(self):
        """Create visualization of model comparison."""
        import matplotlib.pyplot as plt
        
        # Model comparison plot
        model_names = []
        mean_scores = []
        std_scores = []
        
        for name, result in sorted(self.results.items(), 
                                  key=lambda x: x[1]['mean_cv_score'], 
                                  reverse=True):
            model_names.append(name)
            mean_scores.append(result['mean_cv_score'])
            std_scores.append(result.get('std_cv_score', 0.0))
        
        plt.figure(figsize=(12, 6))
        x = range(len(model_names))
        plt.bar(x, mean_scores, yerr=std_scores, capsize=5)
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.ylabel('CV Score')
        plt.title('Model Performance Comparison')
        plt.grid(True, alpha=0.3)
        
        # Highlight best model
        best_idx = model_names.index(self.best_model)
        plt.bar(best_idx, mean_scores[best_idx], color='green', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=100)
        plt.close()
        
        # Feature importance for tree-based models
        if self.best_model in self.models:
            model = self.models[self.best_model]
            if hasattr(model, 'feature_importances_'):
                # Assume we have feature names from the DataFrame
                feature_names = list(range(len(model.feature_importances_)))
                
                # Get top 20 features
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:20]
                
                plt.figure(figsize=(10, 8))
                plt.barh(range(20), importances[indices])
                plt.yticks(range(20), [f'Feature {i}' for i in indices])
                plt.xlabel('Importance')
                plt.title(f'Feature Importance - {self.best_model}')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'feature_importance_model.png', dpi=100)
                plt.close()
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Make predictions using trained model."""
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if self.task_type == 'classification':
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)
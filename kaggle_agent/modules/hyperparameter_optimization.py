"""Hyperparameter optimization module using Optuna."""

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Callable, Optional, List, Tuple
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class HyperparameterOptimizer:
    """Automated hyperparameter optimization using Optuna."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.study_results = {}
        
    def optimize(self, 
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 algorithm: str,
                 task_type: str = 'classification',
                 n_trials: int = 100,
                 timeout: Optional[int] = None,
                 cv_folds: int = 5,
                 metric: Optional[str] = None) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters for a given algorithm."""
        
        print(f"\n🔧 Optimizing hyperparameters for {algorithm}...")
        
        # Create objective function
        objective = self._create_objective(
            X_train, y_train, algorithm, task_type, cv_folds, metric
        )
        
        # Create study
        direction = 'maximize' if task_type == 'classification' else 'minimize'
        study = optuna.create_study(direction=direction, study_name=f"{algorithm}_optimization")
        
        # Optimize
        study.optimize(
            objective, 
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Save results
        best_params = study.best_params
        best_score = study.best_value
        
        self.study_results[algorithm] = {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(study.trials),
            'best_trial': study.best_trial.number
        }
        
        # Save study results
        self._save_optimization_results(algorithm, study)
        
        print(f"✓ Best {algorithm} score: {best_score:.4f}")
        print(f"  Best params: {best_params}")
        
        return best_params, best_score
    
    def _create_objective(self, X_train, y_train, algorithm, task_type, cv_folds, metric):
        """Create objective function for Optuna."""
        
        def objective(trial):
            # Get hyperparameters based on algorithm
            if algorithm == 'lgbm':
                params = self._get_lgbm_params(trial, task_type)
                model = lgb.LGBMClassifier(**params) if task_type == 'classification' else lgb.LGBMRegressor(**params)
            
            elif algorithm == 'xgboost':
                params = self._get_xgboost_params(trial, task_type)
                model = xgb.XGBClassifier(**params) if task_type == 'classification' else xgb.XGBRegressor(**params)
            
            elif algorithm == 'catboost':
                params = self._get_catboost_params(trial, task_type)
                model = CatBoostClassifier(**params) if task_type == 'classification' else CatBoostRegressor(**params)
            
            elif algorithm == 'random_forest':
                params = self._get_rf_params(trial)
                model = RandomForestClassifier(**params) if task_type == 'classification' else RandomForestRegressor(**params)
            
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Cross-validation
            scoring = metric or ('accuracy' if task_type == 'classification' else 'neg_mean_squared_error')
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
            
            return scores.mean()
        
        return objective
    
    def _get_lgbm_params(self, trial, task_type):
        """Get LightGBM hyperparameters."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        if task_type == 'classification':
            params['objective'] = 'binary' if len(np.unique(trial.study.user_attrs.get('y_train', []))) == 2 else 'multiclass'
        
        return params
    
    def _get_xgboost_params(self, trial, task_type):
        """Get XGBoost hyperparameters."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        return params
    
    def _get_catboost_params(self, trial, task_type):
        """Get CatBoost hyperparameters."""
        params = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_state': 42,
            'verbose': False
        }
        
        return params
    
    def _get_rf_params(self, trial):
        """Get Random Forest hyperparameters."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        return params
    
    def _save_optimization_results(self, algorithm: str, study: optuna.Study):
        """Save optimization results and visualizations."""
        # Save study object
        study_path = self.output_dir / f"{algorithm}_study.pkl"
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        # Save results as JSON
        results = {
            'algorithm': algorithm,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'best_trial_number': study.best_trial.number,
            'trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': str(trial.state)
                }
                for trial in study.trials
            ]
        }
        
        results_path = self.output_dir / f"{algorithm}_optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate optimization history plot
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Optimization history
            trials = [trial.value for trial in study.trials if trial.value is not None]
            ax1.plot(trials, 'b-', alpha=0.5)
            ax1.plot(np.maximum.accumulate(trials) if study.direction == optuna.StudyDirection.MAXIMIZE else np.minimum.accumulate(trials), 'r-', linewidth=2)
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Objective Value')
            ax1.set_title(f'{algorithm} Optimization History')
            ax1.legend(['Trial Value', 'Best Value'])
            ax1.grid(True, alpha=0.3)
            
            # Parameter importance
            try:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())
                values = list(importance.values())
                
                ax2.barh(params, values)
                ax2.set_xlabel('Importance')
                ax2.set_title(f'{algorithm} Parameter Importance')
                ax2.grid(True, alpha=0.3)
            except:
                ax2.text(0.5, 0.5, 'Parameter importance not available', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{algorithm}_optimization_history.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not generate optimization plots: {e}")
    
    def get_optimized_model(self, algorithm: str, task_type: str = 'classification'):
        """Get model instance with optimized hyperparameters."""
        if algorithm not in self.study_results:
            raise ValueError(f"No optimization results found for {algorithm}")
        
        params = self.study_results[algorithm]['best_params']
        
        if algorithm == 'lgbm':
            return lgb.LGBMClassifier(**params) if task_type == 'classification' else lgb.LGBMRegressor(**params)
        elif algorithm == 'xgboost':
            return xgb.XGBClassifier(**params) if task_type == 'classification' else xgb.XGBRegressor(**params)
        elif algorithm == 'catboost':
            return CatBoostClassifier(**params) if task_type == 'classification' else CatBoostRegressor(**params)
        elif algorithm == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            return RandomForestClassifier(**params) if task_type == 'classification' else RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def save_summary_report(self):
        """Save a summary report of all optimizations."""
        report_path = self.output_dir / "hyperparameter_optimization_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Hyperparameter Optimization Report\n\n")
            
            for algorithm, results in self.study_results.items():
                f.write(f"## {algorithm.upper()}\n\n")
                f.write(f"- **Best Score**: {results['best_score']:.4f}\n")
                f.write(f"- **Number of Trials**: {results['n_trials']}\n")
                f.write(f"- **Best Trial**: #{results['best_trial']}\n\n")
                
                f.write("### Best Parameters\n")
                for param, value in results['best_params'].items():
                    f.write(f"- `{param}`: {value}\n")
                f.write("\n")
        
        print(f"\n📄 Optimization report saved to: {report_path}")
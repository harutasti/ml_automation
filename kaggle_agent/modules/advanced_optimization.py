"""Advanced hyperparameter optimization for competitive performance."""

import optuna
import numpy as np
from typing import Dict, Any, Callable, Optional
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedOptimizer:
    """Advanced hyperparameter optimization with Optuna."""
    
    def __init__(self, task_type: str = 'classification'):
        self.task_type = task_type
        self.best_params = {}
        self.best_score = None
        
    def get_lgbm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get LightGBM hyperparameters for trial."""
        return {
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'verbosity': -1,
            'n_jobs': 2,
            'random_state': 42
        }
    
    def get_xgb_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get XGBoost hyperparameters for trial."""
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
            'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'n_jobs': 2,
            'random_state': 42,
            'use_label_encoder': False
        }
    
    def get_catboost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get CatBoost hyperparameters for trial."""
        return {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'depth': trial.suggest_int('depth', 3, 16),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
            'od_wait': trial.suggest_int('od_wait', 10, 50),
            'thread_count': 2,
            'random_state': 42,
            'verbose': False
        }
    
    def optimize_lgbm(self, X_train, y_train, n_trials: int = 100, cv_folds: int = 5):
        """Optimize LightGBM with advanced strategies."""
        
        def objective(trial):
            params = self.get_lgbm_params(trial)
            
            if self.task_type == 'classification':
                model = lgb.LGBMClassifier(**params)
                kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            else:
                model = lgb.LGBMRegressor(**params)
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            scores = []
            for train_idx, val_idx in kf.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                
                if self.task_type == 'classification':
                    pred = model.predict_proba(X_fold_val)[:, 1]
                    score = roc_auc_score(y_fold_val, pred)
                else:
                    pred = model.predict(X_fold_val)
                    score = -mean_squared_error(y_fold_val, pred)
                    
                scores.append(score)
                
            return np.mean(scores)
        
        # Use TPE sampler with more advanced settings
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=20,
            n_ei_candidates=50,
            seed=42
        )
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1 hour timeout
        
        self.best_params['lgbm'] = study.best_params
        self.best_score = study.best_value
        
        return study.best_params, study.best_value
    
    def optimize_ensemble_weights(self, predictions_dict: Dict[str, np.ndarray], 
                                y_true: np.ndarray) -> Dict[str, float]:
        """Optimize ensemble weights using Optuna."""
        
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        def objective(trial):
            # Generate weights that sum to 1
            weights = []
            for i in range(n_models - 1):
                weights.append(trial.suggest_float(f'weight_{i}', 0.0, 1.0))
            
            # Last weight to ensure sum = 1
            last_weight = 1.0 - sum(weights)
            if last_weight < 0 or last_weight > 1:
                return -1e10  # Invalid weights
                
            weights.append(last_weight)
            
            # Calculate weighted average
            ensemble_pred = np.zeros_like(list(predictions_dict.values())[0])
            for weight, (model_name, pred) in zip(weights, predictions_dict.items()):
                ensemble_pred += weight * pred
                
            # Calculate score
            if self.task_type == 'classification':
                score = roc_auc_score(y_true, ensemble_pred)
            else:
                score = -mean_squared_error(y_true, ensemble_pred)
                
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=200)
        
        # Extract optimal weights
        optimal_weights = {}
        best_params = study.best_params
        
        for i, model_name in enumerate(model_names[:-1]):
            optimal_weights[model_name] = best_params[f'weight_{i}']
            
        # Last weight
        optimal_weights[model_names[-1]] = 1.0 - sum(optimal_weights.values())
        
        return optimal_weights
    
    def get_model_with_best_params(self, algorithm: str):
        """Get model instance with best parameters."""
        if algorithm not in self.best_params:
            raise ValueError(f"No optimized parameters found for {algorithm}")
            
        params = self.best_params[algorithm]
        
        if algorithm == 'lgbm':
            if self.task_type == 'classification':
                return lgb.LGBMClassifier(**params)
            else:
                return lgb.LGBMRegressor(**params)
        elif algorithm == 'xgboost':
            if self.task_type == 'classification':
                params['eval_metric'] = 'logloss'
                return xgb.XGBClassifier(**params)
            else:
                return xgb.XGBRegressor(**params)
        elif algorithm == 'catboost':
            if self.task_type == 'classification':
                return CatBoostClassifier(**params)
            else:
                return CatBoostRegressor(**params)
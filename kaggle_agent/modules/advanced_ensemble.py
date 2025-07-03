"""Advanced ensemble methods for competitive performance."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsemble:
    """Advanced ensemble techniques including multi-level stacking."""
    
    def __init__(self, task_type: str = 'classification'):
        self.task_type = task_type
        self.meta_models = {}
        self.base_predictions = {}
        
    def create_oof_predictions(self, models: Dict, X_train: pd.DataFrame, 
                             y_train: pd.Series, cv_folds: int = 5) -> pd.DataFrame:
        """Create out-of-fold predictions for stacking."""
        
        if self.task_type == 'classification':
            kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
        oof_predictions = pd.DataFrame(index=X_train.index)
        
        for model_name, model in models.items():
            oof_pred = np.zeros(len(X_train))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                
                # Clone model to avoid fitting the same instance
                model_clone = model.__class__(**model.get_params())
                
                # Special handling for different model types
                if hasattr(model_clone, 'fit'):
                    if 'eval_set' in model_clone.fit.__code__.co_varnames:
                        # LightGBM/XGBoost style
                        model_clone.fit(
                            X_fold_train, y_fold_train,
                            eval_set=[(X_fold_val, y_train.iloc[val_idx])],
                            verbose=False
                        )
                    else:
                        # Sklearn style
                        model_clone.fit(X_fold_train, y_fold_train)
                
                # Get predictions
                if self.task_type == 'classification':
                    if hasattr(model_clone, 'predict_proba'):
                        oof_pred[val_idx] = model_clone.predict_proba(X_fold_val)[:, 1]
                    else:
                        oof_pred[val_idx] = model_clone.predict(X_fold_val)
                else:
                    oof_pred[val_idx] = model_clone.predict(X_fold_val)
                    
            oof_predictions[f'{model_name}_pred'] = oof_pred
            
        return oof_predictions
    
    def create_multi_level_stack(self, models: Dict, X_train: pd.DataFrame, 
                               y_train: pd.Series, X_test: pd.DataFrame,
                               n_levels: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Create multi-level stacking ensemble."""
        
        train_predictions = []
        test_predictions = []
        
        # Level 1: Base models
        level1_train = self.create_oof_predictions(models, X_train, y_train)
        level1_test = pd.DataFrame(index=X_test.index)
        
        # Train base models on full data and predict on test
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                level1_test[f'{model_name}_pred'] = model.predict_proba(X_test)[:, 1]
            else:
                level1_test[f'{model_name}_pred'] = model.predict(X_test)
                
        # Level 2: Meta models on base predictions
        if n_levels >= 2:
            # Add original features (selected important ones)
            feature_importance = self._get_feature_importance(models, X_train, y_train)
            top_features = feature_importance.nlargest(20).index.tolist()
            
            level2_train = pd.concat([
                level1_train,
                X_train[top_features]
            ], axis=1)
            
            level2_test = pd.concat([
                level1_test,
                X_test[top_features]
            ], axis=1)
            
            # Train multiple meta models
            meta_models = {
                'meta_lgb': lgb.LGBMClassifier(n_estimators=100, num_leaves=10, random_state=42, n_jobs=2)
                            if self.task_type == 'classification' 
                            else lgb.LGBMRegressor(n_estimators=100, num_leaves=10, random_state=42, n_jobs=2),
                'meta_lr': LogisticRegression(random_state=42) 
                          if self.task_type == 'classification' 
                          else Ridge(random_state=42)
            }
            
            # Get OOF predictions for meta models
            meta_train = self.create_oof_predictions(meta_models, level2_train, y_train)
            meta_test = pd.DataFrame(index=X_test.index)
            
            for meta_name, meta_model in meta_models.items():
                meta_model.fit(level2_train, y_train)
                
                if self.task_type == 'classification' and hasattr(meta_model, 'predict_proba'):
                    meta_test[f'{meta_name}_pred'] = meta_model.predict_proba(level2_test)[:, 1]
                else:
                    meta_test[f'{meta_name}_pred'] = meta_model.predict(level2_test)
                    
            # Final blend
            train_blend = meta_train.mean(axis=1)
            test_blend = meta_test.mean(axis=1)
            
            return train_blend.values, test_blend.values
            
        else:
            # Simple average if only 1 level
            return level1_train.mean(axis=1).values, level1_test.mean(axis=1).values
    
    def create_feature_weighted_blend(self, models: Dict, X_train: pd.DataFrame,
                                    y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
        """Create predictions weighted by feature importance correlation."""
        
        predictions = []
        weights = []
        
        # Get feature importance for each model
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_test)[:, 1]
            else:
                pred = model.predict(X_test)
                
            predictions.append(pred)
            
            # Calculate weight based on feature importance alignment
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                # Weight is the variance of feature importances (models that use features differently)
                weight = np.var(imp)
            else:
                weight = 1.0
                
            weights.append(weight)
            
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        blend = np.average(predictions, axis=0, weights=weights)
        
        return blend
    
    def _get_feature_importance(self, models: Dict, X_train: pd.DataFrame, 
                              y_train: pd.Series) -> pd.Series:
        """Get aggregated feature importance from all models."""
        
        importance_df = pd.DataFrame(index=X_train.columns)
        
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df[model_name] = model.feature_importances_
                
        # Return mean importance across all models
        return importance_df.mean(axis=1).sort_values(ascending=False)
    
    def create_bayesian_model_average(self, predictions: Dict[str, np.ndarray],
                                    y_valid: np.ndarray) -> np.ndarray:
        """Create Bayesian Model Average ensemble."""
        
        from scipy.stats import norm
        
        # Calculate likelihood for each model
        likelihoods = {}
        for model_name, pred in predictions.items():
            if self.task_type == 'classification':
                # Use log loss as negative log likelihood
                epsilon = 1e-15
                pred_clipped = np.clip(pred, epsilon, 1 - epsilon)
                ll = -np.mean(y_valid * np.log(pred_clipped) + (1 - y_valid) * np.log(1 - pred_clipped))
            else:
                # Use MSE for regression
                ll = -np.mean((y_valid - pred) ** 2)
                
            likelihoods[model_name] = np.exp(ll)
            
        # Calculate posterior weights (assuming uniform prior)
        total_likelihood = sum(likelihoods.values())
        weights = {k: v / total_likelihood for k, v in likelihoods.items()}
        
        # Create weighted average
        ensemble = np.zeros_like(list(predictions.values())[0])
        for model_name, pred in predictions.items():
            ensemble += weights[model_name] * pred
            
        return ensemble
"""Automated Ensemble Methods module."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import advanced ensemble if available
try:
    from .advanced_ensemble import AdvancedEnsemble as AdvancedEnsembleMethods
    ADVANCED_ENSEMBLE_AVAILABLE = True
except ImportError:
    ADVANCED_ENSEMBLE_AVAILABLE = False


class AutoEnsemble:
    """Automated ensemble methods for improved predictions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble_models = {}
        self.ensemble_results = {}
        
    def create_ensemble(self, 
                       models: Dict[str, Any],
                       model_results: Dict[str, Any],
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_test: Optional[pd.DataFrame] = None,
                       task_type: str = 'classification',
                       methods: List[str] = ['voting', 'stacking', 'blending']) -> Dict[str, Any]:
        """Create various ensemble models."""
        
        print("\n🎯 Creating Ensemble Models...")
        
        # Filter models with good performance
        good_models = self._select_good_models(models, model_results)
        
        if len(good_models) < 2:
            print("  ⚠️  Not enough good models for ensemble. Need at least 2.")
            return {}
        
        results = {}
        
        # Voting Ensemble
        if 'voting' in methods:
            results['voting'] = self._create_voting_ensemble(
                good_models, X_train, y_train, X_test, task_type
            )
        
        # Stacking Ensemble
        if 'stacking' in methods:
            results['stacking'] = self._create_stacking_ensemble(
                good_models, X_train, y_train, X_test, task_type
            )
        
        # Weighted Average Ensemble (Blending)
        if 'blending' in methods:
            results['blending'] = self._create_blending_ensemble(
                good_models, model_results, X_test, task_type
            )
        
        # Advanced: Multi-level Stacking
        if 'multi_stacking' in methods and len(good_models) >= 3:
            results['multi_stacking'] = self._create_multi_level_stacking(
                good_models, X_train, y_train, X_test, task_type
            )
        
        # Use advanced ensemble methods if available
        if ADVANCED_ENSEMBLE_AVAILABLE:
            print("  📊 Using advanced ensemble methods...")
            advanced_ensemble = AdvancedEnsembleMethods(task_type)
            
            # Create multi-level stack with original features
            if len(good_models) >= 2 and X_test is not None:
                try:
                    train_pred, test_pred = advanced_ensemble.create_multi_level_stack(
                        good_models, X_train, y_train, X_test, n_levels=2
                    )
                    
                    # Evaluate on validation set
                    from sklearn.model_selection import train_test_split
                    X_val, X_holdout, y_val, y_holdout = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train if task_type == 'classification' else None
                    )
                    
                    # Get validation predictions
                    val_pred = advanced_ensemble.create_multi_level_stack(
                        good_models, X_val, y_val, X_holdout, n_levels=2
                    )[1]
                    
                    # Calculate score
                    if task_type == 'classification':
                        from sklearn.metrics import roc_auc_score
                        score = roc_auc_score(y_holdout, val_pred)
                    else:
                        from sklearn.metrics import mean_squared_error
                        score = -mean_squared_error(y_holdout, val_pred)
                    
                    results['advanced_stacking'] = {
                        'predictions': test_pred,
                        'cv_score': score,
                        'method': 'multi_level_stack'
                    }
                    print(f"    Advanced Stacking Score: {score:.4f}")
                except Exception as e:
                    print(f"    ⚠️ Advanced stacking failed: {str(e)}")
            
            # Optimize ensemble weights
            if len(good_models) >= 2:
                try:
                    # Get validation predictions for weight optimization
                    val_predictions = {}
                    for name, model in good_models.items():
                        if model_results[name].get('test_predictions') is not None:
                            val_predictions[name] = model_results[name]['test_predictions']
                    
                    if len(val_predictions) >= 2:
                        # This would need validation labels, skipping for now
                        pass
                except Exception as e:
                    print(f"    ⚠️ Weight optimization failed: {str(e)}")
        
        # Save ensemble models and results
        self._save_ensemble_results(results)
        
        return results
    
    def _select_good_models(self, models: Dict[str, Any], 
                           model_results: Dict[str, Any], 
                           threshold_percentile: float = 50) -> Dict[str, Any]:
        """Select models that perform above threshold."""
        scores = {name: result['mean_cv_score'] 
                 for name, result in model_results.items() 
                 if 'mean_cv_score' in result}
        
        if not scores:
            return models
        
        threshold = np.percentile(list(scores.values()), threshold_percentile)
        good_models = {name: model 
                      for name, model in models.items() 
                      if name in scores and scores[name] >= threshold}
        
        print(f"  Selected {len(good_models)} models for ensemble (score >= {threshold:.4f})")
        print(f"  Models: {', '.join(good_models.keys())}")
        
        return good_models
    
    def _create_voting_ensemble(self, models: Dict[str, Any],
                               X_train: pd.DataFrame,
                               y_train: pd.Series,
                               X_test: Optional[pd.DataFrame],
                               task_type: str) -> Dict[str, Any]:
        """Create voting ensemble."""
        print("\n  📊 Creating Voting Ensemble...")
        
        estimators = [(name, model) for name, model in models.items()]
        
        if task_type == 'classification':
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',  # Use probability averaging
                n_jobs=-1
            )
        else:
            ensemble = VotingRegressor(
                estimators=estimators,
                n_jobs=-1
            )
        
        # Cross-validation score
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5)
        mean_score = cv_scores.mean()
        
        print(f"    Voting Ensemble CV Score: {mean_score:.4f}")
        
        # Fit ensemble
        ensemble.fit(X_train, y_train)
        
        # Make predictions
        predictions = None
        if X_test is not None:
            if task_type == 'classification':
                predictions = ensemble.predict_proba(X_test)[:, 1]
            else:
                predictions = ensemble.predict(X_test)
        
        self.ensemble_models['voting'] = ensemble
        
        return {
            'model': ensemble,
            'cv_score': mean_score,
            'predictions': predictions,
            'models_used': list(models.keys())
        }
    
    def _create_stacking_ensemble(self, models: Dict[str, Any],
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 X_test: Optional[pd.DataFrame],
                                 task_type: str) -> Dict[str, Any]:
        """Create stacking ensemble."""
        print("\n  📚 Creating Stacking Ensemble...")
        
        estimators = [(name, model) for name, model in models.items()]
        
        # Meta-learner
        if task_type == 'classification':
            meta_learner = LogisticRegression(max_iter=1000, random_state=42)
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=5,  # Use cross-validation to train meta-learner
                n_jobs=-1
            )
        else:
            meta_learner = Ridge(random_state=42)
            ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=-1
            )
        
        # Cross-validation score
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=3)  # Less CV folds due to nested CV
        mean_score = cv_scores.mean()
        
        print(f"    Stacking Ensemble CV Score: {mean_score:.4f}")
        
        # Fit ensemble
        ensemble.fit(X_train, y_train)
        
        # Make predictions
        predictions = None
        if X_test is not None:
            if task_type == 'classification':
                predictions = ensemble.predict_proba(X_test)[:, 1]
            else:
                predictions = ensemble.predict(X_test)
        
        self.ensemble_models['stacking'] = ensemble
        
        return {
            'model': ensemble,
            'cv_score': mean_score,
            'predictions': predictions,
            'models_used': list(models.keys()),
            'meta_learner': type(meta_learner).__name__
        }
    
    def _create_blending_ensemble(self, models: Dict[str, Any],
                                 model_results: Dict[str, Any],
                                 X_test: Optional[pd.DataFrame],
                                 task_type: str) -> Dict[str, Any]:
        """Create weighted average ensemble based on CV scores."""
        print("\n  🔄 Creating Blending Ensemble...")
        
        if X_test is None:
            print("    No test data available for blending")
            return {}
        
        # Get predictions from each model
        predictions = {}
        weights = {}
        
        for name, result in model_results.items():
            if 'test_predictions' in result and result['test_predictions'] is not None:
                predictions[name] = result['test_predictions']
                weights[name] = result['mean_cv_score']
        
        if len(predictions) < 2:
            print("    Not enough predictions for blending")
            return {}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {name: w/total_weight for name, w in weights.items()}
        
        # Create weighted average
        blend_predictions = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            blend_predictions += normalized_weights[name] * pred
        
        print(f"    Blending weights: {normalized_weights}")
        
        return {
            'predictions': blend_predictions,
            'weights': normalized_weights,
            'models_used': list(predictions.keys())
        }
    
    def _create_multi_level_stacking(self, models: Dict[str, Any],
                                    X_train: pd.DataFrame,
                                    y_train: pd.Series,
                                    X_test: Optional[pd.DataFrame],
                                    task_type: str) -> Dict[str, Any]:
        """Create multi-level stacking ensemble."""
        print("\n  🏗️  Creating Multi-Level Stacking Ensemble...")
        
        # Split models into two levels
        model_names = list(models.keys())
        mid_point = len(model_names) // 2
        
        level1_models = {name: models[name] for name in model_names[:mid_point]}
        level2_models = {name: models[name] for name in model_names[mid_point:]}
        
        # First level stacking
        level1_estimators = [(name, model) for name, model in level1_models.items()]
        
        if task_type == 'classification':
            level1_meta = lgb.LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbosity=-1)
            level1_stack = StackingClassifier(
                estimators=level1_estimators,
                final_estimator=level1_meta,
                cv=3,
                n_jobs=-1
            )
        else:
            level1_meta = lgb.LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, verbosity=-1)
            level1_stack = StackingRegressor(
                estimators=level1_estimators,
                final_estimator=level1_meta,
                cv=3,
                n_jobs=-1
            )
        
        # Fit first level
        level1_stack.fit(X_train, y_train)
        
        # Second level combines level1 stack with remaining models
        level2_estimators = [('level1_stack', level1_stack)] + [(name, model) for name, model in level2_models.items()]
        
        if task_type == 'classification':
            final_meta = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
            final_ensemble = StackingClassifier(
                estimators=level2_estimators,
                final_estimator=final_meta,
                cv=3,
                n_jobs=-1
            )
        else:
            final_meta = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
            final_ensemble = StackingRegressor(
                estimators=level2_estimators,
                final_estimator=final_meta,
                cv=3,
                n_jobs=-1
            )
        
        # Cross-validation score
        cv_scores = cross_val_score(final_ensemble, X_train, y_train, cv=3)
        mean_score = cv_scores.mean()
        
        print(f"    Multi-Level Stacking CV Score: {mean_score:.4f}")
        
        # Fit final ensemble
        final_ensemble.fit(X_train, y_train)
        
        # Make predictions
        predictions = None
        if X_test is not None:
            if task_type == 'classification':
                predictions = final_ensemble.predict_proba(X_test)[:, 1]
            else:
                predictions = final_ensemble.predict(X_test)
        
        self.ensemble_models['multi_stacking'] = final_ensemble
        
        return {
            'model': final_ensemble,
            'cv_score': mean_score,
            'predictions': predictions,
            'level1_models': list(level1_models.keys()),
            'level2_models': list(level2_models.keys()),
            'architecture': '2-level stacking'
        }
    
    def _save_ensemble_results(self, results: Dict[str, Any]):
        """Save ensemble models and results."""
        # Save models
        for name, result in results.items():
            if 'model' in result:
                model_path = self.output_dir / f"ensemble_{name}.pkl"
                joblib.dump(result['model'], model_path)
        
        # Save results summary
        summary = {}
        for name, result in results.items():
            summary[name] = {
                'cv_score': result.get('cv_score', None),
                'models_used': result.get('models_used', []),
                'has_predictions': result.get('predictions', None) is not None
            }
            if 'weights' in result:
                summary[name]['weights'] = result['weights']
            if 'architecture' in result:
                summary[name]['architecture'] = result['architecture']
        
        with open(self.output_dir / 'ensemble_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate ensemble comparison plot
        self._plot_ensemble_comparison(summary)
        
        print(f"\n✅ Ensemble models saved to: {self.output_dir}")
    
    def _plot_ensemble_comparison(self, summary: Dict[str, Any]):
        """Plot comparison of ensemble methods."""
        try:
            import matplotlib.pyplot as plt
            
            methods = []
            scores = []
            
            for method, info in summary.items():
                if info.get('cv_score') is not None:
                    methods.append(method)
                    scores.append(info['cv_score'])
            
            if methods:
                plt.figure(figsize=(10, 6))
                bars = plt.bar(methods, scores)
                
                # Color best performing method
                best_idx = np.argmax(scores)
                bars[best_idx].set_color('green')
                
                plt.xlabel('Ensemble Method')
                plt.ylabel('CV Score')
                plt.title('Ensemble Methods Comparison')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f'{score:.4f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'ensemble_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Warning: Could not create ensemble comparison plot: {e}")
    
    def get_best_ensemble(self) -> Tuple[str, Any]:
        """Get the best performing ensemble model."""
        best_name = None
        best_score = -np.inf
        
        # Load results
        results_path = self.output_dir / 'ensemble_results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            for name, info in results.items():
                if info.get('cv_score', -np.inf) > best_score:
                    best_score = info['cv_score']
                    best_name = name
        
        if best_name and (self.output_dir / f"ensemble_{best_name}.pkl").exists():
            model = joblib.load(self.output_dir / f"ensemble_{best_name}.pkl")
            return best_name, model
        
        return None, None
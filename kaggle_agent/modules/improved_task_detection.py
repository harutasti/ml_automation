"""Improved task type detection for competitions."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

class ImprovedTaskDetector:
    """Improved detection of classification vs regression tasks."""
    
    def detect_task_type(self, 
                        y: pd.Series, 
                        competition_info: Optional[Dict[str, Any]] = None,
                        column_name: Optional[str] = None) -> str:
        """
        Detect if it's a classification or regression task using multiple signals.
        
        Args:
            y: Target variable
            competition_info: Competition metadata (evaluation metric, etc.)
            column_name: Name of the target column
            
        Returns:
            'classification' or 'regression'
        """
        
        # 1. Check competition evaluation metric if available
        if competition_info and 'evaluation_metric' in competition_info:
            metric = competition_info['evaluation_metric'].lower()
            
            # Classification metrics
            classification_metrics = [
                'accuracy', 'auc', 'logloss', 'log_loss', 'mlogloss',
                'f1', 'precision', 'recall', 'map@', 'ndcg',
                'categorization', 'multiclass', 'binary'
            ]
            
            # Regression metrics  
            regression_metrics = [
                'rmse', 'mse', 'mae', 'rmsle', 'r2', 'mape',
                'squared', 'absolute', 'error'
            ]
            
            for cls_metric in classification_metrics:
                if cls_metric in metric:
                    return 'classification'
                    
            for reg_metric in regression_metrics:
                if reg_metric in metric:
                    return 'regression'
        
        # 2. Check column name hints
        if column_name:
            column_lower = column_name.lower()
            
            # Classification column names
            if any(word in column_lower for word in [
                'class', 'category', 'label', 'target_class', 'is_',
                'has_', 'type', 'group'
            ]):
                return 'classification'
                
            # Regression column names
            if any(word in column_lower for word in [
                'price', 'cost', 'amount', 'value', 'score',
                'rating', 'count', 'number', 'quantity'
            ]):
                # But check if it's discrete ratings (1-5 stars)
                if y.nunique() <= 10 and all(y.dropna() == y.dropna().astype(int)):
                    return 'classification'
                return 'regression'
        
        # 3. Analyze target variable characteristics
        
        # String/object type is always classification
        if y.dtype == 'object' or y.dtype.name == 'category':
            return 'classification'
        
        # Boolean type
        if y.dtype == 'bool':
            return 'classification'
        
        # Check for float targets that are actually classes
        unique_values = y.dropna().unique()
        n_unique = len(unique_values)
        
        # Binary classification
        if n_unique == 2:
            return 'classification'
        
        # Check if all values are integers (possible class labels)
        if all(y.dropna() == y.dropna().astype(int)):
            # Integer values - could be classification or regression
            
            # Sequential integers starting from 0 or 1 (likely classes)
            if np.array_equal(sorted(unique_values), 
                            np.arange(min(unique_values), max(unique_values) + 1)):
                if n_unique <= 20:  # Reasonable number of classes
                    return 'classification'
            
            # Non-sequential integers with few unique values
            if n_unique <= 20:
                return 'classification'
                
            # Many unique integer values - likely regression
            return 'regression'
        
        # Continuous values
        else:
            # Check for probability-like values
            if y.min() >= 0 and y.max() <= 1 and n_unique > 10:
                # Could be probability output from a model
                return 'regression'
                
            # Check value distribution
            # If values are clustered around specific points, might be classification
            value_counts = y.value_counts()
            if len(value_counts) <= 30 and value_counts.iloc[0] >= len(y) * 0.05:
                # Discrete values with reasonable frequency
                return 'classification'
            
            # Default to regression for continuous values
            return 'regression'
    
    def get_task_details(self, y: pd.Series, task_type: str) -> Dict[str, Any]:
        """Get additional details about the task."""
        details = {
            'task_type': task_type,
            'n_unique': y.nunique(),
            'dtype': str(y.dtype),
            'has_missing': y.isna().any()
        }
        
        if task_type == 'classification':
            details['n_classes'] = y.nunique()
            details['class_distribution'] = y.value_counts().to_dict()
            details['is_binary'] = y.nunique() == 2
            details['is_multiclass'] = y.nunique() > 2
            
            # Check class imbalance
            class_counts = y.value_counts()
            min_class = class_counts.min()
            max_class = class_counts.max()
            details['class_imbalance_ratio'] = max_class / min_class if min_class > 0 else np.inf
            
        else:  # regression
            details['mean'] = float(y.mean())
            details['std'] = float(y.std())
            details['min'] = float(y.min())
            details['max'] = float(y.max())
            details['skewness'] = float(y.skew())
            
        return details
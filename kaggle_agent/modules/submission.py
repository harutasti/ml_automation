"""Automated Submission Generation module."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class AutoSubmission:
    """Automated submission file generation for Kaggle competitions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.submission_info = {}
        
    def generate_submission(self, 
                          test_ids: pd.Series,
                          predictions: Union[np.ndarray, Dict[str, np.ndarray]],
                          sample_submission_path: Optional[str] = None,
                          competition_info: Optional[Dict] = None,
                          task_type: str = 'classification',
                          threshold: Optional[float] = None) -> pd.DataFrame:
        """Generate submission file in the correct format."""
        print("📝 Generating submission file...")
        
        # Handle different prediction formats
        if isinstance(predictions, dict):
            # Multiple predictions available
            if 'ensemble' in predictions:
                print("  Using ensemble predictions")
                pred_values = predictions['ensemble']
                prediction_type = 'ensemble'
            elif 'best_single' in predictions:
                print("  Using best single model predictions")
                pred_values = predictions['best_single']
                prediction_type = 'best_single'
            else:
                # Use first available
                prediction_type = list(predictions.keys())[0]
                pred_values = predictions[prediction_type]
                print(f"  Using {prediction_type} predictions")
        else:
            pred_values = predictions
            prediction_type = 'single'
        
        # Load sample submission if provided
        if sample_submission_path and Path(sample_submission_path).exists():
            sample_df = pd.read_csv(sample_submission_path)
            print(f"  Using sample submission format from: {sample_submission_path}")
            
            # Get column names
            id_col = sample_df.columns[0]
            target_col = sample_df.columns[1]
            
            # Check data type for target
            sample_target_dtype = sample_df[target_col].dtype
            
        else:
            # Use default column names
            # Check if test_ids has the correct name
            if test_ids.name and test_ids.name.lower() == 'passengerid':
                id_col = 'PassengerId'
            elif 'Id' in str(test_ids.name):
                id_col = 'Id'
            else:
                id_col = test_ids.name or 'id'
            
            # Infer target column name from competition info
            if competition_info and 'evaluation_metric' in competition_info:
                metric = competition_info['evaluation_metric'].lower()
                if 'accuracy' in metric or 'auc' in metric:
                    target_col = 'Survived'  # Common for binary classification
                elif 'rmse' in metric or 'mae' in metric:
                    target_col = 'target'  # Common for regression
                else:
                    target_col = 'prediction'
            else:
                target_col = 'target'
            
            sample_target_dtype = None
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            id_col: test_ids
        })
        
        # Process predictions based on task type
        if task_type == 'classification':
            if sample_target_dtype == 'int64' or sample_target_dtype == int:
                # Binary classification with integer output
                if threshold is None:
                    threshold = 0.5
                submission_df[target_col] = (pred_values > threshold).astype(int)
                print(f"  Applied threshold {threshold} for binary classification")
            elif sample_target_dtype == 'bool' or sample_target_dtype == bool:
                # Boolean output (True/False)
                if threshold is None:
                    threshold = 0.5
                submission_df[target_col] = (pred_values > threshold)
                print(f"  Applied threshold {threshold} for boolean classification")
            elif sample_target_dtype == 'object':
                # Check if sample uses True/False strings
                if set(sample_df[target_col].unique()) <= {'True', 'False', True, False}:
                    # Boolean output
                    submission_df[target_col] = (pred_values > 0.5)
                    print(f"  Applied threshold 0.5 for boolean string classification")
                else:
                    # Might need class labels
                    submission_df[target_col] = (pred_values > 0.5).astype(int)
            else:
                # Probability output
                submission_df[target_col] = pred_values
        else:
            # Regression - use predictions as is
            submission_df[target_col] = pred_values
        
        # Validate submission format
        self._validate_submission(submission_df, sample_submission_path)
        
        # Save submission
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        submission_path = self.output_dir / f'submission_{prediction_type}_{timestamp}.csv'
        submission_df.to_csv(submission_path, index=False)
        
        # Also save as latest
        latest_path = self.output_dir / 'submission_latest.csv'
        submission_df.to_csv(latest_path, index=False)
        
        # Store submission info
        self.submission_info = {
            'timestamp': timestamp,
            'prediction_type': prediction_type,
            'task_type': task_type,
            'threshold': threshold,
            'shape': submission_df.shape,
            'id_column': id_col,
            'target_column': target_col,
            'path': str(submission_path),
            'statistics': {
                'mean': float(submission_df[target_col].mean()),
                'std': float(submission_df[target_col].std()),
                'min': float(submission_df[target_col].min()),
                'max': float(submission_df[target_col].max())
            }
        }
        
        # Save submission info
        with open(self.output_dir / 'submission_info.json', 'w') as f:
            json.dump(self.submission_info, f, indent=2)
        
        print(f"✓ Submission saved to: {submission_path}")
        print(f"  Shape: {submission_df.shape}")
        print(f"  Target stats - Mean: {self.submission_info['statistics']['mean']:.4f}, "
              f"Std: {self.submission_info['statistics']['std']:.4f}")
        
        return submission_df
    
    def _validate_submission(self, submission_df: pd.DataFrame, 
                           sample_path: Optional[str] = None):
        """Validate submission format against sample."""
        issues = []
        
        # Check for NaN values
        if submission_df.isnull().any().any():
            nan_cols = submission_df.columns[submission_df.isnull().any()].tolist()
            issues.append(f"NaN values found in columns: {nan_cols}")
            # Fill NaN with median or mode
            for col in nan_cols:
                if submission_df[col].dtype in ['float64', 'int64']:
                    submission_df[col].fillna(submission_df[col].median(), inplace=True)
                else:
                    submission_df[col].fillna(submission_df[col].mode()[0], inplace=True)
        
        # Check against sample if provided
        if sample_path and Path(sample_path).exists():
            sample_df = pd.read_csv(sample_path)
            
            # Check column names
            if list(submission_df.columns) != list(sample_df.columns):
                issues.append(f"Column mismatch. Expected: {list(sample_df.columns)}, "
                            f"Got: {list(submission_df.columns)}")
            
            # Check number of rows
            if len(submission_df) != len(sample_df):
                issues.append(f"Row count mismatch. Expected: {len(sample_df)}, "
                            f"Got: {len(submission_df)}")
            
            # Check ID column
            id_col = sample_df.columns[0]
            if id_col in submission_df.columns and id_col in sample_df.columns:
                if not submission_df[id_col].equals(sample_df[id_col]):
                    issues.append("ID column values don't match sample submission")
        
        if issues:
            print("⚠️  Validation issues found:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  ✓ Submission validation passed")
    
    def create_submission_report(self, model_results: Optional[Dict] = None,
                               feature_importance: Optional[Dict] = None):
        """Create a comprehensive submission report."""
        report_path = self.output_dir / 'submission_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Kaggle Submission Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Submission info
            if self.submission_info:
                f.write("## Submission Details\n")
                f.write(f"- File: `{Path(self.submission_info['path']).name}`\n")
                f.write(f"- Type: {self.submission_info['prediction_type']}\n")
                f.write(f"- Task: {self.submission_info['task_type']}\n")
                f.write(f"- Shape: {self.submission_info['shape']}\n")
                f.write(f"- Target column: {self.submission_info['target_column']}\n\n")
                
                f.write("### Prediction Statistics\n")
                stats = self.submission_info['statistics']
                f.write(f"- Mean: {stats['mean']:.6f}\n")
                f.write(f"- Std: {stats['std']:.6f}\n")
                f.write(f"- Min: {stats['min']:.6f}\n")
                f.write(f"- Max: {stats['max']:.6f}\n\n")
            
            # Model results
            if model_results:
                f.write("## Model Performance\n")
                if 'best_model' in model_results:
                    f.write(f"- Best Model: **{model_results['best_model']}**\n")
                    f.write(f"- CV Score: {model_results['best_cv_score']:.4f}\n\n")
                
                if 'model_results' in model_results:
                    f.write("### All Models Tested\n")
                    sorted_models = sorted(model_results['model_results'].items(),
                                         key=lambda x: x[1]['mean_cv_score'],
                                         reverse=True)
                    for name, result in sorted_models:
                        std_str = f" (+/- {result['std_cv_score']:.4f})" if 'std_cv_score' in result and result['std_cv_score'] > 0 else ""
                        f.write(f"- {name}: {result['mean_cv_score']:.4f}{std_str}\n")
                    f.write("\n")
            
            # Feature importance
            if feature_importance:
                f.write("## Top Features\n")
                sorted_features = sorted(feature_importance.items(),
                                       key=lambda x: x[1], reverse=True)[:20]
                for i, (feat, importance) in enumerate(sorted_features, 1):
                    f.write(f"{i}. {feat}: {importance:.4f}\n")
                f.write("\n")
            
            # Submission checklist
            f.write("## Pre-submission Checklist\n")
            f.write("- [ ] Check submission file format matches sample\n")
            f.write("- [ ] Verify no NaN values in predictions\n")
            f.write("- [ ] Confirm ID column matches test set\n")
            f.write("- [ ] Review prediction distribution\n")
            f.write("- [ ] Compare with previous submissions\n")
            f.write("- [ ] Check competition deadline\n")
        
        print(f"  Report saved to: {report_path}")
    
    def compare_submissions(self, submission_files: List[str]):
        """Compare multiple submission files."""
        print("\n📊 Comparing submissions...")
        
        comparisons = []
        for file in submission_files:
            if Path(file).exists():
                df = pd.read_csv(file)
                target_col = df.columns[-1]  # Assume last column is target
                
                comparison = {
                    'file': Path(file).name,
                    'shape': df.shape,
                    'mean': df[target_col].mean(),
                    'std': df[target_col].std(),
                    'unique_values': df[target_col].nunique()
                }
                comparisons.append(comparison)
        
        if comparisons:
            comparison_df = pd.DataFrame(comparisons)
            print(comparison_df.to_string(index=False))
            
            # Save comparison
            comparison_df.to_csv(self.output_dir / 'submission_comparison.csv', index=False)
        
        return comparisons
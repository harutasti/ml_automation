"""Input validation utilities for Kaggle Agent."""

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
from pathlib import Path


class DataValidator:
    """Validate input data for ML pipeline."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
        """
        Validate a pandas DataFrame.
        
        Args:
            df: DataFrame to validate
            name: Name of the DataFrame for error messages
            
        Raises:
            ValueError: If validation fails
        """
        if df is None:
            raise ValueError(f"{name} is None")
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"{name} must be a pandas DataFrame, got {type(df)}")
        
        if df.empty:
            raise ValueError(f"{name} is empty")
        
        if df.shape[0] == 0:
            raise ValueError(f"{name} has no rows")
        
        if df.shape[1] == 0:
            raise ValueError(f"{name} has no columns")
    
    @staticmethod
    def validate_target(y: Union[pd.Series, np.ndarray], name: str = "Target") -> None:
        """
        Validate target variable.
        
        Args:
            y: Target variable to validate
            name: Name for error messages
            
        Raises:
            ValueError: If validation fails
        """
        if y is None:
            raise ValueError(f"{name} is None")
        
        if isinstance(y, pd.Series):
            if y.empty:
                raise ValueError(f"{name} is empty")
            if y.isnull().all():
                raise ValueError(f"{name} contains only null values")
        elif isinstance(y, np.ndarray):
            if y.size == 0:
                raise ValueError(f"{name} is empty")
            if np.isnan(y).all():
                raise ValueError(f"{name} contains only NaN values")
        else:
            raise ValueError(f"{name} must be pandas Series or numpy array, got {type(y)}")
    
    @staticmethod
    def validate_features_target_match(X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:
        """
        Validate that features and target have the same number of samples.
        
        Args:
            X: Features DataFrame
            y: Target variable
            
        Raises:
            ValueError: If shapes don't match
        """
        n_samples_X = X.shape[0]
        n_samples_y = len(y) if hasattr(y, '__len__') else y.shape[0]
        
        if n_samples_X != n_samples_y:
            raise ValueError(
                f"Features and target must have the same number of samples. "
                f"Features has {n_samples_X} samples, target has {n_samples_y} samples"
            )
    
    @staticmethod
    def validate_test_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        """
        Validate test data compatibility with training data.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Raises:
            ValueError: If validation fails
        """
        if X_test is None:
            return  # Test data is optional
        
        # Check column match
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        
        if train_cols != test_cols:
            missing_in_test = train_cols - test_cols
            extra_in_test = test_cols - train_cols
            
            error_msg = "Train and test data have different columns."
            if missing_in_test:
                error_msg += f"\nMissing in test: {missing_in_test}"
            if extra_in_test:
                error_msg += f"\nExtra in test: {extra_in_test}"
            
            raise ValueError(error_msg)
        
        # Check column order
        if list(X_train.columns) != list(X_test.columns):
            # Reorder test columns to match train
            X_test = X_test[X_train.columns]
    
    @staticmethod
    def validate_numeric_features(df: pd.DataFrame, exclude_cols: Optional[list] = None) -> Tuple[list, list]:
        """
        Validate and identify numeric features.
        
        Args:
            df: DataFrame to check
            exclude_cols: Columns to exclude from validation
            
        Returns:
            Tuple of (numeric_columns, non_numeric_columns)
        """
        exclude_cols = exclude_cols or []
        
        numeric_cols = []
        non_numeric_cols = []
        
        for col in df.columns:
            if col in exclude_cols:
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                non_numeric_cols.append(col)
        
        return numeric_cols, non_numeric_cols
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> dict:
        """
        Check data quality issues.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'n_rows': df.shape[0],
            'n_cols': df.shape[1],
            'missing_values': {},
            'all_null_columns': [],
            'constant_columns': [],
            'duplicate_rows': 0,
            'high_cardinality_columns': [],
            'warnings': []
        }
        
        # Check missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                quality_report['missing_values'][col] = {
                    'count': missing_count,
                    'percentage': missing_pct
                }
                
                if missing_pct > 90:
                    quality_report['warnings'].append(
                        f"Column '{col}' has {missing_pct:.1f}% missing values"
                    )
        
        # Check all-null columns
        all_null_cols = df.columns[df.isnull().all()].tolist()
        if all_null_cols:
            quality_report['all_null_columns'] = all_null_cols
            quality_report['warnings'].append(
                f"Columns with all null values: {all_null_cols}"
            )
        
        # Check constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                quality_report['constant_columns'].append(col)
        
        if quality_report['constant_columns']:
            quality_report['warnings'].append(
                f"Constant columns found: {quality_report['constant_columns']}"
            )
        
        # Check duplicate rows
        quality_report['duplicate_rows'] = df.duplicated().sum()
        if quality_report['duplicate_rows'] > 0:
            quality_report['warnings'].append(
                f"Found {quality_report['duplicate_rows']} duplicate rows"
            )
        
        # Check high cardinality columns (potential issues for encoding)
        for col in df.select_dtypes(include=['object']).columns:
            cardinality = df[col].nunique()
            if cardinality > 0.8 * len(df):
                quality_report['high_cardinality_columns'].append({
                    'column': col,
                    'unique_values': cardinality
                })
        
        return quality_report
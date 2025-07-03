"""Automated Feature Engineering module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# Import advanced features if available
try:
    from .advanced_features import AdvancedFeatureEngineering
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False


class AutoFeatureEngineering:
    """Automated feature engineering for Kaggle competitions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_importance = {}
        self.encoding_mappings = {}
        self.scaling_params = {}
        self.generated_features = []
        self.transformations = []
        
    def engineer_features(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None,
                         target_col: Optional[str] = None, eda_report: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform automated feature engineering."""
        print("🔧 Starting Automated Feature Engineering...")
        
        # Make copies to avoid modifying original data
        train = train_df.copy()
        test = test_df.copy() if test_df is not None else None
        
        # Extract target if present
        target = None
        if target_col and target_col in train.columns:
            target = train[target_col]
            train = train.drop(columns=[target_col])
        
        # Use EDA report if provided
        if eda_report:
            data_types = eda_report.get('data_types', self._detect_data_types(train))
            missing_info = eda_report.get('missing_values', {}).get('train', {})
        else:
            data_types = self._detect_data_types(train)
            missing_info = self._analyze_missing(train)
        
        # 1. Handle missing values
        train, test = self._handle_missing_values(train, test, missing_info, data_types)
        
        # 2. Create datetime features
        train, test = self._create_datetime_features(train, test, data_types)
        
        # 3. Create numeric features
        train, test = self._create_numeric_features(train, test, data_types)
        
        # 4. Encode categorical features
        train, test = self._encode_categorical_features(train, test, data_types)
        
        # 5. Create interaction features
        train, test = self._create_interaction_features(train, test, data_types)
        
        # 6. Create aggregation features
        train, test = self._create_aggregation_features(train, test, data_types)
        
        # 7. Scale features if needed
        train, test = self._scale_features(train, test, data_types)
        
        # 8. Apply advanced feature engineering if available
        if ADVANCED_FEATURES_AVAILABLE and target is not None:
            print("  Applying advanced feature engineering...")
            advanced_fe = AdvancedFeatureEngineering()
            
            # Get updated column types after all transformations
            current_types = self._detect_data_types(train)
            cat_cols = [col for col in current_types['categorical'] if col in train.columns]
            num_cols = [col for col in current_types['numeric'] if col in train.columns]
            
            # Apply advanced techniques
            train, test = advanced_fe.apply_all_techniques(
                train, test, target_col, cat_cols, num_cols
            )
            self.generated_features.extend([col for col in train.columns if col not in self.generated_features])
        
        # 9. Feature selection
        if target is not None:
            train = self._select_features(train, target)
            if test is not None:
                test = test[train.columns]
        
        # Add target back if it was removed
        if target is not None:
            train[target_col] = target
        
        # Save feature engineering report
        self._save_report(train, test)
        
        print(f"✓ Feature engineering completed! Created {len(self.generated_features)} new features")
        
        return train, test
    
    def _detect_data_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Simple data type detection."""
        types = {
            'numeric': list(df.select_dtypes(include=[np.number]).columns),
            'categorical': list(df.select_dtypes(include=['object', 'category']).columns),
            'datetime': [],
            'text': [],
            'id': []
        }
        
        # Detect datetime columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col])
                    types['datetime'].append(col)
                    types['categorical'].remove(col) if col in types['categorical'] else None
                except:
                    pass
        
        return types
    
    def _analyze_missing(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze missing values."""
        missing = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing[col] = {
                    'count': missing_count,
                    'percentage': missing_count / len(df) * 100
                }
        return missing
    
    def _handle_missing_values(self, train: pd.DataFrame, test: Optional[pd.DataFrame],
                              missing_info: Dict, data_types: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle missing values intelligently."""
        print("  Handling missing values...")
        
        for col, info in missing_info.items():
            if col not in train.columns:
                continue
                
            missing_pct = info['percentage']
            
            # Drop columns with too many missing values
            if missing_pct > 80:
                print(f"    Dropping {col} (missing: {missing_pct:.1f}%)")
                train = train.drop(columns=[col])
                if test is not None and col in test.columns:
                    test = test.drop(columns=[col])
                continue
            
            # Handle based on data type
            if col in data_types['numeric']:
                # For numeric: use median or create indicator
                if missing_pct > 20:
                    # Create missing indicator
                    train[f'{col}_was_missing'] = train[col].isnull().astype(int)
                    if test is not None:
                        test[f'{col}_was_missing'] = test[col].isnull().astype(int)
                    self.generated_features.append(f'{col}_was_missing')
                
                # Fill with median
                median_val = train[col].median()
                train[col] = train[col].fillna(median_val)
                if test is not None:
                    test[col] = test[col].fillna(median_val)
                    
            elif col in data_types['categorical']:
                # For categorical: use mode or 'Unknown'
                if train[col].nunique() < 10:
                    mode_val = train[col].mode()[0] if len(train[col].mode()) > 0 else 'Unknown'
                    train[col] = train[col].fillna(mode_val)
                    if test is not None:
                        test[col] = test[col].fillna(mode_val)
                else:
                    train[col] = train[col].fillna('Unknown')
                    if test is not None:
                        test[col] = test[col].fillna('Unknown')
        
        return train, test
    
    def _create_datetime_features(self, train: pd.DataFrame, test: Optional[pd.DataFrame],
                                 data_types: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create features from datetime columns."""
        if not data_types['datetime']:
            return train, test
            
        print("  Creating datetime features...")
        
        for col in data_types['datetime']:
            if col not in train.columns:
                continue
                
            # Convert to datetime
            train[col] = pd.to_datetime(train[col], errors='coerce')
            if test is not None:
                test[col] = pd.to_datetime(test[col], errors='coerce')
            
            # Extract components
            for attr in ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter']:
                new_col = f'{col}_{attr}'
                train[new_col] = getattr(train[col].dt, attr)
                if test is not None:
                    test[new_col] = getattr(test[col].dt, attr)
                self.generated_features.append(new_col)
            
            # Create cyclical features for month and day
            train[f'{col}_month_sin'] = np.sin(2 * np.pi * train[col].dt.month / 12)
            train[f'{col}_month_cos'] = np.cos(2 * np.pi * train[col].dt.month / 12)
            train[f'{col}_day_sin'] = np.sin(2 * np.pi * train[col].dt.day / 31)
            train[f'{col}_day_cos'] = np.cos(2 * np.pi * train[col].dt.day / 31)
            
            if test is not None:
                test[f'{col}_month_sin'] = np.sin(2 * np.pi * test[col].dt.month / 12)
                test[f'{col}_month_cos'] = np.cos(2 * np.pi * test[col].dt.month / 12)
                test[f'{col}_day_sin'] = np.sin(2 * np.pi * test[col].dt.day / 31)
                test[f'{col}_day_cos'] = np.cos(2 * np.pi * test[col].dt.day / 31)
            
            self.generated_features.extend([f'{col}_month_sin', f'{col}_month_cos', 
                                          f'{col}_day_sin', f'{col}_day_cos'])
            
            # Drop original datetime column
            train = train.drop(columns=[col])
            if test is not None:
                test = test.drop(columns=[col])
        
        return train, test
    
    def _create_numeric_features(self, train: pd.DataFrame, test: Optional[pd.DataFrame],
                                data_types: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create features from numeric columns."""
        print("  Creating numeric features...")
        
        numeric_cols = [col for col in data_types['numeric'] if col in train.columns]
        
        if len(numeric_cols) < 2:
            return train, test
        
        # Log transformations for skewed features
        for col in numeric_cols:
            if train[col].min() > 0 and train[col].skew() > 1:
                new_col = f'{col}_log'
                train[new_col] = np.log1p(train[col])
                if test is not None:
                    test[new_col] = np.log1p(test[col])
                self.generated_features.append(new_col)
        
        # Polynomial features (only for top important features to avoid explosion)
        if len(numeric_cols) > 3:
            # Select top 3 numeric features based on variance
            top_features = train[numeric_cols].var().nlargest(3).index.tolist()
            
            for col in top_features:
                # Square
                new_col = f'{col}_squared'
                train[new_col] = train[col] ** 2
                if test is not None:
                    test[new_col] = test[col] ** 2
                self.generated_features.append(new_col)
                
                # Square root (if all positive)
                if train[col].min() >= 0:
                    new_col = f'{col}_sqrt'
                    train[new_col] = np.sqrt(train[col])
                    if test is not None:
                        test[new_col] = np.sqrt(test[col])
                    self.generated_features.append(new_col)
        
        return train, test
    
    def _encode_categorical_features(self, train: pd.DataFrame, test: Optional[pd.DataFrame],
                                   data_types: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features."""
        print("  Encoding categorical features...")
        
        categorical_cols = [col for col in data_types['categorical'] if col in train.columns]
        
        for col in categorical_cols:
            unique_values = train[col].nunique()
            
            if unique_values == 2:
                # Binary encoding
                le = LabelEncoder()
                train[col] = le.fit_transform(train[col].astype(str))
                if test is not None:
                    # Handle unseen categories
                    test[col] = test[col].astype(str)
                    test[col] = test[col].map(lambda x: le.transform([x])[0] 
                                             if x in le.classes_ else -1)
                self.encoding_mappings[col] = {'type': 'label', 'classes': le.classes_.tolist()}
                
            elif unique_values <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(train[col], prefix=col, dummy_na=True)
                train = pd.concat([train.drop(columns=[col]), dummies], axis=1)
                
                if test is not None:
                    test_dummies = pd.get_dummies(test[col], prefix=col, dummy_na=True)
                    # Align columns
                    for dummy_col in dummies.columns:
                        if dummy_col not in test_dummies.columns:
                            test_dummies[dummy_col] = 0
                    test_dummies = test_dummies[dummies.columns]
                    test = pd.concat([test.drop(columns=[col]), test_dummies], axis=1)
                
                self.encoding_mappings[col] = {'type': 'onehot', 'columns': dummies.columns.tolist()}
                self.generated_features.extend(dummies.columns.tolist())
                
            else:
                # Target encoding for high cardinality
                # Count encoding as a simple alternative
                count_map = train[col].value_counts().to_dict()
                new_col = f'{col}_count'
                train[new_col] = train[col].map(count_map)
                if test is not None:
                    test[new_col] = test[col].map(count_map).fillna(0)
                
                # Drop original high cardinality column
                train = train.drop(columns=[col])
                if test is not None:
                    test = test.drop(columns=[col])
                
                self.encoding_mappings[col] = {'type': 'count', 'map': count_map}
                self.generated_features.append(new_col)
        
        return train, test
    
    def _create_interaction_features(self, train: pd.DataFrame, test: Optional[pd.DataFrame],
                                   data_types: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create interaction features between important columns."""
        print("  Creating interaction features...")
        
        numeric_cols = [col for col in data_types['numeric'] if col in train.columns]
        
        if len(numeric_cols) < 2:
            return train, test
        
        # Create interactions for top features only
        if len(numeric_cols) > 5:
            # Select top 5 based on variance
            top_features = train[numeric_cols].var().nlargest(5).index.tolist()
        else:
            top_features = numeric_cols
        
        # Create pairwise products
        interactions_created = 0
        for i, col1 in enumerate(top_features):
            for col2 in top_features[i+1:]:
                if interactions_created >= 10:  # Limit number of interactions
                    break
                    
                new_col = f'{col1}_X_{col2}'
                train[new_col] = train[col1] * train[col2]
                if test is not None:
                    test[new_col] = test[col1] * test[col2]
                self.generated_features.append(new_col)
                interactions_created += 1
        
        return train, test
    
    def _create_aggregation_features(self, train: pd.DataFrame, test: Optional[pd.DataFrame],
                                   data_types: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create aggregation features."""
        print("  Creating aggregation features...")
        
        numeric_cols = [col for col in data_types['numeric'] if col in train.columns]
        
        if len(numeric_cols) < 3:
            return train, test
        
        # Row-wise statistics
        train['numeric_mean'] = train[numeric_cols].mean(axis=1)
        train['numeric_std'] = train[numeric_cols].std(axis=1)
        train['numeric_max'] = train[numeric_cols].max(axis=1)
        train['numeric_min'] = train[numeric_cols].min(axis=1)
        train['numeric_range'] = train['numeric_max'] - train['numeric_min']
        
        if test is not None:
            test['numeric_mean'] = test[numeric_cols].mean(axis=1)
            test['numeric_std'] = test[numeric_cols].std(axis=1)
            test['numeric_max'] = test[numeric_cols].max(axis=1)
            test['numeric_min'] = test[numeric_cols].min(axis=1)
            test['numeric_range'] = test['numeric_max'] - test['numeric_min']
        
        self.generated_features.extend(['numeric_mean', 'numeric_std', 'numeric_max', 
                                       'numeric_min', 'numeric_range'])
        
        return train, test
    
    def _scale_features(self, train: pd.DataFrame, test: Optional[pd.DataFrame],
                       data_types: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numeric features if needed."""
        print("  Scaling features...")
        
        # Get numeric columns (excluding any binary/encoded features)
        numeric_cols = []
        for col in train.columns:
            if train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if train[col].nunique() > 2 and train[col].max() > 10:
                    numeric_cols.append(col)
        
        if not numeric_cols:
            return train, test
        
        # Use StandardScaler
        scaler = StandardScaler()
        train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
        if test is not None:
            test[numeric_cols] = scaler.transform(test[numeric_cols])
        
        # Save scaling parameters
        self.scaling_params = {
            'columns': numeric_cols,
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }
        
        return train, test
    
    def _select_features(self, train: pd.DataFrame, target: pd.Series, 
                        max_features: Optional[int] = None) -> pd.DataFrame:
        """Select important features using mutual information."""
        print("  Selecting important features...")
        
        # Only use numeric columns for mutual information
        numeric_train = train.select_dtypes(include=[np.number])
        
        if numeric_train.empty:
            print("    No numeric features for selection")
            return train
        
        # Determine if classification or regression
        if target.dtype == 'object' or target.nunique() < 10:
            mi_scores = mutual_info_classif(numeric_train, target, random_state=42)
        else:
            mi_scores = mutual_info_regression(numeric_train, target, random_state=42)
        
        # Create feature importance dataframe
        feature_scores = pd.DataFrame({
            'feature': numeric_train.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        # Store feature importance
        self.feature_importance = feature_scores.set_index('feature')['importance'].to_dict()
        
        # Select features with importance > 0
        important_features = feature_scores[feature_scores['importance'] > 0]['feature'].tolist()
        
        # Add non-numeric features back (they might be important encoded features)
        non_numeric_cols = [col for col in train.columns if col not in numeric_train.columns]
        all_selected_features = important_features + non_numeric_cols
        
        # Limit number of features if specified
        if max_features and len(all_selected_features) > max_features:
            # Keep top numeric features and all non-numeric
            n_numeric = max_features - len(non_numeric_cols)
            if n_numeric > 0:
                all_selected_features = important_features[:n_numeric] + non_numeric_cols
            else:
                all_selected_features = all_selected_features[:max_features]
        
        print(f"    Selected {len(all_selected_features)} / {len(train.columns)} features")
        
        return train[all_selected_features]
    
    def _save_report(self, train: pd.DataFrame, test: Optional[pd.DataFrame]):
        """Save feature engineering report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'original_features': len(self.transformations),
            'generated_features': len(self.generated_features),
            'final_features': len(train.columns),
            'new_features': self.generated_features,
            'encoding_mappings': self.encoding_mappings,
            'scaling_params': self.scaling_params,
            'feature_importance': self.feature_importance
        }
        
        # Save JSON report
        report_path = self.output_dir / 'feature_engineering_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save feature importance plot
        if self.feature_importance:
            import matplotlib.pyplot as plt
            
            # Top 20 features
            top_features = sorted(self.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:20]
            
            if top_features:
                features, scores = zip(*top_features)
                
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(features)), scores)
                plt.yticks(range(len(features)), features)
                plt.xlabel('Mutual Information Score')
                plt.title('Top 20 Feature Importance')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'feature_importance.png', dpi=100)
                plt.close()
        
        # Save processed data samples
        train.head(100).to_csv(self.output_dir / 'train_sample.csv', index=False)
        if test is not None:
            test.head(100).to_csv(self.output_dir / 'test_sample.csv', index=False)
        
        print(f"  Report saved to: {self.output_dir}")
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same transformations to new data."""
        # This method would apply saved transformations to new data
        # Implementation depends on saving transformation pipeline
        pass
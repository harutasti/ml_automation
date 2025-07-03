"""Advanced feature engineering module for competitive performance."""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import RFE
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
import lightgbm as lgb
from typing import List, Tuple, Optional

class AdvancedFeatureEngineering:
    """Advanced feature engineering techniques for high performance."""
    
    def __init__(self):
        self.pca_models = {}
        self.cluster_models = {}
        self.target_encoders = {}
        
    def create_target_encoding(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                             cat_columns: List[str], target_col: str, 
                             n_splits: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create target encoding with cross-validation to prevent overfitting."""
        from sklearn.model_selection import KFold
        
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()
        
        for col in cat_columns:
            # K-fold target encoding for training set
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            train_encoded[f'{col}_target_enc'] = 0
            
            for train_idx, val_idx in kf.split(train_df):
                # Calculate mean target for each category in train fold
                target_means = train_df.iloc[train_idx].groupby(col)[target_col].mean()
                # Apply to validation fold
                train_encoded.loc[val_idx, f'{col}_target_enc'] = \
                    train_df.iloc[val_idx][col].map(target_means).fillna(train_df[target_col].mean())
            
            # For test set, use full training data
            target_means = train_df.groupby(col)[target_col].mean()
            test_encoded[f'{col}_target_enc'] = test_df[col].map(target_means).fillna(train_df[target_col].mean())
            
        return train_encoded, test_encoded
    
    def create_frequency_encoding(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                cat_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create frequency encoding for categorical variables."""
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()
        
        for col in cat_columns:
            # Calculate frequencies in training data
            freq_map = train_df[col].value_counts(normalize=True).to_dict()
            
            # Apply to both train and test
            train_encoded[f'{col}_freq'] = train_df[col].map(freq_map).fillna(0)
            test_encoded[f'{col}_freq'] = test_df[col].map(freq_map).fillna(0)
            
        return train_encoded, test_encoded
    
    def create_clustering_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                 numeric_cols: List[str], n_clusters: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create cluster-based features."""
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()
        
        if len(numeric_cols) < 2:
            return train_encoded, test_encoded
            
        # Fit clustering on training data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        train_clusters = kmeans.fit_predict(train_df[numeric_cols].fillna(0))
        test_clusters = kmeans.predict(test_df[numeric_cols].fillna(0))
        
        train_encoded['cluster'] = train_clusters
        test_encoded['cluster'] = test_clusters
        
        # Distance to each cluster center
        train_distances = kmeans.transform(train_df[numeric_cols].fillna(0))
        test_distances = kmeans.transform(test_df[numeric_cols].fillna(0))
        
        for i in range(n_clusters):
            train_encoded[f'dist_to_cluster_{i}'] = train_distances[:, i]
            test_encoded[f'dist_to_cluster_{i}'] = test_distances[:, i]
            
        self.cluster_models['kmeans'] = kmeans
        
        return train_encoded, test_encoded
    
    def create_pca_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                          numeric_cols: List[str], n_components: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create PCA features from numeric columns."""
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()
        
        if len(numeric_cols) < n_components:
            n_components = len(numeric_cols)
            
        # Fit PCA on training data
        pca = PCA(n_components=n_components, random_state=42)
        train_pca = pca.fit_transform(train_df[numeric_cols].fillna(0))
        test_pca = pca.transform(test_df[numeric_cols].fillna(0))
        
        # Add PCA features
        for i in range(n_components):
            train_encoded[f'pca_{i}'] = train_pca[:, i]
            test_encoded[f'pca_{i}'] = test_pca[:, i]
            
        self.pca_models['pca'] = pca
        
        return train_encoded, test_encoded
    
    def create_polynomial_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                 numeric_cols: List[str], degree: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create polynomial and interaction features."""
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()
        
        # Limit columns to prevent explosion
        if len(numeric_cols) > 10:
            # Select most important features using variance
            variances = train_df[numeric_cols].var()
            numeric_cols = variances.nlargest(10).index.tolist()
            
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        train_poly = poly.fit_transform(train_df[numeric_cols].fillna(0))
        test_poly = poly.transform(test_df[numeric_cols].fillna(0))
        
        # Get feature names
        feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Add only interaction features (not original features)
        for i, name in enumerate(feature_names):
            if ' ' in name:  # Interaction feature
                train_encoded[f'poly_{name}'] = train_poly[:, i]
                test_encoded[f'poly_{name}'] = test_poly[:, i]
                
        return train_encoded, test_encoded
    
    def create_lag_features(self, df: pd.DataFrame, group_cols: List[str], 
                          value_cols: List[str], periods: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """Create lag features for time series data."""
        df_encoded = df.copy()
        
        for group_col in group_cols:
            for value_col in value_cols:
                for period in periods:
                    df_encoded[f'{value_col}_lag_{period}_by_{group_col}'] = \
                        df.groupby(group_col)[value_col].shift(period)
                        
        return df_encoded
    
    def create_statistical_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  group_cols: List[str], numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create statistical aggregation features."""
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()
        
        for group_col in group_cols:
            for num_col in numeric_cols:
                # Calculate statistics on training data
                stats = train_df.groupby(group_col)[num_col].agg(['mean', 'std', 'min', 'max', 'median'])
                stats.columns = [f'{num_col}_{stat}_by_{group_col}' for stat in stats.columns]
                
                # Merge with train and test
                train_encoded = train_encoded.merge(stats, left_on=group_col, right_index=True, how='left')
                test_encoded = test_encoded.merge(stats, left_on=group_col, right_index=True, how='left')
                
        return train_encoded, test_encoded
    
    def apply_all_techniques(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           target_col: str, cat_cols: List[str], num_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply all advanced feature engineering techniques."""
        
        # Target encoding
        if cat_cols and target_col in train_df.columns:
            train_df, test_df = self.create_target_encoding(train_df, test_df, cat_cols, target_col)
            
        # Frequency encoding
        if cat_cols:
            train_df, test_df = self.create_frequency_encoding(train_df, test_df, cat_cols)
            
        # Clustering features
        if len(num_cols) >= 2:
            train_df, test_df = self.create_clustering_features(train_df, test_df, num_cols)
            
        # PCA features
        if len(num_cols) >= 3:
            train_df, test_df = self.create_pca_features(train_df, test_df, num_cols)
            
        # Polynomial features (be careful with this)
        if len(num_cols) <= 5:  # Only for small number of features
            train_df, test_df = self.create_polynomial_features(train_df, test_df, num_cols)
            
        return train_df, test_df
"""Automated Exploratory Data Analysis module."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')


class AutoEDA:
    """Automated Exploratory Data Analysis for Kaggle competitions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report = {}
        self.insights = []
        
    def analyze(self, train_path: str, test_path: Optional[str] = None, 
                target_col: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive EDA on the dataset."""
        print("🔍 Starting Automated EDA...")
        
        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path else None
        
        # Basic information
        self.report['basic_info'] = self._analyze_basic_info(train_df, test_df)
        
        # Identify target column if not provided
        if not target_col:
            target_col = self._identify_target_column(train_df, test_df)
        self.report['target_column'] = target_col
        
        # Analyze data types
        self.report['data_types'] = self._analyze_data_types(train_df)
        
        # Missing values analysis
        self.report['missing_values'] = self._analyze_missing_values(train_df, test_df)
        
        # Statistical summary
        self.report['statistics'] = self._analyze_statistics(train_df, target_col)
        
        # Correlation analysis
        self.report['correlations'] = self._analyze_correlations(train_df, target_col)
        
        # Distribution analysis
        self.report['distributions'] = self._analyze_distributions(train_df, target_col)
        
        # Categorical analysis
        self.report['categorical'] = self._analyze_categorical(train_df, target_col)
        
        # Generate visualizations
        self._generate_visualizations(train_df, target_col)
        
        # Generate insights
        self.insights = self._generate_insights()
        self.report['insights'] = self.insights
        
        # Save report
        self._save_report()
        
        print("✓ EDA completed successfully!")
        return self.report
    
    def _analyze_basic_info(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame]) -> Dict:
        """Analyze basic dataset information."""
        info = {
            'train_shape': train_df.shape,
            'test_shape': test_df.shape if test_df is not None else None,
            'train_memory': train_df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            'columns': list(train_df.columns),
            'dtypes_summary': {str(k): int(v) for k, v in train_df.dtypes.value_counts().items()}
        }
        
        print(f"  Train shape: {info['train_shape']}")
        if test_df is not None:
            print(f"  Test shape: {info['test_shape']}")
        
        return info
    
    def _identify_target_column(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame]) -> Optional[str]:
        """Try to identify the target column."""
        if test_df is None:
            return None
        
        # Columns in train but not in test are likely targets
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        diff_cols = train_cols - test_cols
        
        if len(diff_cols) == 1:
            target = list(diff_cols)[0]
            print(f"  Identified target column: {target}")
            return target
        
        # Common target column names
        common_targets = ['target', 'label', 'y', 'class', 'response', 'outcome']
        for col in train_df.columns:
            if col.lower() in common_targets:
                print(f"  Identified target column: {col}")
                return col
        
        return None
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict:
        """Analyze and categorize data types."""
        types = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': [],
            'id': []
        }
        
        for col in df.columns:
            # Check for ID columns
            if 'id' in col.lower() or df[col].nunique() == len(df):
                types['id'].append(col)
            # Check for datetime
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                types['datetime'].append(col)
            # Check for numeric
            elif pd.api.types.is_numeric_dtype(df[col]):
                types['numeric'].append(col)
            # Check for text (high cardinality string)
            elif df[col].dtype == 'object' and df[col].nunique() / len(df) > 0.5:
                types['text'].append(col)
            # Categorical
            else:
                types['categorical'].append(col)
        
        return types
    
    def _analyze_missing_values(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame]) -> Dict:
        """Analyze missing values in the dataset."""
        missing = {
            'train': {},
            'test': {}
        }
        
        # Train missing values
        train_missing = train_df.isnull().sum()
        train_missing_pct = (train_missing / len(train_df) * 100).round(2)
        
        for col in train_missing[train_missing > 0].index:
            missing['train'][col] = {
                'count': int(train_missing[col]),
                'percentage': float(train_missing_pct[col])
            }
        
        # Test missing values
        if test_df is not None:
            test_missing = test_df.isnull().sum()
            test_missing_pct = (test_missing / len(test_df) * 100).round(2)
            
            for col in test_missing[test_missing > 0].index:
                missing['test'][col] = {
                    'count': int(test_missing[col]),
                    'percentage': float(test_missing_pct[col])
                }
        
        # High missing columns
        high_missing = [col for col, info in missing['train'].items() 
                       if info['percentage'] > 50]
        if high_missing:
            self.insights.append(f"⚠️ High missing values (>50%) in: {', '.join(high_missing)}")
        
        return missing
    
    def _analyze_statistics(self, df: pd.DataFrame, target_col: Optional[str]) -> Dict:
        """Generate statistical summaries."""
        stats = {}
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric'] = df[numeric_cols].describe().to_dict()
            
            # Add skewness and kurtosis
            stats['skewness'] = df[numeric_cols].skew().to_dict()
            stats['kurtosis'] = df[numeric_cols].kurtosis().to_dict()
        
        # Target statistics if classification
        if target_col and target_col in df.columns:
            if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() < 10:
                stats['target_distribution'] = df[target_col].value_counts().to_dict()
                
                # Check for class imbalance
                if len(stats['target_distribution']) == 2:
                    values = list(stats['target_distribution'].values())
                    ratio = min(values) / max(values)
                    if ratio < 0.2:
                        self.insights.append(f"⚠️ Severe class imbalance detected (ratio: {ratio:.2f})")
        
        return stats
    
    def _analyze_correlations(self, df: pd.DataFrame, target_col: Optional[str]) -> Dict:
        """Analyze correlations between features."""
        correlations = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # High correlations (excluding diagonal)
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': round(corr_matrix.iloc[i, j], 3)
                        })
            
            correlations['high_correlations'] = high_corr
            
            # Target correlations
            if target_col and target_col in numeric_cols:
                target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
                correlations['target_correlations'] = {
                    'top_positive': target_corr.head(5).to_dict(),
                    'top_negative': target_corr.tail(5).to_dict()
                }
        
        return correlations
    
    def _analyze_distributions(self, df: pd.DataFrame, target_col: Optional[str]) -> Dict:
        """Analyze feature distributions."""
        distributions = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != target_col:
                distributions[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q50': float(df[col].quantile(0.50)),
                    'q75': float(df[col].quantile(0.75)),
                    'outliers_pct': float((
                        (df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) |
                        (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))
                    ).sum() / len(df) * 100)
                }
        
        return distributions
    
    def _analyze_categorical(self, df: pd.DataFrame, target_col: Optional[str]) -> Dict:
        """Analyze categorical features."""
        categorical = {}
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            if col != target_col:
                categorical[col] = {
                    'unique_values': int(df[col].nunique()),
                    'top_values': df[col].value_counts().head(10).to_dict(),
                    'missing_count': int(df[col].isnull().sum())
                }
                
                # High cardinality warning
                if df[col].nunique() > 50:
                    self.insights.append(f"⚠️ High cardinality in {col}: {df[col].nunique()} unique values")
        
        return categorical
    
    def _generate_visualizations(self, df: pd.DataFrame, target_col: Optional[str]):
        """Generate and save visualizations."""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure directory
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        # 1. Missing values heatmap
        if df.isnull().sum().sum() > 0:
            plt.figure(figsize=(12, 6))
            missing_df = df.isnull().sum().sort_values(ascending=False)
            missing_df = missing_df[missing_df > 0]
            plt.bar(range(len(missing_df)), missing_df.values)
            plt.xticks(range(len(missing_df)), missing_df.index, rotation=45, ha='right')
            plt.ylabel('Missing Count')
            plt.title('Missing Values by Column')
            plt.tight_layout()
            plt.savefig(fig_dir / 'missing_values.png', dpi=100)
            plt.close()
        
        # 2. Target distribution (if classification)
        if target_col and target_col in df.columns:
            if df[target_col].nunique() < 20:
                plt.figure(figsize=(8, 6))
                df[target_col].value_counts().plot(kind='bar')
                plt.title(f'Target Distribution: {target_col}')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(fig_dir / 'target_distribution.png', dpi=100)
                plt.close()
        
        # 3. Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlations')
            plt.tight_layout()
            plt.savefig(fig_dir / 'correlation_heatmap.png', dpi=100)
            plt.close()
        
        # 4. Numeric distributions
        if len(numeric_cols) > 0:
            n_cols = min(len(numeric_cols), 9)
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:n_cols]):
                axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
            
            # Hide unused subplots
            for i in range(n_cols, 9):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(fig_dir / 'numeric_distributions.png', dpi=100)
            plt.close()
        
        print(f"  Generated {len(list(fig_dir.glob('*.png')))} visualizations")
    
    def _generate_insights(self) -> List[str]:
        """Generate key insights from the analysis."""
        insights = self.insights.copy()
        
        # Data shape insights
        if self.report['basic_info']['train_shape'][1] > 100:
            insights.append(f"📊 High dimensional data: {self.report['basic_info']['train_shape'][1]} features")
        
        # Missing values insights
        if self.report['missing_values']['train']:
            n_missing = len(self.report['missing_values']['train'])
            insights.append(f"🔍 {n_missing} columns have missing values")
        
        # Correlation insights
        if 'high_correlations' in self.report['correlations']:
            n_high_corr = len(self.report['correlations']['high_correlations'])
            if n_high_corr > 0:
                insights.append(f"🔗 {n_high_corr} feature pairs have high correlation (>0.8)")
        
        # Skewness insights
        if 'skewness' in self.report['statistics']:
            skewed = [col for col, skew in self.report['statistics']['skewness'].items() 
                      if abs(skew) > 1]
            if skewed:
                insights.append(f"📈 {len(skewed)} features are highly skewed")
        
        return insights
    
    def _save_report(self):
        """Save the EDA report."""
        # Save JSON report
        report_path = self.output_dir / 'eda_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        
        # Save markdown summary
        summary_path = self.output_dir / 'eda_summary.md'
        with open(summary_path, 'w') as f:
            f.write("# Automated EDA Summary\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic info
            f.write("## Dataset Overview\n")
            f.write(f"- Train shape: {self.report['basic_info']['train_shape']}\n")
            if self.report['basic_info']['test_shape']:
                f.write(f"- Test shape: {self.report['basic_info']['test_shape']}\n")
            f.write(f"- Memory usage: {self.report['basic_info']['train_memory']:.2f} MB\n")
            f.write(f"- Target column: {self.report.get('target_column', 'Not identified')}\n\n")
            
            # Key insights
            f.write("## Key Insights\n")
            for insight in self.insights:
                f.write(f"- {insight}\n")
            f.write("\n")
            
            # Missing values
            if self.report['missing_values']['train']:
                f.write("## Missing Values\n")
                for col, info in sorted(self.report['missing_values']['train'].items(), 
                                       key=lambda x: x[1]['percentage'], reverse=True)[:10]:
                    f.write(f"- {col}: {info['percentage']}% ({info['count']} rows)\n")
                f.write("\n")
            
            # High correlations
            if 'high_correlations' in self.report['correlations'] and self.report['correlations']['high_correlations']:
                f.write("## High Correlations\n")
                for corr in self.report['correlations']['high_correlations'][:5]:
                    f.write(f"- {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']}\n")
                f.write("\n")
            
            f.write("## Visualizations\n")
            f.write("See the `figures/` directory for generated plots.\n")
        
        print(f"  Report saved to: {self.output_dir}")
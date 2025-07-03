"""Custom code injection system for pipeline customization."""

import ast
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List, Union
import tempfile
import subprocess
import sys
import json
from datetime import datetime
import pandas as pd


class CodeInjector:
    """Manages custom code injection for pipeline customization."""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.custom_dir = project_dir / "custom_modules"
        self.custom_dir.mkdir(exist_ok=True)
        
        self.injection_history = []
        self.loaded_modules = {}
        
        # Add custom directory to Python path
        if str(self.custom_dir) not in sys.path:
            sys.path.insert(0, str(self.custom_dir))
    
    def inject_code(self, code: str, name: str, stage: Optional[str] = None,
                   validate: bool = True) -> Dict[str, Any]:
        """Inject custom code into the pipeline."""
        print(f"\n💉 Injecting custom code: {name}")
        
        # Validate code if requested
        if validate:
            validation_result = self._validate_code(code)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'details': validation_result.get('details')
                }
        
        # Create module file
        module_filename = f"{name.replace(' ', '_').lower()}.py"
        module_path = self.custom_dir / module_filename
        
        # Add metadata to code
        metadata = f'''"""
Custom Module: {name}
Injected at: {datetime.now().isoformat()}
Stage: {stage or 'general'}
"""

'''
        
        full_code = metadata + code
        
        # Write code to file
        with open(module_path, 'w') as f:
            f.write(full_code)
        
        # Try to import the module
        try:
            module = self._import_module(module_path, name)
            self.loaded_modules[name] = module
            
            # Record injection
            self.injection_history.append({
                'name': name,
                'stage': stage,
                'timestamp': datetime.now().isoformat(),
                'module_path': str(module_path),
                'success': True
            })
            
            print(f"✓ Successfully injected: {name}")
            
            return {
                'success': True,
                'module': module,
                'path': module_path,
                'available_functions': self._get_module_functions(module)
            }
            
        except Exception as e:
            # Record failure
            self.injection_history.append({
                'name': name,
                'stage': stage,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            })
            
            return {
                'success': False,
                'error': f"Failed to load module: {str(e)}"
            }
    
    def inject_feature_engineering(self, code: str, name: str = "custom_features") -> Dict[str, Any]:
        """Inject custom feature engineering code."""
        template = '''
def engineer_features(df, is_train=True):
    """
    Custom feature engineering function.
    
    Args:
        df: Input dataframe
        is_train: Whether this is training data
    
    Returns:
        df: Dataframe with new features
    """
'''
        
        full_code = template + "\n" + code
        return self.inject_code(full_code, name, stage='feature_engineering')
    
    def inject_model(self, code: str, name: str = "custom_model") -> Dict[str, Any]:
        """Inject custom model code."""
        template = '''
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy as np

class CustomModel(BaseEstimator):
    """Custom model implementation."""
    
    def __init__(self, **params):
        self.params = params
        
'''
        
        full_code = template + "\n" + code
        return self.inject_code(full_code, name, stage='modeling')
    
    def inject_preprocessing(self, code: str, name: str = "custom_preprocessing") -> Dict[str, Any]:
        """Inject custom preprocessing code."""
        template = '''
def preprocess_data(df, config=None):
    """
    Custom preprocessing function.
    
    Args:
        df: Input dataframe
        config: Optional configuration dict
    
    Returns:
        df: Preprocessed dataframe
    """
'''
        
        full_code = template + "\n" + code
        return self.inject_code(full_code, name, stage='preprocessing')
    
    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Validate Python code for syntax and safety."""
        try:
            # Check syntax
            ast.parse(code)
            
            # Check for dangerous imports/operations
            tree = ast.parse(code)
            dangerous_imports = ['os', 'subprocess', 'eval', 'exec', '__import__']
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_imports:
                            return {
                                'valid': False,
                                'error': f"Dangerous import detected: {alias.name}",
                                'details': "For security reasons, certain modules are restricted"
                            }
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in dangerous_imports:
                        return {
                            'valid': False,
                            'error': f"Dangerous import detected: {node.module}",
                            'details': "For security reasons, certain modules are restricted"
                        }
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile']:
                            return {
                                'valid': False,
                                'error': f"Dangerous function detected: {node.func.id}",
                                'details': "Dynamic code execution is not allowed"
                            }
            
            return {'valid': True}
            
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"Syntax error: {str(e)}",
                'details': f"Line {e.lineno}: {e.text}" if e.text else None
            }
    
    def _import_module(self, module_path: Path, name: str):
        """Import a module from file."""
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def _get_module_functions(self, module) -> List[str]:
        """Get list of functions in a module."""
        functions = []
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                functions.append(name)
        return functions
    
    def execute_custom_function(self, module_name: str, function_name: str, 
                              *args, **kwargs) -> Any:
        """Execute a function from an injected module."""
        if module_name not in self.loaded_modules:
            raise ValueError(f"Module not found: {module_name}")
        
        module = self.loaded_modules[module_name]
        
        if not hasattr(module, function_name):
            raise ValueError(f"Function not found: {function_name}")
        
        func = getattr(module, function_name)
        return func(*args, **kwargs)
    
    def create_custom_module_template(self, module_type: str) -> str:
        """Create a template for custom module development."""
        templates = {
            'feature_engineering': '''"""Custom Feature Engineering Module"""

import pandas as pd
import numpy as np

def create_interaction_features(df):
    """Create interaction features between numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    new_features = pd.DataFrame()
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            # Multiplication interaction
            new_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            
            # Division interaction (with zero handling)
            new_features[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
    
    return pd.concat([df, new_features], axis=1)

def create_statistical_features(df, group_col):
    """Create statistical features based on grouping."""
    if group_col not in df.columns:
        return df
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col != group_col:
            # Group statistics
            df[f'{col}_mean_by_{group_col}'] = df.groupby(group_col)[col].transform('mean')
            df[f'{col}_std_by_{group_col}'] = df.groupby(group_col)[col].transform('std')
            df[f'{col}_max_by_{group_col}'] = df.groupby(group_col)[col].transform('max')
            df[f'{col}_min_by_{group_col}'] = df.groupby(group_col)[col].transform('min')
    
    return df

def engineer_features(df, is_train=True):
    """Main feature engineering function."""
    # Add your custom feature engineering here
    df = create_interaction_features(df)
    
    # Example: Create statistical features if a categorical column exists
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = create_statistical_features(df, categorical_cols[0])
    
    return df
''',
            'modeling': '''"""Custom Model Module"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np

class CustomEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Custom ensemble classifier combining multiple models."""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []
        
    def fit(self, X, y):
        """Fit the ensemble model."""
        # Model 1: Random Forest
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            random_state=self.random_state
        )
        rf.fit(X, y)
        self.models.append(('rf', rf, 0.5))
        
        # Model 2: Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=self.n_estimators // 2,
            max_depth=5,
            random_state=self.random_state
        )
        gb.fit(X, y)
        self.models.append(('gb', gb, 0.5))
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        predictions = []
        weights = []
        
        for name, model, weight in self.models:
            pred = model.predict_proba(X)
            predictions.append(pred)
            weights.append(weight)
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_pred += pred * weight
        
        return weighted_pred
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

def get_model(**params):
    """Factory function to create model instance."""
    return CustomEnsembleClassifier(**params)
''',
            'preprocessing': '''"""Custom Preprocessing Module"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

def handle_outliers(df, columns=None, method='iqr', threshold=1.5):
    """Handle outliers in numeric columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Clip outliers
            df_clean[col] = df[col].clip(lower_bound, upper_bound)
        
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            # Clip outliers
            df_clean[col] = df[col].clip(lower_bound, upper_bound)
    
    return df_clean

def scale_features(df, method='standard', exclude_cols=None):
    """Scale numeric features."""
    if exclude_cols is None:
        exclude_cols = []
    
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    
    df_scaled = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        return df_scaled
    
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df_scaled

def preprocess_data(df, config=None):
    """Main preprocessing function."""
    if config is None:
        config = {}
    
    # Handle outliers
    df = handle_outliers(
        df, 
        method=config.get('outlier_method', 'iqr'),
        threshold=config.get('outlier_threshold', 1.5)
    )
    
    # Scale features
    df = scale_features(
        df,
        method=config.get('scaling_method', 'standard'),
        exclude_cols=config.get('exclude_from_scaling', [])
    )
    
    return df
'''
        }
        
        return templates.get(module_type, "# Custom module template not found")
    
    def save_injection_history(self):
        """Save injection history to file."""
        history_file = self.custom_dir / "injection_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.injection_history, f, indent=2)
    
    def list_custom_modules(self) -> List[Dict[str, Any]]:
        """List all available custom modules."""
        modules = []
        
        for module_file in self.custom_dir.glob("*.py"):
            if module_file.name.startswith('__'):
                continue
            
            module_info = {
                'name': module_file.stem,
                'path': str(module_file),
                'size': module_file.stat().st_size,
                'modified': datetime.fromtimestamp(module_file.stat().st_mtime).isoformat()
            }
            
            # Try to get module docstring
            try:
                with open(module_file, 'r') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    docstring = ast.get_docstring(tree)
                    if docstring:
                        module_info['description'] = docstring.split('\n')[0]
            except:
                pass
            
            modules.append(module_info)
        
        return modules
    
    def test_custom_module(self, module_name: str, test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Test a custom module with sample data."""
        if module_name not in self.loaded_modules:
            return {'success': False, 'error': 'Module not loaded'}
        
        module = self.loaded_modules[module_name]
        results = {'success': True, 'tests': []}
        
        # Test each function in the module
        for func_name in self._get_module_functions(module):
            func = getattr(module, func_name)
            
            try:
                # Create test data if not provided
                if test_data is None:
                    test_data = pd.DataFrame({
                        'feature1': np.random.randn(100),
                        'feature2': np.random.randn(100),
                        'feature3': np.random.choice(['A', 'B', 'C'], 100),
                        'target': np.random.choice([0, 1], 100)
                    })
                
                # Test the function
                if 'df' in inspect.signature(func).parameters:
                    result = func(test_data.copy())
                    results['tests'].append({
                        'function': func_name,
                        'success': True,
                        'output_shape': result.shape if hasattr(result, 'shape') else None
                    })
                else:
                    results['tests'].append({
                        'function': func_name,
                        'skipped': True,
                        'reason': 'Function signature not compatible with test'
                    })
                    
            except Exception as e:
                results['tests'].append({
                    'function': func_name,
                    'success': False,
                    'error': str(e)
                })
        
        return results
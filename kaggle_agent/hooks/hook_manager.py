"""Hook system for pipeline intervention points."""

import inspect
import importlib.util
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List
import json
import yaml
from datetime import datetime


class HookManager:
    """Manages hooks for pipeline intervention."""
    
    def __init__(self, hooks_dir: Path):
        self.hooks_dir = hooks_dir
        self.hooks_dir.mkdir(exist_ok=True)
        self.registered_hooks = {}
        self.hook_history = []
        
        # Create default hook templates
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default hook template files."""
        templates = {
            'after_download_data.py': '''"""Hook executed after data download."""

def hook(pipeline, context):
    """
    Args:
        pipeline: The pipeline instance
        context: Hook context with stage info and results
    
    Returns:
        dict: Modifications to apply (optional)
    """
    print("🔗 Custom hook: after_download_data")
    
    # Example: Print downloaded files
    if 'data_files' in pipeline.results:
        files = pipeline.results['data_files']
        print(f"  Downloaded files: {list(files.keys())}")
    
    # Example: Add custom data preprocessing
    # return {
    #     'add_preprocessing': True,
    #     'preprocessing_steps': ['remove_outliers', 'normalize']
    # }
    
    return {}
''',
            'after_eda.py': '''"""Hook executed after EDA."""

def hook(pipeline, context):
    """
    Args:
        pipeline: The pipeline instance
        context: Hook context with stage info and results
    
    Returns:
        dict: Modifications to apply (optional)
    """
    print("🔗 Custom hook: after_eda")
    
    # Example: Analyze EDA results
    if 'eda_report' in pipeline.results:
        report = pipeline.results['eda_report']
        missing_cols = len(report.get('missing_values', {}).get('train', []))
        print(f"  Found {missing_cols} columns with missing values")
    
    # Example: Override feature engineering settings based on EDA
    # if missing_cols > 10:
    #     return {
    #         'feature_engineering': {
    #             'aggressive_imputation': True
    #         }
    #     }
    
    return {}
''',
            'after_feature_engineering.py': '''"""Hook executed after feature engineering."""

def hook(pipeline, context):
    """
    Args:
        pipeline: The pipeline instance
        context: Hook context with stage info and results
    
    Returns:
        dict: Modifications to apply (optional)
    """
    print("🔗 Custom hook: after_feature_engineering")
    
    # Example: Add custom features
    # processed_dir = pipeline.project_dir / "data" / "processed"
    # train_df = pd.read_csv(processed_dir / "train_processed.csv")
    # 
    # # Add your custom features here
    # train_df['custom_feature'] = train_df['feature1'] * train_df['feature2']
    # 
    # train_df.to_csv(processed_dir / "train_processed.csv", index=False)
    
    return {}
''',
            'after_modeling.py': '''"""Hook executed after model training."""

def hook(pipeline, context):
    """
    Args:
        pipeline: The pipeline instance
        context: Hook context with stage info and results
    
    Returns:
        dict: Modifications to apply (optional)
    """
    print("🔗 Custom hook: after_modeling")
    
    # Example: Analyze model results
    if 'modeling' in pipeline.results:
        best_model = pipeline.results['modeling'].get('best_model')
        best_score = pipeline.results['modeling'].get('best_score')
        print(f"  Best model: {best_model} (Score: {best_score})")
    
    # Example: Force retraining with different parameters
    # if best_score < 0.85:
    #     return {
    #         'retrain': True,
    #         'algorithms': ['xgboost', 'catboost'],
    #         'optimize_hyperparameters': True
    #     }
    
    return {}
''',
            'before_submission.py': '''"""Hook executed before submission generation."""

def hook(pipeline, context):
    """
    Args:
        pipeline: The pipeline instance
        context: Hook context with stage info and results
    
    Returns:
        dict: Modifications to apply (optional)
    """
    print("🔗 Custom hook: before_submission")
    
    # Example: Verify predictions before submission
    # model_output_dir = pipeline.project_dir / "model_output"
    # if (model_output_dir / "test_predictions.csv").exists():
    #     pred_df = pd.read_csv(model_output_dir / "test_predictions.csv")
    #     print(f"  Prediction shape: {pred_df.shape}")
    #     print(f"  Prediction range: [{pred_df.values.min():.4f}, {pred_df.values.max():.4f}]")
    
    # Example: Apply post-processing
    # return {
    #     'post_processing': 'clip_predictions',
    #     'clip_range': [0, 1]
    # }
    
    return {}
'''
        }
        
        # Create template files if they don't exist
        for filename, content in templates.items():
            filepath = self.hooks_dir / filename
            if not filepath.exists():
                filepath.write_text(content)
    
    def register_hook(self, stage: str, hook_func: Callable, priority: int = 0):
        """Register a hook function for a specific stage."""
        if stage not in self.registered_hooks:
            self.registered_hooks[stage] = []
        
        self.registered_hooks[stage].append({
            'function': hook_func,
            'priority': priority,
            'registered_at': datetime.now().isoformat()
        })
        
        # Sort by priority (higher priority first)
        self.registered_hooks[stage].sort(key=lambda x: x['priority'], reverse=True)
    
    def load_hook_from_file(self, stage: str, filepath: Path):
        """Load and register a hook from a Python file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Hook file not found: {filepath}")
        
        # Load the module
        spec = importlib.util.spec_from_file_location(f"hook_{stage}", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the hook function
        if hasattr(module, 'hook'):
            self.register_hook(stage, module.hook)
            print(f"  ✓ Loaded hook from: {filepath}")
        else:
            raise ValueError(f"Hook file must contain a 'hook' function: {filepath}")
    
    def load_hooks_from_config(self, config: Dict[str, Any]):
        """Load hooks specified in configuration."""
        hooks_config = config.get('hooks', {})
        
        for stage, hook_info in hooks_config.items():
            if hook_info and isinstance(hook_info, dict):
                # Hook specified in config
                if 'file' in hook_info:
                    # Validate file path security
                    try:
                        filepath = self.hooks_dir / hook_info['file']
                        # Ensure the file is within the hooks directory
                        filepath = filepath.resolve()
                        hooks_dir_resolved = self.hooks_dir.resolve()
                        
                        if not str(filepath).startswith(str(hooks_dir_resolved)):
                            print(f"  ❌ Security: Hook file path outside hooks directory: {hook_info['file']}")
                            continue
                        
                        if filepath.exists():
                            self.load_hook_from_file(stage, filepath)
                        else:
                            print(f"  ⚠️  Hook file not found: {filepath}")
                    except Exception as e:
                        print(f"  ❌ Error loading hook file: {str(e)}")
                
                # Inline hook code (advanced usage)
                elif 'code' in hook_info:
                    # Validate code safety before execution
                    try:
                        import ast
                        tree = ast.parse(hook_info['code'])
                        
                        # Check for dangerous operations
                        for node in ast.walk(tree):
                            # Check for dangerous imports
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    if alias.name in ['os', 'subprocess', 'eval', 'exec', '__import__']:
                                        print(f"  ❌ Security: Dangerous import '{alias.name}' not allowed")
                                        continue
                            
                            # Check for dangerous function calls
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                                if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                                    print(f"  ❌ Security: Dangerous function '{node.func.id}' not allowed")
                                    continue
                        
                        # Execute in restricted namespace
                        restricted_globals = {
                            '__builtins__': {
                                'print': print,
                                'len': len,
                                'range': range,
                                'str': str,
                                'int': int,
                                'float': float,
                                'list': list,
                                'dict': dict,
                                'set': set,
                                'tuple': tuple,
                                'bool': bool,
                                'isinstance': isinstance,
                                'hasattr': hasattr,
                                'getattr': getattr,
                                'setattr': setattr,
                                'min': min,
                                'max': max,
                                'sum': sum,
                                'sorted': sorted,
                                'enumerate': enumerate,
                                'zip': zip,
                                'map': map,
                                'filter': filter,
                            }
                        }
                        
                        # Execute the code
                        exec(hook_info['code'], restricted_globals)
                        if 'hook' in restricted_globals:
                            self.register_hook(stage, restricted_globals['hook'])
                    
                    except SyntaxError as e:
                        print(f"  ❌ Syntax error in hook code: {str(e)}")
                    except Exception as e:
                        print(f"  ❌ Error executing hook code: {str(e)}")
    
    def execute_hooks(self, stage: str, pipeline: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute all hooks for a specific stage."""
        if stage not in self.registered_hooks:
            # Try to load from file if exists
            hook_file = self.hooks_dir / f"{stage}.py"
            if hook_file.exists():
                self.load_hook_from_file(stage, hook_file)
            else:
                return {}
        
        # Prepare context
        context = context or {}
        context.update({
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'pipeline_state': pipeline.state,
            'results': pipeline.results
        })
        
        # Execute hooks in priority order
        all_modifications = {}
        
        for hook_info in self.registered_hooks.get(stage, []):
            try:
                print(f"\n  🔗 Executing hook (priority: {hook_info['priority']})")
                
                # Execute hook
                modifications = hook_info['function'](pipeline, context)
                
                if modifications:
                    print(f"    → Hook returned modifications: {list(modifications.keys())}")
                    all_modifications.update(modifications)
                
                # Record in history
                self.hook_history.append({
                    'stage': stage,
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'modifications': modifications
                })
                
            except Exception as e:
                print(f"    ❌ Hook error: {str(e)}")
                self.hook_history.append({
                    'stage': stage,
                    'timestamp': datetime.now().isoformat(),
                    'success': False,
                    'error': str(e)
                })
        
        return all_modifications
    
    def create_custom_hook(self, stage: str, description: str = "") -> Path:
        """Create a custom hook file for user editing."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"custom_{stage}_{timestamp}.py"
        filepath = self.hooks_dir / filename
        
        template = f'''"""Custom hook for {stage}
{description}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import pandas as pd
import numpy as np
from pathlib import Path

def hook(pipeline, context):
    """
    Custom hook implementation.
    
    Args:
        pipeline: The pipeline instance with access to:
            - pipeline.project_dir: Project directory path
            - pipeline.results: Current pipeline results
            - pipeline.state: Current pipeline state
            - pipeline.config: Pipeline configuration
        
        context: Hook context containing:
            - stage: Current stage name
            - timestamp: Execution timestamp
            - pipeline_state: Full pipeline state
            - results: All results so far
    
    Returns:
        dict: Modifications to apply (optional)
            Can include directives like:
            - 'retrain': True/False
            - 'skip_next_stage': True/False
            - 'modify_config': dict with config changes
            - 'add_features': list of features
            - etc.
    """
    print(f"🔗 Executing custom hook for {stage}")
    
    # Your custom code here
    # Example: Print current state
    print(f"  Current stage: {context['stage']}")
    print(f"  Results available: {list(pipeline.results.keys())}")
    
    # Example: Access data
    # data_dir = pipeline.project_dir / "data"
    # processed_dir = data_dir / "processed"
    
    # Example: Modify pipeline behavior
    # return {{
    #     'message': 'Custom hook executed successfully'
    # }}
    
    return {{}}
'''
        
        filepath.write_text(template)
        print(f"✓ Created custom hook file: {filepath}")
        return filepath
    
    def save_hook_history(self, output_path: Path):
        """Save hook execution history."""
        with open(output_path, 'w') as f:
            json.dump(self.hook_history, f, indent=2)
    
    def interactive_hook_editor(self, stage: str):
        """Open interactive hook editor (placeholder for future GUI)."""
        print(f"\n📝 Hook Editor for: {stage}")
        print("Options:")
        print("1. Create new custom hook")
        print("2. Edit existing hook")
        print("3. View hook examples")
        print("4. Test hook")
        
        choice = input("\nSelect option (1-4): ")
        
        if choice == '1':
            description = input("Enter hook description: ")
            filepath = self.create_custom_hook(stage, description)
            print(f"\nEdit the file: {filepath}")
            print("Then reload the pipeline to use the new hook.")
        
        elif choice == '2':
            # List existing hooks
            hooks = list(self.hooks_dir.glob(f"*{stage}*.py"))
            if hooks:
                print("\nExisting hooks:")
                for i, hook in enumerate(hooks):
                    print(f"{i+1}. {hook.name}")
                
                idx = int(input("\nSelect hook to edit: ")) - 1
                if 0 <= idx < len(hooks):
                    print(f"\nEdit the file: {hooks[idx]}")
            else:
                print("No existing hooks found.")
        
        elif choice == '3':
            # Show example
            example_file = self.hooks_dir / f"{stage}.py"
            if example_file.exists():
                print(f"\nExample hook for {stage}:")
                print("=" * 60)
                print(example_file.read_text())
                print("=" * 60)
        
        elif choice == '4':
            print("Hook testing not yet implemented.")
    
    def get_available_hooks(self) -> Dict[str, List[str]]:
        """Get list of available hooks by stage."""
        available = {}
        
        # Scan hook files
        for hook_file in self.hooks_dir.glob("*.py"):
            # Extract stage from filename
            for stage in ['after_download_data', 'after_eda', 'after_feature_engineering', 
                         'after_modeling', 'before_submission']:
                if stage in hook_file.stem:
                    if stage not in available:
                        available[stage] = []
                    available[stage].append(hook_file.name)
        
        return available
"""Main pipeline controller for Kaggle Agent."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import json
import yaml
from datetime import datetime
import traceback

from .core import CompetitionManager, ProjectManager, CompetitionAuth
from .core.experiment_tracker import ExperimentTracker
from .core.state_manager import StateManager
from .core.code_injector import CodeInjector
from .hooks.hook_manager import HookManager
from .modules import AutoEDA, AutoFeatureEngineering, AutoModeling, AutoSubmission


class KagglePipeline:
    """Main pipeline controller that orchestrates the entire ML workflow."""
    
    def __init__(self, project_dir: Path, config: Dict[str, Any]):
        self.project_dir = Path(project_dir)
        self.config = config
        self.state = {}
        self.hooks = {}
        self.results = {}
        self.experiment_id = None
        self.current_run_id = None
        
        # Initialize project manager
        self.project_manager = ProjectManager(
            project_name=self.project_dir.name,
            base_dir=self.project_dir.parent
        )
        
        # Initialize experiment tracker
        self.tracker = ExperimentTracker(self.project_dir)
        
        # Initialize state manager
        self.state_manager = StateManager(self.project_dir)
        
        # Initialize hook manager
        self.hook_manager = HookManager(self.project_dir / "hooks")
        self.hook_manager.load_hooks_from_config(config)
        
        # Initialize code injector
        self.code_injector = CodeInjector(self.project_dir)
        
    def run(self, competition_name: str, 
            mode: str = 'full-auto',
            resume_from: Optional[str] = None,
            stages: Optional[List[str]] = None,
            experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """Run the ML pipeline."""
        
        print(f"\n🚀 Starting Kaggle Agent Pipeline")
        print(f"   Competition: {competition_name}")
        print(f"   Mode: {mode}")
        print(f"   Project: {self.project_dir}\n")
        
        # Create experiment
        experiment_name = experiment_name or f"{competition_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_id = self.tracker.create_experiment(
            name=experiment_name,
            competition=competition_name,
            config={
                'mode': mode,
                'stages': stages or 'all',
                'resume_from': resume_from,
                'pipeline_config': self.config
            }
        )
        
        # Load state if resuming
        if resume_from:
            self._load_state()
            print(f"📂 Resuming from: {resume_from}\n")
        
        # Define pipeline stages
        all_stages = [
            'download_data',
            'eda',
            'feature_engineering',
            'modeling',
            'submission'
        ]
        
        # Use specified stages or all
        stages_to_run = stages or all_stages
        
        # Skip completed stages if resuming
        if resume_from and resume_from in all_stages:
            start_idx = all_stages.index(resume_from)
            stages_to_run = all_stages[start_idx:]
        
        # Run each stage
        for stage in stages_to_run:
            if self.config['pipeline']['stages'].get(stage, {}).get('enabled', True):
                print(f"\n{'='*60}")
                print(f"📍 Stage: {stage.upper()}")
                print(f"{'='*60}\n")
                
                # Update state
                self._update_state({'current_stage': stage})
                
                # Start run tracking
                self.current_run_id = self.tracker.start_run(self.experiment_id, stage)
                
                # Execute hooks
                hook_stage = f"before_{stage}" if stage != 'submission' else "before_submission"
                if hook_stage in self.config.get('hooks', {}):
                    modifications = self.hook_manager.execute_hooks(hook_stage, self)
                    if modifications:
                        self._apply_hook_modifications(modifications)
                
                stage_start_time = datetime.now()
                stage_metrics = {}
                
                try:
                    # Execute stage
                    if stage == 'download_data':
                        self._run_download_data(competition_name)
                    elif stage == 'eda':
                        self._run_eda()
                    elif stage == 'feature_engineering':
                        self._run_feature_engineering()
                    elif stage == 'modeling':
                        self._run_modeling()
                    elif stage == 'submission':
                        self._run_submission()
                    
                    # Calculate stage duration
                    stage_duration = (datetime.now() - stage_start_time).total_seconds()
                    stage_metrics['duration_seconds'] = stage_duration
                    
                    # Complete run tracking
                    self.tracker.complete_run(self.current_run_id, metrics=stage_metrics)
                    
                    # Mark stage as completed
                    self._update_state({
                        'completed_stages': self.state.get('completed_stages', []) + [stage]
                    })
                    
                except Exception as e:
                    print(f"\n❌ Error in stage {stage}: {str(e)}")
                    # Mark run as failed
                    self.tracker.complete_run(
                        self.current_run_id, 
                        status='failed',
                        logs=str(e)
                    )
                    if mode == 'interactive':
                        action = input("\nContinue anyway? (y/n): ")
                        if action.lower() != 'y':
                            self.tracker.complete_experiment(
                                self.experiment_id,
                                results=self.results,
                                status='failed'
                            )
                            raise
                    else:
                        self.tracker.complete_experiment(
                            self.experiment_id,
                            results=self.results,
                            status='failed'
                        )
                        raise
                
                # Execute post-stage hooks
                hook_key = f"after_{stage}"
                modifications = self.hook_manager.execute_hooks(hook_key, self)
                if modifications:
                    self._apply_hook_modifications(modifications)
                
                # Auto-checkpoint if enabled
                if self.state_manager.should_auto_checkpoint(stage):
                    self.state_manager.create_checkpoint(stage)
                
                # Save state after each stage
                self._save_state()
                self.state_manager.save_stage_result(stage, self.results.get(stage, {}))
        
        print(f"\n{'='*60}")
        print(f"✅ Pipeline completed successfully!")
        print(f"{'='*60}\n")
        
        # Complete experiment tracking
        final_metrics = self._calculate_final_metrics()
        self.tracker.log_metrics(self.experiment_id, final_metrics)
        self.tracker.complete_experiment(self.experiment_id, self.results)
        
        # Generate final report
        self._generate_final_report()
        
        # Generate experiment report
        exp_report = self.tracker.generate_experiment_report(self.experiment_id)
        exp_report_path = self.project_dir / f"experiment_report_{self.experiment_id}.md"
        with open(exp_report_path, 'w') as f:
            f.write(exp_report)
        print(f"📊 Experiment report saved to: {exp_report_path}")
        
        return self.results
    
    def _run_download_data(self, competition_name: str):
        """Download competition data."""
        comp_manager = CompetitionManager(
            competition_name=competition_name,
            data_dir=self.project_dir / "data"
        )
        
        # Get competition info
        comp_info = comp_manager.get_competition_info()
        self.results['competition_info'] = comp_info
        
        # Download data
        downloaded_files = comp_manager.download_data(handle_auth=True)
        
        # Identify data files
        data_files = comp_manager.identify_data_files()
        self.results['data_files'] = data_files
        
        print(f"\n✓ Downloaded {len(downloaded_files)} files")
        
        # Log artifacts
        for file in downloaded_files:
            self.tracker.log_artifact(
                self.experiment_id,
                name=Path(file).name,
                artifact_type='data',
                path=Path(file),
                run_id=self.current_run_id
            )
    
    def _run_eda(self):
        """Run exploratory data analysis."""
        eda = AutoEDA(output_dir=self.project_dir / "eda_output")
        
        data_files = self.results.get('data_files', {})
        target_col = self._infer_target_column()
        
        eda_report = eda.analyze(
            train_path=data_files['train'],
            test_path=data_files.get('test'),
            target_col=target_col
        )
        
        self.results['eda_report'] = eda_report
        self.results['target_column'] = eda_report.get('target_column')
        
        print(f"\n✓ EDA completed. Found {len(eda_report.get('insights', []))} insights")
        
        # Log EDA metrics
        eda_metrics = {
            'n_features': eda_report['basic_info']['train_shape'][1],
            'n_samples_train': eda_report['basic_info']['train_shape'][0],
            'n_insights': len(eda_report.get('insights', [])),
            'missing_value_columns': len(eda_report['missing_values']['train'])
        }
        self.tracker.log_metrics(self.experiment_id, eda_metrics)
    
    def _run_feature_engineering(self):
        """Run feature engineering."""
        # Load data
        data_files = self.results.get('data_files', {})
        train_df = pd.read_csv(data_files['train'])
        test_df = pd.read_csv(data_files.get('test')) if 'test' in data_files else None
        
        fe = AutoFeatureEngineering(
            output_dir=self.project_dir / "feature_output"
        )
        
        train_processed, test_processed = fe.engineer_features(
            train_df=train_df,
            test_df=test_df,
            target_col=self.results.get('target_column'),
            eda_report=self.results.get('eda_report')
        )
        
        # Save processed data
        processed_dir = self.project_dir / "data" / "processed"
        processed_dir.mkdir(exist_ok=True, parents=True)
        
        train_processed.to_csv(processed_dir / "train_processed.csv", index=False)
        if test_processed is not None:
            test_processed.to_csv(processed_dir / "test_processed.csv", index=False)
        
        self.results['feature_engineering'] = {
            'n_features_created': len(fe.generated_features),
            'final_features': train_processed.shape[1]
        }
        
        print(f"\n✓ Created {len(fe.generated_features)} new features")
        
        # Log feature engineering metrics
        fe_metrics = {
            'n_features_original': train_df.shape[1],
            'n_features_engineered': len(fe.generated_features),
            'n_features_final': train_processed.shape[1]
        }
        self.tracker.log_metrics(self.experiment_id, fe_metrics)
    
    def _run_modeling(self):
        """Run model training."""
        # Load processed data
        processed_dir = self.project_dir / "data" / "processed"
        train_df = pd.read_csv(processed_dir / "train_processed.csv")
        test_df = None
        if (processed_dir / "test_processed.csv").exists():
            test_df = pd.read_csv(processed_dir / "test_processed.csv")
        
        # Apply custom preprocessing if available
        if 'custom_preprocessing' in self.code_injector.loaded_modules:
            print("  Applying custom preprocessing...")
            train_df = self.code_injector.execute_custom_function(
                'custom_preprocessing', 'preprocess_data', train_df
            )
            if test_df is not None:
                test_df = self.code_injector.execute_custom_function(
                    'custom_preprocessing', 'preprocess_data', test_df
                )
        
        # Prepare data
        target_col = self.results.get('target_column')
        if target_col and target_col in train_df.columns:
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df
            
            # Remove string columns
            string_cols = X_train.select_dtypes(include=['object']).columns
            if len(string_cols) > 0:
                X_train = X_train.drop(columns=string_cols)
                if X_test is not None:
                    X_test = X_test.drop(columns=string_cols)
            
            modeler = AutoModeling(
                output_dir=self.project_dir / "model_output",
                competition_info=self.results.get('competition_info')
            )
            
            # Get algorithms from config
            modeling_config = self.config['pipeline']['stages']['modeling']
            algorithms = modeling_config.get('algorithms')
            cv_folds = modeling_config.get('cv_folds', 5)
            # Check both locations for hyperparameter optimization setting
            optimize_hyperparams = modeling_config.get('optimize_hyperparameters', False)
            if not optimize_hyperparams and 'optimization' in self.config:
                optimize_hyperparams = self.config['optimization'].get('hyperparameter_tuning', {}).get('enabled', False)
            optimization_trials = modeling_config.get('optimization_trials', 50)
            create_ensemble = modeling_config.get('create_ensemble', True)
            ensemble_methods = modeling_config.get('ensemble_methods', ['voting'])
            
            model_results = modeler.train_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                cv_folds=cv_folds,
                algorithms=algorithms,
                optimize_hyperparameters=optimize_hyperparams,
                optimization_trials=optimization_trials,
                create_ensemble=create_ensemble,
                ensemble_methods=ensemble_methods
            )
            
            self.results['modeling'] = {
                'best_model': modeler.best_model,
                'best_score': modeler.best_score,
                'predictions_available': bool(modeler.predictions)
            }
            
            print(f"\n✓ Best model: {modeler.best_model} (Score: {modeler.best_score:.4f})")
            
            # Log modeling metrics
            modeling_metrics = {
                'best_model': modeler.best_model,
                'best_cv_score': float(modeler.best_score),
                'n_models_trained': len(model_results)
            }
            self.tracker.log_metrics(self.experiment_id, modeling_metrics)
            
            # Log model artifacts
            for file in (self.project_dir / "model_output").glob("*.pkl"):
                self.tracker.log_artifact(
                    self.experiment_id,
                    name=file.name,
                    artifact_type='model',
                    path=file,
                    run_id=self.current_run_id
                )
    
    def _run_submission(self):
        """Generate submission file."""
        # Load test data for IDs
        data_files = self.results.get('data_files', {})
        test_df = pd.read_csv(data_files['test'])
        
        # Determine ID column - check for common ID column names
        id_col = None
        for col_name in ['PassengerId', 'Id', 'ID', 'id']:
            if col_name in test_df.columns:
                id_col = col_name
                break
        
        # If not found, use first column
        if id_col is None:
            id_col = test_df.columns[0]
            
        test_ids = test_df[id_col]
        test_ids.name = id_col  # Preserve the column name
        
        # Load predictions
        model_output_dir = self.project_dir / "model_output"
        predictions_file = model_output_dir / "test_predictions.csv"
        
        if predictions_file.exists():
            pred_df = pd.read_csv(predictions_file)
            predictions = pred_df.to_dict('series')
        else:
            # Get from modeling results
            raise ValueError("No predictions found")
        
        submitter = AutoSubmission(
            output_dir=self.project_dir / "submissions"
        )
        
        # Get task type from modeling
        model_results_file = model_output_dir / "model_results.json"
        task_type = 'classification'
        if model_results_file.exists():
            with open(model_results_file, 'r') as f:
                model_info = json.load(f)
                task_type = model_info.get('task_type', 'classification')
        
        submission_df = submitter.generate_submission(
            test_ids=test_ids,
            predictions=predictions,
            sample_submission_path=data_files.get('sample_submission'),
            competition_info=self.results.get('competition_info'),
            task_type=task_type
        )
        
        # Create report
        submitter.create_submission_report(
            model_results=model_info if 'model_info' in locals() else None
        )
        
        self.results['submission'] = {
            'file_created': True,
            'shape': submission_df.shape
        }
        
        print(f"\n✓ Submission file created: {submission_df.shape}")
        
        # Log submission artifact
        submission_file = self.project_dir / "submissions" / "submission_latest.csv"
        if submission_file.exists():
            self.tracker.log_artifact(
                self.experiment_id,
                name="submission_latest.csv",
                artifact_type='submission',
                path=submission_file,
                run_id=self.current_run_id,
                metadata={'shape': submission_df.shape}
            )
    
    def _infer_target_column(self) -> Optional[str]:
        """Infer target column from competition info."""
        comp_info = self.results.get('competition_info', {})
        
        # Common patterns based on evaluation metric
        metric = comp_info.get('evaluation_metric', '').lower()
        
        if 'accuracy' in metric:
            # Classification tasks
            common_targets = ['target', 'label', 'class', 'y']
            
            # Check specific competitions
            if 'titanic' in comp_info.get('name', ''):
                return 'Survived'
            elif 'spaceship' in comp_info.get('name', ''):
                return 'Transported'
                
        elif 'rmse' in metric or 'mae' in metric:
            # Regression tasks
            if 'house' in comp_info.get('name', ''):
                return 'SalePrice'
        
        return None
    
    def _update_state(self, updates: Dict[str, Any]):
        """Update pipeline state."""
        self.state.update(updates)
        self.state['last_updated'] = datetime.now().isoformat()
        self.state_manager.update_state(updates)
    
    def _save_state(self):
        """Save pipeline state to file."""
        state_file = self.project_dir / ".pipeline_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        # Also save to state manager
        self.state_manager.save_state()
    
    def _load_state(self):
        """Load pipeline state from file."""
        state_file = self.project_dir / ".pipeline_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                self.state = json.load(f)
        # Also sync with state manager
        self.state.update(self.state_manager.current_state)
    
    def _generate_final_report(self):
        """Generate comprehensive final report."""
        report_path = self.project_dir / "final_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Kaggle Agent Pipeline Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Competition info
            if 'competition_info' in self.results:
                comp = self.results['competition_info']
                f.write("## Competition\n")
                f.write(f"- **Name**: {comp.get('title', 'Unknown')}\n")
                f.write(f"- **Metric**: {comp.get('evaluation_metric', 'Unknown')}\n")
                f.write(f"- **Teams**: {comp.get('team_count', 'Unknown')}\n\n")
            
            # Results summary
            f.write("## Pipeline Results\n")
            
            if 'eda_report' in self.results:
                eda = self.results['eda_report']
                f.write(f"- **Data Shape**: {eda['basic_info']['train_shape']}\n")
                f.write(f"- **Missing Values**: {len(eda['missing_values']['train'])} columns\n")
            
            if 'feature_engineering' in self.results:
                fe = self.results['feature_engineering']
                f.write(f"- **Features Created**: {fe['n_features_created']}\n")
                f.write(f"- **Final Features**: {fe['final_features']}\n")
            
            if 'modeling' in self.results:
                model = self.results['modeling']
                f.write(f"- **Best Model**: {model['best_model']}\n")
                f.write(f"- **CV Score**: {model['best_score']:.4f}\n")
            
            if 'submission' in self.results:
                sub = self.results['submission']
                f.write(f"- **Submission Created**: {sub['file_created']}\n")
                f.write(f"- **Submission Shape**: {sub['shape']}\n")
            
            f.write("\n## Next Steps\n")
            f.write("1. Review generated reports in each output directory\n")
            f.write("2. Submit to Kaggle and check leaderboard score\n")
            f.write("3. Iterate based on results\n")
        
        print(f"\n📄 Final report saved to: {report_path}")
        
        # Save state summary
        state_summary = self.state_manager.export_state_summary()
        state_summary_path = self.project_dir / "pipeline_state_summary.md"
        with open(state_summary_path, 'w') as f:
            f.write(state_summary)
    
    def register_hook(self, stage: str, hook_type: str, callback: Callable):
        """Register a hook for a specific stage."""
        if stage not in self.hooks:
            self.hooks[stage] = {}
        self.hooks[stage][hook_type] = callback
    
    def pause(self):
        """Pause the pipeline execution."""
        self._update_state({'status': 'paused'})
        self._save_state()
        print("\n⏸️  Pipeline paused. State saved.")
    
    def resume(self):
        """Resume pipeline execution."""
        self._load_state()
        last_stage = self.state.get('current_stage')
        print(f"\n▶️  Resuming from stage: {last_stage}")
        return last_stage
    
    def _calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate final metrics for the experiment."""
        metrics = {}
        
        if 'modeling' in self.results:
            metrics['final_best_model'] = self.results['modeling'].get('best_model')
            metrics['final_best_score'] = self.results['modeling'].get('best_score')
        
        if 'submission' in self.results:
            metrics['submission_created'] = self.results['submission'].get('file_created', False)
        
        # Count total stages completed
        metrics['stages_completed'] = len(self.state.get('completed_stages', []))
        
        return metrics
    
    def _apply_hook_modifications(self, modifications: Dict[str, Any]):
        """Apply modifications from hooks."""
        print(f"\n🔧 Applying hook modifications: {list(modifications.keys())}")
        
        # Handle specific modifications
        if 'retrain' in modifications and modifications['retrain']:
            print("  ➡️ Retraining requested by hook")
            # Mark modeling stage for re-execution
            if 'completed_stages' in self.state and 'modeling' in self.state['completed_stages']:
                self.state['completed_stages'].remove('modeling')
        
        if 'skip_next_stage' in modifications and modifications['skip_next_stage']:
            print("  ➡️ Skipping next stage as requested by hook")
            self.state['skip_next'] = True
        
        if 'modify_config' in modifications:
            print("  ➡️ Updating configuration")
            self.config.update(modifications['modify_config'])
        
        if 'create_checkpoint' in modifications and modifications['create_checkpoint']:
            print("  ➡️ Creating checkpoint")
            self.state_manager.create_checkpoint(self.state.get('current_stage', 'unknown'))
        
        # Store modifications in state
        if 'hook_modifications' not in self.state:
            self.state['hook_modifications'] = []
        self.state['hook_modifications'].append({
            'timestamp': datetime.now().isoformat(),
            'stage': self.state.get('current_stage'),
            'modifications': modifications
        })
    
    def create_checkpoint(self, name: Optional[str] = None):
        """Manually create a checkpoint."""
        checkpoint_name = name or self.state.get('current_stage', 'manual')
        return self.state_manager.create_checkpoint(checkpoint_name)
    
    def restore_checkpoint(self, checkpoint_id: str):
        """Restore from a checkpoint."""
        objects = self.state_manager.restore_checkpoint(checkpoint_id)
        self._load_state()
        
        # Restore pipeline results if available
        if 'results' in objects:
            self.results.update(objects['results'])
        
        print(f"\n✅ Restored pipeline to checkpoint: {checkpoint_id}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        return self.state_manager.list_checkpoints()
    
    def inject_custom_code(self, code: str, name: str, code_type: str = 'general') -> bool:
        """Inject custom code into the pipeline."""
        if code_type == 'feature_engineering':
            result = self.code_injector.inject_feature_engineering(code, name)
        elif code_type == 'model':
            result = self.code_injector.inject_model(code, name)
        elif code_type == 'preprocessing':
            result = self.code_injector.inject_preprocessing(code, name)
        else:
            result = self.code_injector.inject_code(code, name)
        
        return result['success']
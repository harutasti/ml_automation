"""Command-line interface for Kaggle Agent."""

import click
import sys
import yaml
import json
from pathlib import Path
from .core import KaggleAuth, ProjectManager
from .pipeline import KagglePipeline


@click.group()
def cli():
    """Kaggle Agent - Automated ML pipeline for Kaggle competitions."""
    pass


@cli.command()
@click.argument('project_name')
@click.option('--competition', '-c', required=True, help='Kaggle competition name')
@click.option('--path', '-p', help='Base directory for the project')
def init(project_name, competition, path):
    """Initialize a new Kaggle project."""
    click.echo(f"🚀 Initializing Kaggle Agent project: {project_name}")
    
    # Check Kaggle authentication
    auth = KaggleAuth()
    if not auth.setup():
        click.echo("❌ Kaggle authentication failed. Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        sys.exit(1)
    
    if not auth.validate():
        click.echo("❌ Kaggle API validation failed. Please check your credentials.")
        sys.exit(1)
    
    # Initialize project
    project = ProjectManager(project_name, path)
    if project.initialize(competition):
        click.echo(f"✅ Project initialized at: {project.base_dir}")
        click.echo("\nNext steps:")
        click.echo("1. cd " + str(project.base_dir))
        click.echo("2. kaggle-agent run --full-auto")
    else:
        click.echo("❌ Project initialization failed.")
        sys.exit(1)


@cli.command()
@click.option('--full-auto', is_flag=True, help='Run in full automation mode')
@click.option('--interactive', is_flag=True, help='Run with intervention points')
@click.option('--stage', help='Run specific stage only')
@click.option('--resume', is_flag=True, help='Resume from last checkpoint')
def run(full_auto, interactive, stage, resume):
    """Run the Kaggle automation pipeline."""
    if full_auto and interactive:
        click.echo("❌ Cannot use both --full-auto and --interactive flags")
        sys.exit(1)
    
    click.echo("🏃 Starting Kaggle Agent pipeline...")
    
    # Check if we're in a project directory
    config_path = Path("config.yaml")
    if not config_path.exists():
        click.echo("❌ Not in a Kaggle Agent project directory. Run 'kaggle-agent init' first.")
        sys.exit(1)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get project directory
    project_dir = Path.cwd()
    
    # Initialize pipeline
    pipeline = KagglePipeline(project_dir, config)
    
    # Determine mode
    mode = 'interactive' if interactive else 'full-auto'
    
    # Determine stages to run
    stages = None
    if stage:
        stages = [stage]
    
    # Determine resume point
    resume_from = None
    if resume:
        state_file = project_dir / ".pipeline_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
                resume_from = state.get('current_stage')
    
    try:
        # Run pipeline
        results = pipeline.run(
            competition_name=config['project']['competition'],
            mode=mode,
            resume_from=resume_from,
            stages=stages
        )
        
        click.echo("\n✅ Pipeline completed successfully!")
        click.echo(f"📄 Check the final report at: {project_dir}/final_report.md")
        
    except Exception as e:
        click.echo(f"\n❌ Pipeline failed: {str(e)}")
        if mode == 'interactive':
            click.echo("💡 You can resume from the last successful stage using: kaggle-agent run --resume")
        sys.exit(1)


@cli.command()
def pause():
    """Pause the running pipeline."""
    click.echo("⏸️  Pausing pipeline...")
    
    # Create a pause signal file
    pause_file = Path(".pipeline_pause")
    pause_file.touch()
    
    click.echo("✅ Pause signal sent. The pipeline will pause at the next checkpoint.")


@cli.command()
@click.option('--inject', help='Path to custom code to inject')
def resume(inject):
    """Resume the paused pipeline."""
    click.echo("▶️  Resuming pipeline...")
    
    # Remove pause signal if exists
    pause_file = Path(".pipeline_pause")
    if pause_file.exists():
        pause_file.unlink()
    
    if inject:
        click.echo(f"💉 Injecting custom code from: {inject}")
        # TODO: Implement custom code injection
    
    # Run with resume flag
    ctx = click.get_current_context()
    ctx.invoke(run, resume=True)


@cli.command()
def status():
    """Show current pipeline status."""
    click.echo("📊 Pipeline Status")
    
    # Check if we're in a project directory
    if not Path("config.yaml").exists():
        click.echo("❌ Not in a Kaggle Agent project directory.")
        sys.exit(1)
    
    # Load pipeline state
    state_file = Path(".pipeline_state.json")
    if not state_file.exists():
        click.echo("📍 No pipeline has been run yet.")
        return
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    # Display status
    click.echo(f"\n📍 Current Stage: {state.get('current_stage', 'Unknown')}")
    click.echo(f"⏰ Last Updated: {state.get('last_updated', 'Unknown')}")
    
    completed_stages = state.get('completed_stages', [])
    if completed_stages:
        click.echo(f"\n✅ Completed Stages: {', '.join(completed_stages)}")
    
    # Check for pause signal
    if Path(".pipeline_pause").exists():
        click.echo("\n⏸️  Pipeline is paused")
    
    # Show available stages
    all_stages = ['download_data', 'eda', 'feature_engineering', 'modeling', 'submission']
    remaining_stages = [s for s in all_stages if s not in completed_stages]
    if remaining_stages:
        click.echo(f"\n📋 Remaining Stages: {', '.join(remaining_stages)}")


@cli.command()
def hooks():
    """List available hooks for intervention."""
    click.echo("🔗 Available Hooks\n")
    
    hooks = [
        ("after_download_data", "After data is downloaded and files are identified"),
        ("after_eda", "After exploratory data analysis is complete"),
        ("after_feature_engineering", "After features are engineered"),
        ("after_modeling", "After models are trained"),
        ("after_submission", "After submission file is generated")
    ]
    
    for hook_name, description in hooks:
        click.echo(f"  • {hook_name}: {description}")
    
    click.echo("\n💡 To use hooks, set them in config.yaml or use --interactive mode")


@cli.command()
def config():
    """Show current configuration."""
    click.echo("⚙️  Configuration\n")
    
    # Check if we're in a project directory
    config_path = Path("config.yaml")
    if not config_path.exists():
        click.echo("❌ Not in a Kaggle Agent project directory.")
        sys.exit(1)
    
    # Load and display configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    click.echo(f"📁 Competition: {config.get('competition', 'Unknown')}")
    click.echo(f"📂 Project Name: {config.get('project_name', 'Unknown')}")
    
    # Pipeline stages
    stages = config.get('pipeline', {}).get('stages', {})
    click.echo("\n🔧 Pipeline Stages:")
    for stage, settings in stages.items():
        enabled = settings.get('enabled', True)
        status = "✅" if enabled else "❌"
        click.echo(f"  {status} {stage}")
    
    # Hooks
    hooks_config = config.get('hooks', {})
    if hooks_config:
        click.echo("\n🔗 Configured Hooks:")
        for hook in hooks_config:
            click.echo(f"  • {hook}")


@cli.command()
@click.option('--type', 'code_type', type=click.Choice(['feature_engineering', 'model', 'preprocessing']), 
              required=True, help='Type of code to inject')
@click.option('--file', 'code_file', type=click.Path(exists=True), help='Path to code file')
@click.option('--name', default=None, help='Name for the custom module')
def inject(code_type, code_file, name):
    """Inject custom code into the pipeline."""
    click.echo(f"💉 Injecting custom {code_type} code...")
    
    # Check if we're in a project directory
    if not Path("config.yaml").exists():
        click.echo("❌ Not in a Kaggle Agent project directory.")
        sys.exit(1)
    
    # Read code from file
    if code_file:
        with open(code_file, 'r') as f:
            code = f.read()
    else:
        # Create template
        from .core.code_injector import CodeInjector
        injector = CodeInjector(Path.cwd())
        template = injector.create_custom_module_template(code_type)
        
        # Save template to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(template)
            temp_file = f.name
        
        click.echo(f"\n📝 Template created at: {temp_file}")
        click.echo("Edit the template and run the command again with --file option")
        return
    
    # Generate name if not provided
    if not name:
        name = f"custom_{code_type}_{Path(code_file).stem}"
    
    # Inject code
    from .core.code_injector import CodeInjector
    injector = CodeInjector(Path.cwd())
    
    if code_type == 'feature_engineering':
        result = injector.inject_feature_engineering(code, name)
    elif code_type == 'model':
        result = injector.inject_model(code, name)
    elif code_type == 'preprocessing':
        result = injector.inject_preprocessing(code, name)
    
    if result['success']:
        click.echo(f"\n✅ Successfully injected custom code: {name}")
        click.echo(f"📄 Module saved to: {result['path']}")
        if result.get('available_functions'):
            click.echo(f"\n🔧 Available functions: {', '.join(result['available_functions'])}")
    else:
        click.echo(f"\n❌ Failed to inject code: {result['error']}")
        if result.get('details'):
            click.echo(f"Details: {result['details']}")


@cli.command()
def list_modules():
    """List all custom modules."""
    click.echo("📁 Custom Modules\n")
    
    from .core.code_injector import CodeInjector
    injector = CodeInjector(Path.cwd())
    
    modules = injector.list_custom_modules()
    if modules:
        for module in modules:
            click.echo(f"• {module['name']}")
            if 'description' in module:
                click.echo(f"  {module['description']}")
            click.echo(f"  Size: {module['size']} bytes")
            click.echo(f"  Modified: {module['modified']}")
            click.echo()
    else:
        click.echo("No custom modules found.")


@cli.command()
@click.argument('checkpoint_id', required=False)
@click.option('--list', 'list_checkpoints', is_flag=True, help='List available checkpoints')
@click.option('--create', is_flag=True, help='Create a new checkpoint')
def checkpoint(checkpoint_id, list_checkpoints, create):
    """Manage pipeline checkpoints."""
    if not Path("config.yaml").exists():
        click.echo("❌ Not in a Kaggle Agent project directory.")
        sys.exit(1)
    
    from .pipeline import KagglePipeline
    
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    pipeline = KagglePipeline(Path.cwd(), config)
    
    if list_checkpoints:
        click.echo("📦 Available Checkpoints\n")
        checkpoints = pipeline.list_checkpoints()
        
        if checkpoints:
            for cp in checkpoints:
                click.echo(f"• {cp['id']}")
                click.echo(f"  Stage: {cp['stage']}")
                click.echo(f"  Created: {cp['created_at']}")
                click.echo(f"  Completed stages: {', '.join(cp['completed_stages'])}")
                click.echo()
        else:
            click.echo("No checkpoints found.")
    
    elif create:
        name = checkpoint_id or input("Checkpoint name (optional): ")
        cp_id = pipeline.create_checkpoint(name)
        click.echo(f"\n✅ Created checkpoint: {cp_id}")
    
    elif checkpoint_id:
        click.echo(f"\n🔄 Restoring checkpoint: {checkpoint_id}")
        try:
            pipeline.restore_checkpoint(checkpoint_id)
            click.echo("✅ Checkpoint restored successfully!")
        except Exception as e:
            click.echo(f"❌ Failed to restore checkpoint: {str(e)}")


@cli.command()
def experiments():
    """View experiment history and results."""
    click.echo("📊 Experiment History\n")
    
    if not Path("config.yaml").exists():
        click.echo("❌ Not in a Kaggle Agent project directory.")
        sys.exit(1)
    
    from .core.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker(Path.cwd())
    
    # Get recent experiments
    experiments = tracker.get_experiments(limit=10)
    
    if experiments:
        for exp in experiments:
            status_icon = "✅" if exp['status'] == 'completed' else "🔄" if exp['status'] == 'running' else "❌"
            click.echo(f"{status_icon} {exp['name']} (ID: {exp['id']})")
            click.echo(f"  Competition: {exp['competition']}")
            click.echo(f"  Created: {exp['created_at']}")
            
            if exp['metrics']:
                metrics = json.loads(exp['metrics'])
                if 'final_best_score' in metrics:
                    click.echo(f"  Best Score: {metrics['final_best_score']:.4f}")
            
            click.echo()
    else:
        click.echo("No experiments found.")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
"""Enhanced pipeline state management with checkpointing."""

import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


class StateManager:
    """Manages pipeline state with advanced checkpointing and recovery."""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.state_dir = project_dir / ".pipeline_state"
        self.state_dir.mkdir(exist_ok=True)
        
        self.state_file = self.state_dir / "state.json"
        self.checkpoint_dir = self.state_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.current_state = self._load_or_create_state()
    
    def _load_or_create_state(self) -> Dict[str, Any]:
        """Load existing state or create new one."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'current_stage': None,
                'completed_stages': [],
                'stage_results': {},
                'pipeline_config': {},
                'checkpoints': [],
                'status': 'initialized'
            }
    
    def update_state(self, updates: Dict[str, Any]):
        """Update current state."""
        self.current_state.update(updates)
        self.current_state['last_updated'] = datetime.now().isoformat()
        self.save_state()
    
    def save_state(self):
        """Save current state to disk."""
        with open(self.state_file, 'w') as f:
            json.dump(self.current_state, f, indent=2)
    
    def create_checkpoint(self, stage: str, include_data: bool = True) -> str:
        """Create a checkpoint for current state."""
        checkpoint_id = f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save state
        state_copy = self.current_state.copy()
        state_copy['checkpoint_id'] = checkpoint_id
        state_copy['checkpoint_stage'] = stage
        state_copy['checkpoint_created'] = datetime.now().isoformat()
        
        with open(checkpoint_path / "state.json", 'w') as f:
            json.dump(state_copy, f, indent=2)
        
        # Save pipeline objects
        objects_to_save = {
            'results': getattr(self, '_pipeline_results', {}),
            'models': {},
            'data_info': {}
        }
        
        # Save models if available
        model_dir = self.project_dir / "model_output"
        if model_dir.exists():
            for model_file in model_dir.glob("*.pkl"):
                objects_to_save['models'][model_file.stem] = str(model_file)
        
        with open(checkpoint_path / "objects.pkl", 'wb') as f:
            pickle.dump(objects_to_save, f)
        
        # Optionally save data snapshots
        if include_data:
            data_snapshot_dir = checkpoint_path / "data_snapshot"
            data_snapshot_dir.mkdir(exist_ok=True)
            
            # Save processed data info
            processed_dir = self.project_dir / "data" / "processed"
            if processed_dir.exists():
                data_info = {}
                for data_file in processed_dir.glob("*.csv"):
                    df_info = {
                        'shape': pd.read_csv(data_file, nrows=0).shape,
                        'columns': list(pd.read_csv(data_file, nrows=0).columns),
                        'size_mb': data_file.stat().st_size / 1024 / 1024
                    }
                    data_info[data_file.stem] = df_info
                
                with open(data_snapshot_dir / "data_info.json", 'w') as f:
                    json.dump(data_info, f, indent=2)
        
        # Update checkpoint list
        self.current_state['checkpoints'].append({
            'id': checkpoint_id,
            'stage': stage,
            'created_at': datetime.now().isoformat(),
            'include_data': include_data
        })
        self.save_state()
        
        print(f"✓ Created checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore from a checkpoint."""
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        # Load checkpoint state
        with open(checkpoint_path / "state.json", 'r') as f:
            checkpoint_state = json.load(f)
        
        # Load objects
        with open(checkpoint_path / "objects.pkl", 'rb') as f:
            objects = pickle.load(f)
        
        # Restore state
        self.current_state = checkpoint_state
        self.current_state['restored_from'] = checkpoint_id
        self.current_state['restored_at'] = datetime.now().isoformat()
        self.save_state()
        
        print(f"✓ Restored from checkpoint: {checkpoint_id}")
        return objects
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        
        for checkpoint_dir in self.checkpoint_dir.iterdir():
            if checkpoint_dir.is_dir():
                state_file = checkpoint_dir / "state.json"
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    
                    checkpoints.append({
                        'id': checkpoint_dir.name,
                        'stage': state.get('checkpoint_stage'),
                        'created_at': state.get('checkpoint_created'),
                        'completed_stages': state.get('completed_stages', [])
                    })
        
        # Sort by creation time
        checkpoints.sort(key=lambda x: x['created_at'], reverse=True)
        return checkpoints
    
    def get_stage_state(self, stage: str) -> Optional[Dict[str, Any]]:
        """Get state for a specific stage."""
        return self.current_state.get('stage_results', {}).get(stage)
    
    def save_stage_result(self, stage: str, result: Dict[str, Any]):
        """Save result for a specific stage."""
        if 'stage_results' not in self.current_state:
            self.current_state['stage_results'] = {}
        
        self.current_state['stage_results'][stage] = {
            'result': result,
            'completed_at': datetime.now().isoformat()
        }
        self.save_state()
    
    def export_state_summary(self) -> str:
        """Export a human-readable state summary."""
        summary = []
        summary.append("# Pipeline State Summary")
        summary.append(f"\nCreated: {self.current_state.get('created_at', 'Unknown')}")
        summary.append(f"Last Updated: {self.current_state.get('last_updated', 'Unknown')}")
        summary.append(f"Status: {self.current_state.get('status', 'Unknown')}")
        summary.append(f"Current Stage: {self.current_state.get('current_stage', 'None')}")
        
        # Completed stages
        completed = self.current_state.get('completed_stages', [])
        if completed:
            summary.append(f"\n## Completed Stages ({len(completed)})")
            for stage in completed:
                summary.append(f"- ✓ {stage}")
        
        # Stage results summary
        stage_results = self.current_state.get('stage_results', {})
        if stage_results:
            summary.append("\n## Stage Results")
            for stage, info in stage_results.items():
                summary.append(f"\n### {stage}")
                if isinstance(info, dict) and 'completed_at' in info:
                    summary.append(f"Completed: {info['completed_at']}")
                if isinstance(info, dict) and 'result' in info:
                    result = info['result']
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if not isinstance(value, (dict, list)):
                                summary.append(f"- {key}: {value}")
        
        # Checkpoints
        checkpoints = self.current_state.get('checkpoints', [])
        if checkpoints:
            summary.append(f"\n## Checkpoints ({len(checkpoints)})")
            for cp in checkpoints[-5:]:  # Show last 5
                summary.append(f"- {cp['id']} (Stage: {cp['stage']})")
        
        return "\n".join(summary)
    
    def clean_old_checkpoints(self, keep_last: int = 5):
        """Clean old checkpoints, keeping only the most recent ones."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > keep_last:
            # Remove old checkpoints
            for cp in checkpoints[keep_last:]:
                checkpoint_path = self.checkpoint_dir / cp['id']
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                    print(f"  Removed old checkpoint: {cp['id']}")
            
            # Update state
            self.current_state['checkpoints'] = self.current_state['checkpoints'][-keep_last:]
            self.save_state()
    
    def enable_auto_checkpoint(self, stages: List[str]):
        """Enable automatic checkpointing for specific stages."""
        self.current_state['auto_checkpoint_stages'] = stages
        self.save_state()
    
    def should_auto_checkpoint(self, stage: str) -> bool:
        """Check if auto-checkpoint is enabled for a stage."""
        auto_stages = self.current_state.get('auto_checkpoint_stages', [])
        return stage in auto_stages
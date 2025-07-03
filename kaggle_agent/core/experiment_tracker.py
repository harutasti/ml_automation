"""Experiment tracking and management system."""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import hashlib
import pandas as pd
import numpy as np
from contextlib import contextmanager


class ExperimentTracker:
    """Track experiments, results, and provide analytics."""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.db_path = project_dir / "experiments.db"
        self.experiments_dir = project_dir / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for experiment tracking."""
        with self._get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    competition TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'running',
                    config TEXT NOT NULL,
                    results TEXT,
                    metrics TEXT,
                    artifacts_path TEXT,
                    duration_seconds REAL,
                    notes TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT DEFAULT 'running',
                    metrics TEXT,
                    logs TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    run_id INTEGER,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_competition 
                ON experiments(competition)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_created 
                ON experiments(created_at)
            """)
    
    @contextmanager
    def _get_db(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def create_experiment(self, name: str, competition: str, config: Dict[str, Any]) -> str:
        """Create a new experiment."""
        # Generate unique ID
        timestamp = datetime.now().isoformat()
        exp_id = hashlib.md5(f"{name}_{competition}_{timestamp}".encode()).hexdigest()[:12]
        
        # Create experiment directory
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save to database
        with self._get_db() as conn:
            conn.execute("""
                INSERT INTO experiments (id, name, competition, config, artifacts_path)
                VALUES (?, ?, ?, ?, ?)
            """, (exp_id, name, competition, json.dumps(config), str(exp_dir)))
        
        print(f"📊 Created experiment: {name} (ID: {exp_id})")
        return exp_id
    
    def start_run(self, experiment_id: str, stage: str) -> int:
        """Start a new run within an experiment."""
        with self._get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO runs (experiment_id, stage)
                VALUES (?, ?)
            """, (experiment_id, stage))
            return cursor.lastrowid
    
    def complete_run(self, run_id: int, metrics: Optional[Dict[str, Any]] = None, 
                    logs: Optional[str] = None, status: str = 'completed'):
        """Complete a run and record results."""
        with self._get_db() as conn:
            conn.execute("""
                UPDATE runs 
                SET completed_at = CURRENT_TIMESTAMP,
                    status = ?,
                    metrics = ?,
                    logs = ?
                WHERE id = ?
            """, (status, json.dumps(metrics) if metrics else None, logs, run_id))
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, Any]):
        """Log metrics for an experiment."""
        with self._get_db() as conn:
            # Get existing metrics
            result = conn.execute(
                "SELECT metrics FROM experiments WHERE id = ?", 
                (experiment_id,)
            ).fetchone()
            
            existing_metrics = json.loads(result['metrics']) if result and result['metrics'] else {}
            existing_metrics.update(metrics)
            
            conn.execute("""
                UPDATE experiments
                SET metrics = ?
                WHERE id = ?
            """, (json.dumps(existing_metrics), experiment_id))
    
    def log_artifact(self, experiment_id: str, name: str, artifact_type: str, 
                    path: Path, run_id: Optional[int] = None, 
                    metadata: Optional[Dict[str, Any]] = None):
        """Log an artifact (file) for an experiment."""
        size = path.stat().st_size if path.exists() else 0
        
        with self._get_db() as conn:
            conn.execute("""
                INSERT INTO artifacts (experiment_id, run_id, name, type, path, 
                                     size_bytes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, run_id, name, artifact_type, str(path), 
                  size, json.dumps(metadata) if metadata else None))
    
    def complete_experiment(self, experiment_id: str, results: Dict[str, Any], 
                          status: str = 'completed'):
        """Complete an experiment and record final results."""
        with self._get_db() as conn:
            # Calculate duration
            exp = conn.execute(
                "SELECT created_at FROM experiments WHERE id = ?", 
                (experiment_id,)
            ).fetchone()
            
            if exp:
                created = datetime.fromisoformat(exp['created_at'])
                duration = (datetime.now() - created).total_seconds()
            else:
                duration = 0
            
            conn.execute("""
                UPDATE experiments
                SET status = ?,
                    results = ?,
                    duration_seconds = ?
                WHERE id = ?
            """, (status, json.dumps(results), duration, experiment_id))
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details."""
        with self._get_db() as conn:
            exp = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", 
                (experiment_id,)
            ).fetchone()
            
            if exp:
                return dict(exp)
        return None
    
    def get_experiments(self, competition: Optional[str] = None, 
                       status: Optional[str] = None,
                       limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of experiments."""
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []
        
        if competition:
            query += " AND competition = ?"
            params.append(competition)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with self._get_db() as conn:
            results = conn.execute(query, params).fetchall()
            return [dict(row) for row in results]
    
    def get_best_experiment(self, competition: str, metric: str, 
                           minimize: bool = False) -> Optional[Dict[str, Any]]:
        """Get best experiment for a competition based on a metric."""
        experiments = self.get_experiments(competition=competition, status='completed')
        
        best_exp = None
        best_value = float('inf') if minimize else float('-inf')
        
        for exp in experiments:
            if exp['metrics']:
                metrics = json.loads(exp['metrics'])
                if metric in metrics:
                    value = metrics[metric]
                    if (minimize and value < best_value) or (not minimize and value > best_value):
                        best_value = value
                        best_exp = exp
        
        return best_exp
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments."""
        data = []
        
        with self._get_db() as conn:
            for exp_id in experiment_ids:
                exp = conn.execute(
                    "SELECT * FROM experiments WHERE id = ?", 
                    (exp_id,)
                ).fetchone()
                
                if exp:
                    row = {
                        'id': exp['id'],
                        'name': exp['name'],
                        'competition': exp['competition'],
                        'status': exp['status'],
                        'created_at': exp['created_at'],
                        'duration_seconds': exp['duration_seconds']
                    }
                    
                    # Add metrics
                    if exp['metrics']:
                        metrics = json.loads(exp['metrics'])
                        for key, value in metrics.items():
                            row[f'metric_{key}'] = value
                    
                    # Add results
                    if exp['results']:
                        results = json.loads(exp['results'])
                        if 'best_model' in results:
                            row['best_model'] = results['best_model']
                        if 'best_score' in results:
                            row['best_score'] = results['best_score']
                    
                    data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_experiment_report(self, experiment_id: str) -> str:
        """Generate a comprehensive report for an experiment."""
        exp = self.get_experiment(experiment_id)
        if not exp:
            return "Experiment not found"
        
        report = []
        report.append(f"# Experiment Report: {exp['name']}")
        report.append(f"\n**ID**: {exp['id']}")
        report.append(f"**Competition**: {exp['competition']}")
        report.append(f"**Status**: {exp['status']}")
        report.append(f"**Created**: {exp['created_at']}")
        
        if exp['duration_seconds']:
            duration_min = exp['duration_seconds'] / 60
            report.append(f"**Duration**: {duration_min:.1f} minutes")
        
        # Configuration
        if exp['config']:
            config = json.loads(exp['config'])
            report.append("\n## Configuration")
            for key, value in config.items():
                report.append(f"- **{key}**: {value}")
        
        # Metrics
        if exp['metrics']:
            metrics = json.loads(exp['metrics'])
            report.append("\n## Metrics")
            for key, value in metrics.items():
                if isinstance(value, float):
                    report.append(f"- **{key}**: {value:.4f}")
                else:
                    report.append(f"- **{key}**: {value}")
        
        # Results
        if exp['results']:
            results = json.loads(exp['results'])
            report.append("\n## Results")
            for key, value in results.items():
                if isinstance(value, dict):
                    report.append(f"\n### {key}")
                    for k, v in value.items():
                        report.append(f"- **{k}**: {v}")
                else:
                    report.append(f"- **{key}**: {value}")
        
        # Runs
        with self._get_db() as conn:
            runs = conn.execute("""
                SELECT * FROM runs 
                WHERE experiment_id = ? 
                ORDER BY started_at
            """, (experiment_id,)).fetchall()
            
            if runs:
                report.append("\n## Pipeline Runs")
                for run in runs:
                    report.append(f"\n### {run['stage']}")
                    report.append(f"- Status: {run['status']}")
                    if run['metrics']:
                        run_metrics = json.loads(run['metrics'])
                        for key, value in run_metrics.items():
                            report.append(f"- {key}: {value}")
        
        # Artifacts
        with self._get_db() as conn:
            artifacts = conn.execute("""
                SELECT * FROM artifacts 
                WHERE experiment_id = ? 
                ORDER BY created_at
            """, (experiment_id,)).fetchall()
            
            if artifacts:
                report.append("\n## Artifacts")
                for artifact in artifacts:
                    size_mb = artifact['size_bytes'] / 1024 / 1024
                    report.append(f"- **{artifact['name']}** ({artifact['type']}, {size_mb:.2f} MB)")
        
        return "\n".join(report)
    
    def clean_old_experiments(self, days: int = 30):
        """Clean up old experiments and their artifacts."""
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        
        with self._get_db() as conn:
            # Get old experiments
            old_exps = conn.execute("""
                SELECT id, artifacts_path FROM experiments 
                WHERE created_at < ? AND status != 'running'
            """, (cutoff_date.isoformat(),)).fetchall()
            
            for exp in old_exps:
                # Remove experiment directory
                exp_dir = Path(exp['artifacts_path'])
                if exp_dir.exists():
                    import shutil
                    shutil.rmtree(exp_dir)
                
                # Remove from database
                conn.execute("DELETE FROM artifacts WHERE experiment_id = ?", (exp['id'],))
                conn.execute("DELETE FROM runs WHERE experiment_id = ?", (exp['id'],))
                conn.execute("DELETE FROM experiments WHERE id = ?", (exp['id'],))
            
            print(f"🧹 Cleaned up {len(old_exps)} old experiments")
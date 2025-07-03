"""Competition data management module."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime


class CompetitionManager:
    """Manages Kaggle competition data and metadata."""
    
    def __init__(self, competition_name: str, data_dir: Path):
        self.competition_name = competition_name
        self.data_dir = data_dir
        self.raw_data_dir = data_dir / "raw"
        self.competition_info = None
        
    def get_competition_info(self) -> Dict[str, Any]:
        """Get competition metadata."""
        try:
            import kaggle
            competitions = kaggle.api.competitions_list(search=self.competition_name)
            
            for comp in competitions:
                # Check both ref and the URL format
                comp_ref = comp.ref
                if comp_ref.startswith("https://www.kaggle.com/competitions/"):
                    comp_ref = comp_ref.split("/")[-1]
                
                if comp_ref == self.competition_name:
                    # Extract tags using _tags attribute for better results
                    tags = []
                    if hasattr(comp, '_tags') and comp._tags:
                        for tag in comp._tags:
                            if hasattr(tag, 'name'):
                                tags.append(tag.name)
                            elif isinstance(tag, dict):
                                tags.append(tag.get('name', ''))
                            else:
                                tags.append(str(tag))
                    elif comp.tags:
                        # Fallback to regular tags if _tags not available
                        tags = [tag.name if hasattr(tag, 'name') else str(tag) for tag in comp.tags]
                    
                    self.competition_info = {
                        "name": comp_ref,  # Use the extracted name, not the full URL
                        "title": comp.title,
                        "description": comp.description,
                        "evaluation_metric": comp.evaluation_metric,
                        "is_kernels_submissions_only": comp.is_kernels_submissions_only,
                        "submission_deadline": str(comp.deadline),
                        "team_count": comp.team_count,
                        "reward": comp.reward,
                        "tags": tags
                    }
                    
                    # Detect task type from tags and evaluation metric
                    self._detect_task_type_from_competition_info()
                    
                    print(f"✓ Competition found: {comp.title}")
                    print(f"  Evaluation metric: {comp.evaluation_metric}")
                    print(f"  Teams: {comp.team_count}")
                    
                    # Print task type if detected
                    if self.competition_info.get('task_type'):
                        task_info = f"  Task type: {self.competition_info['task_type']}"
                        if self.competition_info.get('task_subtype'):
                            task_info += f" ({self.competition_info['task_subtype']})"
                        print(task_info)
                    
                    return self.competition_info
            
            raise ValueError(f"Competition '{self.competition_name}' not found")
            
        except Exception as e:
            print(f"✗ Failed to get competition info: {e}")
            raise
    
    def _detect_task_type_from_competition_info(self):
        """Detect task type from competition metadata."""
        # Check tags first
        if self.competition_info.get('tags'):
            tag_names = [tag.lower() for tag in self.competition_info['tags']]
            
            # Binary classification
            if 'binary classification' in tag_names:
                self.competition_info['task_type'] = 'classification'
                self.competition_info['task_subtype'] = 'binary'
                return
            
            # Multiclass classification
            elif any(tag in tag_names for tag in ['multiclass classification', 'multiclass', 'multi-class']):
                self.competition_info['task_type'] = 'classification'
                self.competition_info['task_subtype'] = 'multiclass'
                return
                
            # Regression
            elif 'regression' in tag_names:
                self.competition_info['task_type'] = 'regression'
                return
        
        # Check evaluation metric
        if self.competition_info.get('evaluation_metric'):
            metric = self.competition_info['evaluation_metric'].lower()
            
            # Classification metrics
            classification_keywords = [
                'accuracy', 'auc', 'log loss', 'logloss', 'f1', 'f-score',
                'precision', 'recall', 'classification', 'categorization'
            ]
            if any(keyword in metric for keyword in classification_keywords):
                self.competition_info['task_type'] = 'classification'
                # Try to detect if binary
                if 'binary' in metric or 'auc' in metric:
                    self.competition_info['task_subtype'] = 'binary'
                return
            
            # Regression metrics
            regression_keywords = [
                'rmse', 'mse', 'mae', 'rmsle', 'r2', 'squared error',
                'absolute error', 'mean error', 'regression'
            ]
            if any(keyword in metric for keyword in regression_keywords):
                self.competition_info['task_type'] = 'regression'
                return
    
    def download_data(self, handle_auth: bool = True) -> List[str]:
        """Download competition data files."""
        try:
            import kaggle
            # Create raw data directory
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📥 Downloading data for competition: {self.competition_name}")
            
            # Handle authentication if needed
            if handle_auth:
                from .competition_auth import CompetitionAuth
                auth_handler = CompetitionAuth()
                if not auth_handler.ensure_competition_access(self.competition_name):
                    raise Exception("Failed to authenticate for competition")
            
            # Get list of files
            files_response = kaggle.api.competition_list_files(self.competition_name)
            
            # Handle the API response correctly
            if hasattr(files_response, 'files'):
                files = files_response.files
            else:
                files = files_response
            
            file_names = [f.name if hasattr(f, 'name') else str(f) for f in files]
            
            print(f"  Found {len(file_names)} files to download")
            
            # Download all files
            kaggle.api.competition_download_files(
                self.competition_name,
                path=str(self.raw_data_dir),
                quiet=False
            )
            
            # Extract if zip file
            zip_path = self.raw_data_dir / f"{self.competition_name}.zip"
            if zip_path.exists():
                import zipfile
                print("  Extracting zip file...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.raw_data_dir)
                zip_path.unlink()  # Remove zip file
            
            # List downloaded files
            downloaded_files = list(self.raw_data_dir.glob("*"))
            print(f"✓ Downloaded {len(downloaded_files)} files:")
            for file in downloaded_files:
                print(f"  - {file.name} ({file.stat().st_size / 1024 / 1024:.2f} MB)")
            
            return [str(f) for f in downloaded_files]
            
        except Exception as e:
            print(f"✗ Failed to download data: {e}")
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """Analyze downloaded data files."""
        data_info = {
            "files": [],
            "total_size_mb": 0,
            "file_types": {},
            "csv_files": []
        }
        
        for file_path in self.raw_data_dir.glob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / 1024 / 1024
                file_type = file_path.suffix.lower()
                
                file_info = {
                    "name": file_path.name,
                    "path": str(file_path),
                    "size_mb": size_mb,
                    "type": file_type
                }
                
                # For CSV files, get basic info
                if file_type == ".csv":
                    try:
                        df = pd.read_csv(file_path, nrows=5)
                        file_info["columns"] = list(df.columns)
                        file_info["shape"] = (sum(1 for _ in open(file_path)) - 1, len(df.columns))
                        data_info["csv_files"].append(file_path.name)
                    except:
                        pass
                
                data_info["files"].append(file_info)
                data_info["total_size_mb"] += size_mb
                
                # Count file types
                if file_type not in data_info["file_types"]:
                    data_info["file_types"][file_type] = 0
                data_info["file_types"][file_type] += 1
        
        return data_info
    
    def identify_data_files(self) -> Dict[str, str]:
        """Identify train, test, and submission files."""
        files = {}
        csv_files = list(self.raw_data_dir.glob("*.csv"))
        
        for file in csv_files:
            filename_lower = file.name.lower()
            
            # Common patterns for data files
            if "train" in filename_lower:
                files["train"] = str(file)
            elif "test" in filename_lower and "sample" not in filename_lower:
                files["test"] = str(file)
            elif "sample" in filename_lower and "submission" in filename_lower:
                files["sample_submission"] = str(file)
            elif "submission" in filename_lower or filename_lower == "gender_submission.csv":
                files["sample_submission"] = str(file)
        
        # If not found by name, try to identify by structure
        if not files:
            print("⚠️  Could not identify files by name, analyzing structure...")
            
            for file in csv_files:
                try:
                    df = pd.read_csv(file, nrows=10)
                    
                    # Check if it looks like a submission file
                    if len(df.columns) == 2 and "id" in [col.lower() for col in df.columns]:
                        files["sample_submission"] = str(file)
                    # Check if it has many features (likely train/test)
                    elif len(df.columns) > 5:
                        if "train" not in files:
                            files["train"] = str(file)
                        else:
                            files["test"] = str(file)
                except:
                    pass
        
        print("📁 Identified data files:")
        for key, path in files.items():
            print(f"  {key}: {Path(path).name}")
        
        return files
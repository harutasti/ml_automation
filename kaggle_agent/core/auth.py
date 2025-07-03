"""Kaggle API authentication and initialization module."""

import os
import json
from pathlib import Path
from typing import Optional, Dict


class KaggleAuth:
    """Handles Kaggle API authentication and configuration."""
    
    def __init__(self, kaggle_json_path: Optional[str] = None):
        self.kaggle_json_path = kaggle_json_path or os.path.expanduser("~/.config/kaggle/kaggle.json")
        self.credentials: Optional[Dict[str, str]] = None
        
    def setup(self) -> bool:
        """Set up Kaggle API authentication."""
        if self._check_existing_auth():
            return True
            
        return self._create_auth_from_env()
    
    def _check_existing_auth(self) -> bool:
        """Check if Kaggle credentials already exist."""
        kaggle_path = Path(self.kaggle_json_path)
        
        if kaggle_path.exists():
            try:
                with open(kaggle_path, 'r') as f:
                    self.credentials = json.load(f)
                    
                # Set environment variables
                os.environ['KAGGLE_USERNAME'] = self.credentials['username']
                os.environ['KAGGLE_KEY'] = self.credentials['key']
                
                print(f"✓ Kaggle credentials found at {kaggle_path}")
                return True
            except Exception as e:
                print(f"✗ Error reading Kaggle credentials: {e}")
                return False
        
        return False
    
    def _create_auth_from_env(self) -> bool:
        """Create Kaggle authentication from environment variables."""
        username = os.environ.get('KAGGLE_USERNAME')
        key = os.environ.get('KAGGLE_KEY')
        
        if not username or not key:
            print("✗ KAGGLE_USERNAME and KAGGLE_KEY environment variables not found")
            return False
        
        # Create directory if it doesn't exist
        kaggle_dir = Path(self.kaggle_json_path).parent
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        
        # Save credentials
        self.credentials = {"username": username, "key": key}
        
        with open(self.kaggle_json_path, 'w') as f:
            json.dump(self.credentials, f)
        
        # Set permissions (Unix-like systems only)
        try:
            os.chmod(self.kaggle_json_path, 0o600)
        except:
            pass
        
        print(f"✓ Kaggle credentials saved to {self.kaggle_json_path}")
        return True
    
    def validate(self) -> bool:
        """Validate Kaggle API credentials by making a test API call."""
        try:
            import kaggle
            # Test API call
            kaggle.api.competitions_list(page=1, search="")
            print("✓ Kaggle API authentication validated")
            return True
        except Exception as e:
            print(f"✗ Kaggle API validation failed: {e}")
            return False
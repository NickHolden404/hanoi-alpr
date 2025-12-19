"""
Configuration loader for Hanoi ALPR system.
Loads and validates configuration from YAML file.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for ALPR system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {self.config_path}")
        return config
    
    def _validate_config(self):
        """Validate required configuration sections exist."""
        required_sections = ['processing', 'ocr', 'plate_validation', 'output']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate ranges
        proc = self.config['processing']
        if not 0 <= proc['confidence_threshold'] <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if proc['frame_skip'] < 1:
            raise ValueError("frame_skip must be >= 1")
        
        logger.info("Configuration validation passed")
    
    def _create_directories(self):
        """Create output directories if they don't exist."""
        dirs = [
            self.config['output']['output_dir'],
            self.config['output']['image_dir'],
            self.config['output']['log_dir'],
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'processing.frame_skip')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get('processing.frame_skip')
            5
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.config[key]
    
    def __repr__(self) -> str:
        return f"Config(path={self.config_path}, sections={list(self.config.keys())})"


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)


# Example usage
if __name__ == "__main__":
    # Test the config loader
    config = load_config()
    
    print("Configuration loaded successfully!")
    print(f"Frame skip: {config.get('processing.frame_skip')}")
    print(f"Use GPU: {config.get('ocr.use_gpu')}")
    print(f"Output dir: {config.get('output.output_dir')}")
    
    # Test validation
    print(f"\nAll sections: {list(config.config.keys())}")

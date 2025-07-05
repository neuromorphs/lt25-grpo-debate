import os
import yaml
from typing import Dict, Any


class Config:
    """Configuration class that allows dot notation access to nested dictionaries."""
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_config(config_path: str = "configs/default.yaml") -> Config:
    """Load configuration from YAML file, merging with base config."""
    # Load base config
    with open("configs/base.yaml", 'r') as file:
        base_config = yaml.safe_load(file)
    
    # Load override config
    with open(config_path, 'r') as file:
        override_config = yaml.safe_load(file)
    
    # Merge configs (override takes precedence)
    def merge_dicts(base, override):
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    config_dict = merge_dicts(base_config, override_config)
    config = Config(config_dict)
    
    # Set environment variables for wandb
    if os.environ.get("WANDB_ENTITY") is None and config.logging.wandb_entity is not None:
        print(f"Setting WANDB_ENTITY to {config.logging.wandb_entity}")
        os.environ["WANDB_ENTITY"] = config.logging.wandb_entity
    if os.environ.get("WANDB_PROJECT") is None and config.logging.wandb_project is not None:
        print(f"Setting WANDB_PROJECT to {config.logging.wandb_project}")
        os.environ["WANDB_PROJECT"] = config.logging.wandb_project
    
    # print config in yaml format
    print(yaml.dump(config_dict, indent=2))
    
    return config
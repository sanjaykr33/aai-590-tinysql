import os
from dotenv import load_dotenv
import yaml
from functools import reduce
import operator

class ConfigLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Load .env file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        load_dotenv(dotenv_path=f"{project_root}/.env")
        self.env_vars = os.environ

        # Load config.yaml
        config_path = os.path.join(project_root, 'config', 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.yaml_config = yaml.safe_load(f)
        else:
            self.yaml_config = {}

    def get(self, key: str, default=None):
        """
        Retrieves a value from the loaded configuration using a dot-separated key.
        Checks environment variables first (converts 'a.b' to 'A_B').

        Args:
            key (str): The dot-separated key (e.g., "project.id").
            default: The value to return if the key is not found. Defaults to None.

        Returns:
            The configuration value or the default.
        """
        # 1. Environment Variable Precedence (Safety First)
        env_map = {
            "huggingface.token": "HF_TOKEN",
            "HF_TOKEN": "HF_TOKEN",
            "project.id": "GOOGLE_CLOUD_PROJECT",
            "PROJECT_ID": "GOOGLE_CLOUD_PROJECT",
            "gcp.service_account": "GCP_SERVICE_ACCOUNT",
            "GCP_SERVICE_ACCOUNT": "GCP_SERVICE_ACCOUNT"
        }

        env_key_lookup = env_map.get(key)
        if env_key_lookup:
            env_value = os.getenv(env_key_lookup)
            if env_value is not None:
                return env_value.strip('"').strip("'")

        # Also check for direct key if not found in env_map
        direct_env_key = key.replace('.', '_').upper()
        env_value = os.getenv(direct_env_key)
        if env_value is not None:
            return env_value.strip('"').strip('"')

        # 2. YAML Configuration Lookup
        parts = key.split('.')
        current = self.yaml_config
        try:
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default # Key not found in YAML config path
            return current
        except (KeyError, TypeError): # Catch errors if any part of the path is not a dict or key is missing
            return default

        # Fallback to default if not found anywhere
        return default


# Instantiate the config_loader to make it a singleton and load configs on import
config_loader = ConfigLoader()

if __name__ == "__main__":
    # Get a nested property
    lora_rank = config_loader.get("training.lora.r")
    print(f"LoRA Rank: {lora_rank}")

    # Get a top-level property
    project_id = config_loader.get("project.id")
    print(f"Project ID: {project_id}")
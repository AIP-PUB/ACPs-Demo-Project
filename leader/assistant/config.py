import json
import logging
import sys
import copy
from pathlib import Path
from typing import Dict, Any
import tomllib as toml

logger = logging.getLogger(__name__)

# Default Configuration
DEFAULT_CONFIG = {
    "app": {
        "acs_json": "atr/acs.json",
    },
    "uvicorn": {
        "host": "0.0.0.0",
        "port": 59210,
        "reload": False,
    },
    "rabbitmq": {
        "host": "localhost",
        "port": 5672,
        "user": "guest",
        "password": "guest",
        "vhost": "/",
    },
    "llm": {
        "default": {
            "api_type": "openai",
            "model": "gpt-4",
            # api_key and base_url are required in config.toml
        }
    },
    "discovery": {
        "server_base_url": "",
        "timeout": 30,
        "limit": 5,
    },
}


class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load()
        return cls._instance

    def _deep_update(self, target: Dict, source: Dict):
        for key, value in source.items():
            if (
                isinstance(value, dict)
                and key in target
                and isinstance(target[key], dict)
            ):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def _load(self):
        # Start with defaults
        self._config = copy.deepcopy(DEFAULT_CONFIG)

        # Load TOML
        # Assuming this file is in leader/assistant/config.py
        # We want project root to be leader/
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        config_path = project_root / "config.toml"

        if not config_path.exists():
            logger.error(f"Config file not found at {config_path.absolute()}. Exiting.")
            sys.exit(1)

        try:
            with open(config_path, "rb") as f:
                toml_config = toml.load(f)
                self._deep_update(self._config, toml_config)
        except Exception as e:
            logger.error(f"Failed to load config.toml: {e}")
            sys.exit(1)

        # Load leader AIC from acs.json
        self._load_leader_aic(project_root)

        self._validate()

    def _load_leader_aic(self, project_root: Path):
        """从 acs.json 文件中解析 leader_aic"""
        acs_json_rel = self._config.get("app", {}).get("acs_json", "atr/acs.json")
        acs_json_path = project_root / acs_json_rel

        if not acs_json_path.exists():
            logger.error(f"ACS file not found: {acs_json_path.absolute()}. Exiting.")
            sys.exit(1)

        try:
            with open(acs_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse {acs_json_path}: {e}")
            sys.exit(1)

        aic = data.get("aic")
        if not aic or not isinstance(aic, str) or not aic.strip():
            logger.error(f"Missing or empty 'aic' field in {acs_json_path}. Exiting.")
            sys.exit(1)

        self._config["app"]["leader_aic"] = aic
        logger.info(f"Loaded leader AIC from {acs_json_rel}: {aic}")

    def _validate(self):
        """Validate configuration values"""
        # 1. Check for None or empty values in critical sections
        for section in ["app", "uvicorn", "rabbitmq"]:
            if section not in self._config:
                logger.error(f"Missing configuration section: [{section}]")
                sys.exit(1)
            for key, value in self._config[section].items():
                if value is None or (isinstance(value, str) and not value.strip()):
                    logger.error(f"Missing value for config: [{section}].{key}")
                    sys.exit(1)

        # 2. Validate LLM configuration
        llm_config = self._config.get("llm", {})
        if not llm_config:
            logger.error("Missing [llm] configuration section")
            sys.exit(1)

        required_llm_keys = ["api_type", "model", "api_key", "base_url"]

        # Iterate over all profiles (e.g., default, fast, pro)
        for profile_name, profile_data in llm_config.items():
            if not isinstance(profile_data, dict):
                continue  # Skip non-dict entries if any

            for key in required_llm_keys:
                value = profile_data.get(key)
                if value is None or (isinstance(value, str) and not value.strip()):
                    logger.error(
                        f"Missing required LLM config in profile [{profile_name}]: {key}"
                    )
                    sys.exit(1)

    @property
    def config(self) -> Dict[str, Any]:
        return self._config


# Singleton instance
settings = ConfigManager().config

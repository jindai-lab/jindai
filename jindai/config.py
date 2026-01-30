"""Configuration management for Jindai application"""

import os
import sys
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigObject(BaseModel):
    """Configuration object for Jindai application settings.

    Supports additional fields beyond the defined ones for extensibility.
    Extra fields are accessible via the model_extra attribute.
    """
    model_config = ConfigDict(extra="allow")

    # defined fields
    secret_key: str = Field(description="Secret key for Flask application")
    concurrent: int = Field(default=3, description="Default concurrency level")
    storage: str = Field(description="Storage root directory path")
    database: str = Field(description="Database connection string")
    redis: str = Field(description="Redis connection string")
    plugins: List[str] = Field(
        default_factory=lambda: ["*"], description="List of plugins to load"
    )
    oidc_secrets: str = Field(description="Path to OIDC client secrets file")
    port: int = Field(default=8370, description="Default port for web server")
    embedding_model: str = Field(
        description="Sentence transformer model name for embeddings"
    )
    embedding_dims: int = Field(description="Embedding vector dimensions")

    ui_dist: str = Field(description="Path to UI distribution files")
    paddle_remote: str = Field(description="PaddleOCR remote service URL")

    constants: dict = Field(default_factory=dict, description="Application constants")

    @field_validator("redis", mode="after")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        """Remove trailing slash from Redis URL"""
        return v.rstrip("/")

    @classmethod
    def load_from_yaml(cls, file_path: str) -> "ConfigObject":
        """Load and parse configuration from YAML file

        :param file_path: Path to YAML configuration file
        :type file_path: str
        :return: Validated configuration object
        :rtype: ConfigObject
        :raises: ValidationError if YAML contains invalid configuration
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        print("* loaded config from", file_path)
        return cls.model_validate(data)

    def get_extra(self) -> Dict[str, Any]:
        """Get all extra configuration fields not defined in the class

        :return: Dictionary of extra configuration fields
        :rtype: Dict[str, Any]
        """
        return self.model_extra or {}


def load_config_from_args() -> ConfigObject:
    """Load configuration from command line arguments

    :return: Loaded configuration instance
    :rtype: ConfigObject
    """
    if "-c" in sys.argv:
        config_arg = sys.argv.index("-c")
        config_file = sys.argv[config_arg + 1]
        sys.argv.pop(config_arg + 1)
        sys.argv.pop(config_arg)
        os.environ["CONFIG_FILE"] = config_file

    return ConfigObject.load_from_yaml(os.environ.get("CONFIG_FILE", "config.yaml"))


# Load configuration instance when module is imported
instance: ConfigObject = load_config_from_args()


if "-c" in sys.argv:
    config_arg = sys.argv.index("-c")
    config_file = sys.argv[config_arg + 1]
    sys.argv.pop(config_arg + 1)
    sys.argv.pop(config_arg)
    os.environ["CONFIG_FILE"] = config_file


instance = ConfigObject.load_from_yaml(os.environ.get("CONFIG_FILE", "config.yaml"))

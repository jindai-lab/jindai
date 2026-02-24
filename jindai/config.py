"""Configuration management for Jindai application"""

import logging
import os
import sys
from typing import Any, Dict, List

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from jose import JWTError, jwt


class ConfigObject(BaseModel):
    """Configuration object for Jindai application settings.

    Supports additional fields beyond the defined ones for extensibility.
    Extra fields are accessible via the model_extra attribute.
    """
    model_config = ConfigDict(extra="allow")

    # defined fields
    concurrent: int = Field(default=3, description="Default concurrency level")
    storage: str = Field(description="Storage root directory path")
    database: str = Field(description="Database connection string")
    redis: str = Field(description="Redis connection string")
    plugins: List[str] = Field(
        default_factory=lambda: ["*"], description="List of plugins to load"
    )
    oidc: dict = Field(description="OIDC config")
    port: int = Field(default=8370, description="Default port for web server")
    embedding_model: str = Field(
        description="Sentence transformer model name for embeddings"
    )
    embedding_dims: int = Field(description="Embedding vector dimensions")

    ui_dist: str = Field(description="Path to UI distribution files", default='./dist/')
    paddle_remote: str = Field(description="PaddleOCR remote service URL")

    constants: dict = Field(default_factory=dict, description="Application constants")

    @field_validator("redis")
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

        logging.debug("* loaded config from", file_path)
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
config: ConfigObject = load_config_from_args()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=config.oidc["token_uri"])


class OIDCValidator:
    def __init__(self, config):
        self.jwks = None
        self.last_refresh = None
        self.config = config

    async def get_jwks(self):
        """动态获取 Authentik 公钥集"""
        if self.jwks is None:
            async with httpx.AsyncClient() as client:
                # 首先获取配置文档
                config_resp = await client.get(self.config["config_uri"])
                config_data = config_resp.json()
                # 得到 jwks_uri
                jwks_uri = config_data.get("jwks_uri")
                # 获取实际公钥
                jwks_resp = await client.get(jwks_uri)
                self.jwks = jwks_resp.json()
        return self.jwks

    async def validate_token(self, token: str = Depends(oauth2_scheme)):
        try:
            jwks = await self.get_jwks()
            # 解码并验证
            # jose 会自动从 jwks 中匹配对应的 kid (Key ID) 并验证签名
            payload = jwt.decode(
                token,
                jwks,
                algorithms=["RS256"],
                audience=self.config["client_id"],
                issuer=self.config["issuer"],
            )
            return payload
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token validation failed: {str(e)}",
            )


# 实例化验证器
oidc_validator = OIDCValidator(config.oidc)

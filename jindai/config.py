"""Configuration management for Jindai application.

This module provides:
- ConfigObject: Pydantic model for configuration validation
- load_config_from_args: Load config from command line arguments
- OIDCValidator: JWT token validation using OIDC
"""

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

    Attributes:
        concurrent: Default concurrency level for operations.
        storage: Storage root directory path.
        database: Database connection string.
        redis: Redis connection string.
        plugins: List of plugins to load (default: ["*"] for all).
        oidc: OIDC configuration dictionary.
        port: Default port for web server (default: 8370).
        embedding_model: Sentence transformer model name for embeddings.
        embedding_dims: Embedding vector dimensions.
        ui_dist: Path to UI distribution files (default: './dist/').
        paddle_remote: PaddleOCR remote service URL.
        constants: Application constants dictionary.
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
    cors_origins: List[str] = Field(
        default_factory=list, description="List of allowed CORS origins"
    )

    constants: dict = Field(default_factory=dict, description="Application constants")

    @field_validator("redis")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        """Remove trailing slash from Redis URL.

        Args:
            v: Redis URL string.

        Returns:
            Redis URL with trailing slash removed.
        """
        return v.rstrip("/")

    @classmethod
    def load_from_yaml(cls, file_path: str) -> "ConfigObject":
        """Load and parse configuration from YAML file.

        Args:
            file_path: Path to YAML configuration file.

        Returns:
            Validated configuration object.

        Raises:
            ValidationError: If YAML contains invalid configuration.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        logging.debug("* loaded config from", file_path)
        return cls.model_validate(data)

    def get_extra(self) -> Dict[str, Any]:
        """Get all extra configuration fields not defined in the class.

        Returns:
            Dictionary of extra configuration fields.
        """
        return self.model_extra or {}


def load_config_from_args() -> ConfigObject:
    """Load configuration from command line arguments.

    Parses command line for -c flag to specify config file path.

    Returns:
        Loaded configuration instance.
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
    """Validator for OIDC (OpenID Connect) tokens.

    Handles JWT token validation using JWKS (JSON Web Key Set) from
    the OIDC provider (Authentik).
    """

    def __init__(self, config: dict) -> None:
        """Initialize OIDC validator.

        Args:
            config: OIDC configuration dictionary containing:
                - config_uri: OIDC configuration endpoint
                - client_id: OAuth2 client ID
                - issuer: Token issuer URL
        """
        self.jwks = None
        self.last_refresh = None
        self.config = config

    async def get_jwks(self) -> dict:
        """Dynamically fetch Authentik public key set.

        Retrieves the JWKS (JSON Web Key Set) from the OIDC provider
        to validate JWT signatures.

        Returns:
            Dictionary containing the JWKS.
        """
        if self.jwks is None:
            async with httpx.AsyncClient() as client:
                # First get the configuration document
                config_resp = await client.get(self.config["config_uri"])
                config_data = config_resp.json()
                # Get the jwks_uri
                jwks_uri = config_data.get("jwks_uri")
                # Get actual public keys
                jwks_resp = await client.get(jwks_uri)
                self.jwks = jwks_resp.json()
        return self.jwks

    async def validate_token(self, token: str = Depends(oauth2_scheme)) -> dict:
        """Validate JWT token using OIDC provider's public keys.

        Args:
            token: JWT token to validate (injected by FastAPI dependency).

        Returns:
            Decoded JWT payload if validation succeeds.

        Raises:
            HTTPException: If token validation fails.
        """
        try:
            jwks = await self.get_jwks()
            # Decode and verify
            # jose automatically matches the kid (Key ID) from jwks and validates the signature
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


# Instantiate validator
oidc_validator = OIDCValidator(config.oidc)

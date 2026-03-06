"""API Web Service"""

import logging
import os
from typing import Any, Dict, Union

import uvicorn
from asteval import Interpreter
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, status
from fastapi.responses import FileResponse
from sqlalchemy import select

from .config import config
from .config import oidc_validator
from .helpers import get_context
from .models import UserInfo, get_db_session
from .storage import storage

app = FastAPI(
    docs_url="/api/v2/docs",
    openapi_url="/api/v2/openapi.json",
    title="Jindai",
    version="2.0.0",
)


router = APIRouter(
    prefix="/api/v2", dependencies=[Depends(oidc_validator.validate_token)]
)


wsrouter = APIRouter(
    prefix="/api/ws",
)


def get_current_username(
    token_payload: dict = Depends(oidc_validator.validate_token),
) -> str:
    return token_payload.get("preferred_username", "")


async def get_current_admin(
    token_payload: dict = Depends(oidc_validator.validate_token), username: str = ""
) -> UserInfo:
    """
    Verify if the current user has the admin role
    """
    # Retrieve username from JWT payload (corresponds to preferred_username)
    username = username or get_current_username(token_payload)

    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing username information",
        )

    async with get_db_session() as session:
        # Query database to verify role
        user = (
            await session.execute(
                select(UserInfo)
                .filter(
                    UserInfo.username == username, UserInfo.roles.contains(["admin"])
                )
                .limit(1)
            )
        ).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Must be admin. You are logged in as: {username}",
            )

        return user


async def serve_static(path: str = ""):
    search_paths = [
        path,
        os.path.join(config.ui_dist, path),
        os.path.join(config.ui_dist, "index.html"),
    ]
    
    logging.info('Searching for: ' + ', '.join(search_paths))

    for p in search_paths:
        if os.path.exists(p) and os.path.isfile(p):
            return FileResponse(p)

    raise HTTPException(status_code=404, detail=f"Not found for {path}")


def run_service(host: str = "0.0.0.0", port: int = 8370) -> None:
    """Run the FastAPI web service. Must run `prepare_plugins` first."""

    if port is None:
        port = config.port

    uvicorn.run(
        "jindai:app",  # Use string import path to support hot reload (reload)
        host=host,
        port=int(port),
        reload=True,  # Equivalent to debug=True
        workers=1,  # Typically set to 1 in development environment
        log_level="info",
    )


@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    return await serve_static(request.url.path.lstrip('/'))
    
"""API Web Service"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Union

import uvicorn
from asteval import Interpreter
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, select

from .config import config, oidc_validator
from .helpers import get_context
from .models import APIKey, UserInfo, get_db_session, redis_client
from .storage import storage

app = FastAPI(
    docs_url="/api/v2/docs",
    openapi_url="/api/v2/openapi.json",
    title="Jindai",
    version="2.0.687",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins if config.cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Key authentication constants
API_KEY_CACHE_PREFIX = "api_key:"
API_KEY_CACHE_TTL = 3600  # 1 hour in seconds


async def get_current_user(request: Request = None, websocket: WebSocket = None) -> Dict[str, Any]:
    """Get current user from either OAuth token or API key.
    
    Checks Authorization header for Bearer token (OAuth) or plain API key.
    Returns user info dict with username, roles, and user_id.
    
    Args:
        request: FastAPI request object.
        
    Returns:
        Dict with user info.
        
    Raises:
        HTTPException: If authentication fails.
    """
    if websocket is not None:
        auth_header = 'Bearer ' + websocket.query_params.get("token")
    else:
        auth_header = request.headers.get("Authorization", "")   
    # Check for Bearer token (OAuth)
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            token_payload = await oidc_validator.validate_token(token)
            username = token_payload.get("preferred_username", "")
            if not username:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token missing username information",
                )
            
            # Get user from database
            async with get_db_session() as session:
                result = await session.execute(
                    select(UserInfo).filter(UserInfo.username == username)
                )
                user = result.scalar_one_or_none()
                if not user:
                    # Check if this is the first user (table is empty)
                    count_result = await session.execute(select(func.count(UserInfo.id)))
                    total_count = count_result.scalar_one()
                    
                    # Create new user with appropriate roles
                    new_user = UserInfo(username=username)
                    if total_count == 0:
                        # First user gets admin role
                        new_user.roles = ["admin"]
                    else:
                        # Regular user gets default roles
                        new_user.roles = ["user"]
                    session.add(new_user)
                    await session.commit()
                    await session.refresh(new_user)
                    user = new_user
                return {
                    "username": username,
                    "roles": user.roles,
                    "user_id": str(user.id)
                }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    # Check for API key
    if auth_header.startswith("sk_"):
        api_key = auth_header[7:]
        key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        cache_key = f"{API_KEY_CACHE_PREFIX}{key_hash}"
        
        # Try to get from cache first
        cached = redis_client.get(cache_key)
        if cached:
            user_info = json.loads(cached)
            return user_info
        
        # Not in cache, query database
        async with get_db_session() as session:
            result = await session.execute(
                select(APIKey)
                .filter(
                    APIKey.key_hash == key_hash,
                    APIKey.is_active == True
                )
            )
            api_key_obj = result.scalar_one_or_none()
            
            if not api_key_obj:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                )
            
            # Check expiration
            if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key expired",
                )
            
            # Update last_used_at
            api_key_obj.last_used_at = datetime.utcnow()
            await session.commit()
            
            # Get user info
            user_result = await session.execute(
                select(UserInfo).filter(UserInfo.id == api_key_obj.user_id)
            )
            user = user_result.scalar_one_or_none()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                )
            
            user_info = {
                "username": user.username,
                "roles": user.roles,
                "user_id": str(user.id)
            }
            
            # Cache the result
            redis_client.setex(cache_key, API_KEY_CACHE_TTL, json.dumps(user_info))
            
            return user_info
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid authorization header",
    )


async def get_current_username(request: Request) -> str:
    """Get current username from either OAuth token or API key.
    
    Args:
        request: FastAPI request object.
        
    Returns:
        Username string.
    """
    user = await get_current_user(request)
    return user.get("username", "")


async def get_current_admin(request: Request = None, username: str = "") -> Dict[str, Any]:
    """
    Verify if the current user has the admin role.
    Supports both OAuth token and API key authentication.
    
    Args:
        request: FastAPI request object (optional for backward compatibility).
        username: Optional username to check (if not provided, uses current user).
        
    Returns:
        Dict with user info if user is admin.
        
    Raises:
        HTTPException: If authentication fails or user is not admin.
    """
    # Handle backward compatibility: if request is None or dict, use username directly
    if request is None or isinstance(request, dict):
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Username required when request is not provided",
            )
        # Query database directly
        async with get_db_session() as session:
            user = (
                await session.execute(
                    select(UserInfo)
                    .filter(
                        UserInfo.username == username, UserInfo.roles.contains(["admin"])
                    )
                    .limit(1)
                )
            ).scalar()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Must be admin. You are logged in as: {username}",
                )
            return user.as_dict()
    
    # New signature: request is a Request object
    if not username:
        username = await get_current_username(request)
    
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token/API key missing username information",
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
        ).scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Must be admin. You are logged in as: {username}",
            )

        return {"username": username, "roles": user.roles, "user_id": str(user.id)}


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
    # FastAPI 路由匹配失败的 404，detail 固定为 "Not Found"
    DEFAULT_404_DETAIL = "Not Found"
    
    # 1. 仅处理路由匹配失败的 404（detail 为默认值）
    if exc.detail == DEFAULT_404_DETAIL:
        return await serve_static(request.url.path.lstrip('/'))
    # 2. 手动抛出的 404（自定义 detail），保留原生 404 行为
    else:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
        

router = APIRouter(prefix="/api/v2", dependencies=[Depends(get_current_user)])

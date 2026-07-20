"""API Web Service"""

from contextlib import AsyncExitStack, asynccontextmanager
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

from .config import config
from .helpers import get_context
from .models import get_current_admin, get_current_user, get_current_username
from .storage import storage
from .resources import combined_lifespan, router

app = FastAPI(
    docs_url="/api/v2/docs",
    openapi_url="/api/v2/openapi.json",
    title="Jindai",
    version="2.0.698",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins if config.cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# mount lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(combined_lifespan(app))
        print('stacked')
        yield


app.router.lifespan_context = lifespan     



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
    # 404 when FastAPI route matching fails, detail is fixed to "Not Found"
    DEFAULT_404_DETAIL = "Not Found"
    
    # 1. Only handle 404 from route matching failure (detail is default value)
    if exc.detail == DEFAULT_404_DETAIL:
        return await serve_static(request.url.path.lstrip('/'))
    # 2. Manually raised 404 (custom detail), preserve original 404 behavior
    else:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
        

router = APIRouter(prefix="/api/v2", dependencies=[Depends(get_current_user)])

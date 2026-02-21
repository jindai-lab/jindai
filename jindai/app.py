"""API Web Service"""

from contextlib import asynccontextmanager
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Union

import httpx
from sqlalchemy import select
import uvicorn
from asteval import Interpreter
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from .config import instance as config
from .helpers import get_context
from .models import UserInfo, get_db_session
from .storage import instance as storage
from .worker import AsyncWorkerManager


app = FastAPI(
    docs_url="/api/v2/docs",
    openapi_url="/api/v2/openapi.json",
    title="Jindai",
    version="2.0.0",
)


CLIENT_ID = config.oidc["client_id"]
OIDC_CONFIG_URL = config.oidc["config_uri"]


oauth2_scheme = OAuth2PasswordBearer(tokenUrl=config.oidc["token_uri"])


class OIDCValidator:
    def __init__(self):
        self.jwks = None
        self.last_refresh = None

    async def get_jwks(self):
        """动态获取 Authentik 公钥集"""
        if self.jwks is None:
            async with httpx.AsyncClient() as client:
                # 首先获取配置文档
                config_resp = await client.get(OIDC_CONFIG_URL)
                config_data = config_resp.json()
                # 得到 jwks_uri
                jwks_uri = config_data.get("jwks_uri")
                # 获取实际公钥
                jwks_resp = await client.get(jwks_uri)
                self.jwks = jwks_resp.json()
        return self.jwks

    async def validate_token(self, token: str = Depends(oauth2_scheme)):
        """代替 oidc.accept_token()"""
        try:
            jwks = await self.get_jwks()
            # 解码并验证
            # jose 会自动从 jwks 中匹配对应的 kid (Key ID) 并验证签名
            payload = jwt.decode(
                token,
                jwks,
                algorithms=["RS256"],
                audience=CLIENT_ID,
                issuer=config.oidc["issuer"],
            )
            return payload
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token validation failed: {str(e)}",
            )


# 实例化验证器
oidc_validator = OIDCValidator()


router = APIRouter(
    prefix="/api/v2", dependencies=[Depends(oidc_validator.validate_token)]
)


def aeval(expr: str, context: Union[Dict[str, Any], Any]) -> Any:
    """Evaluate an expression in a safe context

    :param expr: Expression to evaluate
    :type expr: str
    :param context: Context object for evaluation
    :type context: dict or object with as_dict() method
    :return: Evaluation result
    :rtype: Any
    """
    if not isinstance(context, dict):
        context = context.as_dict()
    ee = Interpreter(context)
    result = ee(expr)
    # Return result as is, since asteval can return various types
    return result


def get_current_username(token_payload: dict = Depends(oidc_validator.validate_token)) -> str:
    return token_payload.get("preferred_username", "")


async def get_current_admin(
    token_payload: dict = Depends(oidc_validator.validate_token),
    username: str = ''
) -> UserInfo:
    """
    验证当前用户是否具有 admin 角色
    """
    # 从 JWT payload 中获取用户名 (对应之前的 preferred_username)
    username = username or get_current_username(token_payload)

    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing username information",
        )

    async with get_db_session() as session:
        # 查询数据库校验角色
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


UI_DIST = config.ui_dist or "./dist/"


async def serve_static(path: str = "index.html"):
    # 过滤 api 请求，防止误触（虽然路由优先级通常能处理，但手动过滤更稳）
    if path.startswith("api/"):
        raise HTTPException(status_code=404)

    # 模拟原有的路径尝试逻辑
    search_paths = [
        path,
        f"{path}.html",
        os.path.join(UI_DIST, path),
        os.path.join(UI_DIST, f"{path}.html"),
    ]

    for p in search_paths:
        if os.path.exists(p) and os.path.isfile(p):
            return FileResponse(p)

    # 如果是 SPA（单页应用），通常找不到路径时返回 index.html
    index_path = os.path.join(UI_DIST, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)

    raise HTTPException(status_code=404, detail=f"Not found for {path}")


def run_service(host: str = "0.0.0.0", port: int = 8370) -> None:
    """运行 FastAPI Web 服务。必须先运行 `prepare_plugins`。"""

    if port is None:
        port = config.port

    uvicorn.run(
        "jindai:app",  # 建议使用 字符串导入路径 以支持热重载 (reload)
        host=host,
        port=int(port),
        reload=True,  # 相当于 debug=True
        workers=1,  # 开发环境下通常设为 1
        log_level="info",
    )

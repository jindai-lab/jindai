"""API Web Service"""

import os
from typing import Any, Dict, Union

import uvicorn
from asteval import Interpreter
from fastapi import APIRouter, Depends, FastAPI, HTTPException, WebSocket, status
from fastapi.responses import FileResponse
from sqlalchemy import select

from .config import instance as config
from .config import oidc_validator
from .helpers import get_context
from .models import UserInfo, get_db_session
from .storage import instance as storage

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


def get_current_username(
    token_payload: dict = Depends(oidc_validator.validate_token),
) -> str:
    return token_payload.get("preferred_username", "")


async def get_current_admin(
    token_payload: dict = Depends(oidc_validator.validate_token), username: str = ""
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


async def serve_static(path: str = "index.html"):
    # 过滤 api 请求，防止误触（虽然路由优先级通常能处理，但手动过滤更稳）
    if path.startswith("api/"):
        raise HTTPException(status_code=404)

    # 模拟原有的路径尝试逻辑
    search_paths = [
        path,
        f"{path}.html",
        os.path.join(config.ui_dist, path),
        os.path.join(config.ui_dist, f"{path}.html"),
    ]

    for p in search_paths:
        if os.path.exists(p) and os.path.isfile(p):
            return FileResponse(p)

    # 如果是 SPA（单页应用），通常找不到路径时返回 index.html
    index_path = os.path.join(config.ui_dist, "index.html")
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

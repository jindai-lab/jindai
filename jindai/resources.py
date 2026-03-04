import asyncio
import json
import os
import httpx
import logging
from typing import Optional, Type, List, Any, Dict
from contextlib import asynccontextmanager, AsyncExitStack

from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    Request,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.sql import func, select
from sqlalchemy.ext.asyncio import AsyncSession

# 导入原有组件
from .app import get_current_admin, get_current_username, router, wsrouter, app
from .config import config
from .maintenance import maintenance_manager
from .models import (
    Base,
    Dataset,
    EmbeddingPendingQueue,
    History,
    Paragraph,
    QueryFilters,
    TaskDBO,
    TextEmbeddings,
    UserInfo,
    get_db_session,
)
from .plugin import plugins
from .storage import storage
from .worker import worker_manager

# --- 基础工具与依赖 ---

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def get_db():
    async with get_db_session() as session:
        yield session


# --- 面向对象资源管理基类 ---


class ResourceRegistry:
    """自动处理 APIRouter 注册与权限的基类"""

    def __init__(self, model: Type[Base], prefix: str, tags: List[str]):
        self.model = model
        self.prefix = prefix
        self.tags = tags

    async def get_auth_filters(self, username: str, permission: str = "r"):
        """默认权限：管理员全开，普通用户仅限本人数据"""
        if await get_current_admin({}, username):
            return True
        return self.model.user.username == username

    async def paginate(self, session: AsyncSession, stmt, sort : str, offset: int, limit: int):
        total_stmt = select(func.count()).select_from(stmt.order_by(None).subquery())
        total = (await session.execute(total_stmt)).scalar()
        stmt = stmt.offset(offset).limit(limit)
        if sorts := self.model.parse_sort_string(sort):
            stmt = stmt.order_by(*sorts)
        results = (
            (await session.execute(stmt)).scalars().all()
        )
        return {"total": total, "results": [r.as_dict() for r in results]}

    def register(self, parent_router: APIRouter, dependencies: List[Any] = None):
        res_router = APIRouter(
            prefix=self.prefix, tags=self.tags, dependencies=dependencies
        )

        @res_router.get("/")
        async def list_items(
            offset: int = 0,
            limit: int = 100,
            sort: str = Query(''),
            session: AsyncSession = Depends(get_db),
            username: str = Depends(get_current_username),
        ):
            stmt = select(self.model).filter(await self.get_auth_filters(username))
            return await self.paginate(session, stmt, sort, offset, limit)

        @res_router.get("/{resource_id}")
        async def get_item(
            resource_id: str,
            session: AsyncSession = Depends(get_db),
            username: str = Depends(get_current_username),
        ):
            filters = await self.get_auth_filters(username, "r")
            item = (
                await session.execute(
                    select(self.model).filter(filters, self.model.id == resource_id)
                )
            ).scalar_one_or_none()
            if not item:
                raise HTTPException(404, detail="Resource not found")
            return item.as_dict()

        @res_router.put("/{resource_id}")
        async def put_item(
            resource_id: str,
            data: dict = Body(...),
            session: AsyncSession = Depends(get_db),
            username: str = Depends(get_current_username),
        ):
            filters = await self.get_auth_filters(username, "w")
            item = (
                await session.execute(
                    select(self.model).filter(filters, self.model.id == resource_id)
                )
            ).scalar_one_or_none()
            if not item:
                raise HTTPException(404)
            for k, v in data.items():
                setattr(item, k, v)
            await session.commit()
            return item.as_dict()

        @res_router.delete("/{resource_id}")
        async def delete_item(
            resource_id: str,
            session: AsyncSession = Depends(get_db),
            username: str = Depends(get_current_username),
        ):
            filters = await self.get_auth_filters(username, "w")
            item = (
                await session.execute(
                    select(self.model).filter(filters, self.model.id == resource_id)
                )
            ).scalar_one_or_none()
            if not item:
                raise HTTPException(404)
            await session.delete(item)
            await session.commit()
            return {"message": "Deleted"}

        parent_router.include_router(res_router)


class TaskResource(ResourceRegistry):
    def __init__(self):
        super().__init__(TaskDBO, "/tasks", ["Tasks"])

    async def get_auth_filters(self, username: str, permission="r"):
        filters = await super().get_auth_filters(username, permission)
        if filters is not True and permission == "r":
            # 扩展权限：允许查看共享任务
            filters = (filters) | (TaskDBO.shared == True)
        return filters


# --- 核心业务逻辑管理类 ---


class EmbeddingManager:
    """处理 Embedding 统计与后台轮询"""

    def __init__(self):
        self.polling_task: Optional[asyncio.Task] = None

    async def polling_loop(self):
        logging.info("Embedding Polling Loop started.")
        try:
            while True:
                async with get_db_session() as session:
                    pending = await session.execute(
                        select(EmbeddingPendingQueue).limit(1)
                    )
                    has_pending = pending.first() is not None
                if has_pending:
                    logging.info("Pending queue not empty")
                    await maintenance_manager.update_text_embeddings()
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            logging.info("Embedding Polling Loop is being cancelled.")

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        self.polling_task = asyncio.create_task(self.polling_loop())
        yield
        if self.polling_task:
            self.polling_task.cancel()
            await asyncio.gather(self.polling_task, return_exceptions=True)

    def register_routes(self, router: APIRouter):
        @router.get("/embeddings/", tags=["Embeddings"])
        async def stat_embeddings(session: AsyncSession = Depends(get_db)):
            finished = (
                await session.execute(
                    select(TextEmbeddings)
                    .filter(TextEmbeddings.chunk_id == 1)
                    .with_only_columns(func.count(1))
                )
            ).scalar()
            queued = (
                await session.execute(
                    select(EmbeddingPendingQueue).with_only_columns(
                        func.count(EmbeddingPendingQueue.id)
                    )
                )
            ).scalar()
            return {"finished": finished, "queued": queued}

        @router.post("/embeddings/{resource_id}", status_code=201, tags=["Embeddings"])
        async def update_single_embedding(
            resource_id: str, session: AsyncSession = Depends(get_db)
        ):
            para = await session.get(Paragraph, resource_id)
            if not para or not para.content:
                raise HTTPException(404)
            emb_val = await TextEmbeddings.get_embedding(para.content)
            te = (
                await session.execute(
                    select(TextEmbeddings).filter(TextEmbeddings.id == para.id)
                )
            ).scalar_one_or_none()
            if te:
                te.embedding = emb_val
            else:
                session.add(TextEmbeddings(id=resource_id, embedding=emb_val))
            await session.commit()
            return {"id": resource_id}


class ContentManager(ResourceRegistry):
    """处理数据集与段落逻辑"""
    
    def __init__(self):
        super().__init__(Paragraph, '', [])

    def register_routes(self, router: APIRouter):
        @router.get("/datasets", tags=["Datasets"])
        @router.get("/datasets/{resource_id}", tags=["Datasets"])
        async def get_datasets(resource_id: Optional[str] = None):
            if not resource_id:
                return {"results": await Dataset.get_hierarchy()}
            res = await Dataset.get(resource_id)
            if not res:
                raise HTTPException(404)
            return res.as_dict()

        @router.put(
            "/datasets/{resource_id}",
            tags=["Datasets"],
            dependencies=[Depends(get_current_admin)],
        )
        async def rename_dataset(resource_id: str, name: str = Body(embed=True)):
            ds = await Dataset.get(resource_id, False)
            if not ds:
                raise HTTPException(404)
            return await ds.rename_dataset(name)

        @router.delete(
            "/datasets/{resource_id}",
            tags=["Datasets"],
            dependencies=[Depends(get_current_admin)],
        )
        async def delete_dataset(
            resource_id: str, session: AsyncSession = Depends(get_db)
        ):
            ds = await Dataset.get(resource_id, False)
            if not ds:
                raise HTTPException(404)
            await session.delete(ds)
            await session.commit()
            return {"message": "Dataset deleted"}

        @router.post("/paragraphs", tags=["Paragraphs"])
        async def post_paragraphs(
            data: dict = Body(...), session: AsyncSession = Depends(get_db)
        ):
            new_para = Paragraph.from_dict(data)
            session.add(new_para)
            await session.flush()
            await session.commit()
            return new_para.as_dict()

        @router.post("/paragraphs/search")
        async def search_paragraphs(
            filters: QueryFilters,
            session: AsyncSession = Depends(get_db),
            username: str = Depends(get_current_username),
        ):
            limit, offset = filters.limit, filters.offset
            filters.offset, filters.limit = 0, 0
            hist = History(
                user_id=(
                    await session.execute(
                        select(UserInfo).filter(UserInfo.username == username)
                    )
                )
                .scalar_one()
                .id,
                queries=json.loads(filters.model_dump_json()),
            )
            session.add(hist)
            await session.commit()
            query = await Paragraph.build_query(filters)
            resp = await self.paginate(session, query, '', offset or 0, limit or 0)
            resp['query'] = str(query.compile())
            return resp

        @router.post("/paragraphs/filters/{column}")
        async def filter_paragraphs_items(
            column: str, filters: QueryFilters, session: AsyncSession = Depends(get_db)
        ):
            filters.q, filters.embeddings, filters.sort, filters.groupBy = (
                "*",
                None,
                "",
                "",
            )
            setattr(filters, column, "")
            query = await Paragraph.build_query(filters)
            col_attr = getattr(
                Paragraph, "source_url" if column == "sources" else column
            )
            query = query.with_only_columns(
                col_attr.label("value"), func.count(1).label("count")
            ).group_by(col_attr)
            return (await session.execute(query)).mappings().all() or []


class StorageManager:
    """文件存储与上传下载"""

    def register_routes(self, router: APIRouter):
        @router.get("/files/{file_path:path}", tags=["Files"])
        def get_file(
            file_path: str = "",
            search: str = "",
            metadata: bool = False,
            page: Optional[int] = None,
            format: Optional[str] = None,
        ):
            try:
                target_path = storage.safe_join(file_path)
            except ValueError:
                raise HTTPException(403)
            if not os.path.exists(target_path):
                raise HTTPException(404)
            if os.path.isdir(target_path):
                return (
                    storage.search(target_path, search, detailed=metadata)
                    if search
                    else storage.ls(target_path, detailed=metadata)
                )
            if metadata:
                return storage.fileinfo(target_path)
            buf, mime, name = storage.read_file(file_path, page, format)
            return StreamingResponse(buf, media_type=mime)

        @router.post("/files/{file_path:path}", tags=["Files"])
        def upload_file(
            file_path: str = "",
            is_directory: bool = False,
            name: str = "",
            file: UploadFile = File(None),
        ):
            if is_directory and name:
                return storage.mkdir(file_path, name)
            if not file:
                raise HTTPException(400, detail="No file")
            return storage.save(file.file, file_path)


# --- 路由组装与生命周期绑定 ---

emb_manager = EmbeddingManager()
app.router.lifespan_context = emb_manager.lifespan

# 注册通用 CRUD 资源
ResourceRegistry(UserInfo, "/users", ["Users"]).register(
    router, dependencies=[Depends(get_current_admin)]
)
ResourceRegistry(History, "/histories", ["Histories"]).register(router)
TaskResource().register(router)

# 注册业务逻辑模块
ContentManager().register_routes(router)
emb_manager.register_routes(router)
StorageManager().register_routes(router)


# 辅助功能
@router.post("/translator", tags=["Translator"])
async def translator(params: dict = Body(...)):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    payload = {
        "model": "glm-4.7-flash",
        "messages": [
            {
                "role": "system",
                "content": f"你是一个AI翻译器。翻译文本为{params['lang']}，只返回翻译结果。",
            },
            {"role": "user", "content": params["text"]},
        ],
        "stream": False,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {params['zhipu_api_key']}"},
            timeout=30,
        )
        return response.json()["choices"][0]["message"]["content"]


# 4. 外部管理器集成
router.include_router(plugins.get_router())
router.include_router(maintenance_manager.get_router())
router.include_router(worker_manager.get_router())
wsrouter.include_router(worker_manager.get_wsrouter())
app.include_router(router)
app.include_router(wsrouter)


# 5. Worker 任务注册
tasks_to_reg = [
    (maintenance_manager.custom_task, "custom"),
    (maintenance_manager.ocr, "ocr"),
    (maintenance_manager.update_text_embeddings, "text_embedding"),
    (maintenance_manager.sync_terms, "sync_terms"),
    (maintenance_manager.update_pdate_from_url, "sync_pdate"),
    (maintenance_manager.sync_sources, "sync_sources"),
    (maintenance_manager.test_task, "test_task"),
]
for func_ref, name in tasks_to_reg:
    worker_manager.register_task(func_ref, name)

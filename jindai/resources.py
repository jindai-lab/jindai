import os
import uuid
import httpx
from typing import Optional, Type

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.sql import func, select

from .app import get_current_admin, get_current_username, router
from .config import instance as config
from .storage import instance as storage
from .maintenance import maintenance_manager
from .worker import worker_manager
from .models import (
    AsyncSession,
    Base,
    Dataset,
    History,
    Paragraph,
    QueryFilters,
    TaskDBO,
    TextEmbeddings,
    UserInfo,
    get_db_session,
    is_uuid_literal,
)
from .plugin import plugins


async def get_db():
    async with get_db_session() as session:
        yield session


class JindaiResource:

    def __init__(self, cls: Type[Base], prefix: str, tags: list) -> None:
        self.model = cls
        self.prefix = prefix
        self.tags = tags

    @staticmethod
    async def paginate(session, stmt, model_cls, offset, limit):
        total = await session.execute(
            select(func.count()).select_from(stmt.order_by(None).subquery())
        )
        total = total.scalar()
        results = await session.execute(stmt.offset(offset).limit(limit))
        results = results.scalars().all()
        return {"total": total, "results": [r.as_dict() for r in results]}

    @staticmethod
    def get_pagination(offset: int = 0, limit: int = 100):
        return {"offset": offset, "limit": limit}

    async def auth_filters(self, username: str, permission : str = 'r'):
        if await get_current_admin({}, username):
            return True
        return self.model.user.username == username

    def create_router(self):
        router = APIRouter(prefix=self.prefix, tags=self.tags)

        @router.get("/")
        async def list_items(
            paging: dict = Depends(JindaiResource.get_pagination),
            session=Depends(get_db),
            username=Depends(get_current_username),
        ):
            stmt = select(self.model).filter(await self.auth_filters(username))
            return await JindaiResource.paginate(
                session, stmt, self.model, paging["offset"], paging["limit"]
            )

        @router.put("/{resource_id}")
        async def put_item(
            resource_id: str, data: dict = Body(...), session=Depends(get_db)
        ):
            item = await session.get(
                self.model,
                resource_id if is_uuid_literal(resource_id) else resource_id,
            )
            if not item:
                raise HTTPException(404, detail="Resource not found")
            for k, v in data.items():
                setattr(item, k, v)

        @router.get("/{resource_id}")
        async def get_item(resource_id: str, session=Depends(get_db)):
            item = await session.get(
                self.model,
                resource_id if is_uuid_literal(resource_id) else resource_id,
            )
            if not item:
                raise HTTPException(404, detail="Resource not found")
            return item.as_dict()

        @router.delete("/{resource_id}")
        async def delete_item(
            resource_id: str,
            session=Depends(get_db),
            username=Depends(get_current_username),
        ):
            item = (await session.execute(
                select(self.model)
                .filter(await self.auth_filters(username, 'w'))
                .filter(self.model.id == resource_id)
            )).first()
            if not item:
                raise HTTPException(404)
            await session.delete(item)
            return {"message": "Deleted"}

        return router


class UserInfoResource(JindaiResource):

    def __init__(self) -> None:
        super().__init__(UserInfo, "/users", ["Users"])


class HistoryResource(JindaiResource):

    def __init__(self) -> None:
        super().__init__(UserInfo, "/histories", ["Histories"])


class TaskDBOResource(JindaiResource):

    def __init__(self) -> None:
        super().__init__(TaskDBO, "/tasks", ["Tasks"])
    
    async def auth_filters(self, username: str, permission = 'r'):
        filters = await super().auth_filters(username, permission)
        if filters is not True and permission == 'r':
            filters |= TaskDBO.shared == True
        return filters


router.include_router(
    UserInfoResource().create_router(), dependencies=[Depends(get_current_admin)]
)

router.include_router(HistoryResource().create_router())

router.include_router(TaskDBOResource().create_router())


@router.get("/datasets", tags=["Datasets"])
async def get_datasets(resource_id: Optional[str] = None, session=Depends(get_db)):
    if not resource_id:
        return {"results": await Dataset.get_hierarchy()}

    res = await Dataset.get(resource_id)
    if not res:
        raise HTTPException(404)
    return res.as_dict()


@router.put("/datasets/{resource_id}", tags=["Datasets"])
async def rename_dataset(resource_id: str, name: str):
    ds = await Dataset.get(resource_id, True)
    if not ds:
        raise HTTPException(404)
    return await ds.rename_dataset(name)


@router.get("/embeddings/", tags=["Embeddings"])
async def stat_embeddings(session: AsyncSession = Depends(get_db)):
    result = await session.execute(
        select(TextEmbeddings)
        .filter(TextEmbeddings.chunk_id == 1)
        .with_only_columns(func.count(1))
    )
    return result.scalar()


@router.post("/embeddings/{resource_id}", status_code=201, tags=["Embeddings"])
async def update_embedding(
    resource_id: str,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
):

    async def background_embedding(resource_id, session):
        para = await session.get(Paragraph, resource_id)
        if not para:
            raise HTTPException(404)

        if para.content:
            emb_val = await TextEmbeddings.get_embedding(para.content)
            te = await session.get(TextEmbeddings, resource_id)
            if te:
                te.embedding = emb_val
            else:
                session.add(TextEmbeddings(id=resource_id, embedding=emb_val))

    background_tasks.add_task(background_embedding, resource_id, session)
    return {"id": resource_id}


@router.post("/paragraphs", tags=["Paragraphs"])
async def post_paragraphs(
    data: dict = Body(...),
    session: AsyncSession = Depends(get_db),
):
    new_para = Paragraph.from_dict(data)
    session.add(new_para)
    await session.flush()
    return new_para.as_dict()


@router.post("/paragraphs/search", tags=["Paragraphs"])
async def search_paragraphs(
    filters: QueryFilters,
    paging: dict = Depends(JindaiResource.get_pagination),
    session: AsyncSession = Depends(get_db),
):
    filters.offset = 0
    filters.limit = None
    query = await Paragraph.build_query(filters)
    resp = await JindaiResource.paginate(session, query, Paragraph, paging["offset"], paging["limit"])
    return resp


@router.post("/paragraphs/filters/{column}", tags=["Paragraphs"])
async def filter_paragraphs_items(
    column: str, filters: QueryFilters, session: AsyncSession = Depends(get_db)
):
    filters.q = '*'
    filters.embeddings = None
    filters.sort = ""
    filters.groupBy = ""
    setattr(filters, column, "")

    query = await Paragraph.build_query(filters)
    column = getattr(Paragraph, column)
    query = query.with_only_columns(
        column.label("value"), func.count(1).label("count")
    ).group_by(column)
    results = await session.execute(query)
    return results.mappings().all() or []


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

    # 下载文件
    buf, mime, name = storage.read_file(file_path, page, format)
    return StreamingResponse(
        buf,
        media_type=mime,
        # headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


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


@router.post("/translator", tags=["Translator"])
async def translator(params: dict = Body(...)):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    payload = {
        "model": "glm-4.7-flash",
        "messages": [
            {
                "role": "system",
                "content": f"你是一个AI翻译器。翻译用户输入的文本为{params['lang']}，只返回翻译后的文本。",
            },
            {"role": "user", "content": params["text"]},
        ],
        "stream": False,
        "temperature": 1,
    }
    headers = {"Authorization": "Bearer " + params["zhipu_api_key"]}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, timeout=30000)
        return response.json()["choices"][0]["message"]["content"]


router.include_router(plugins.get_router())
router.include_router(maintenance_manager.get_router())
router.include_router(worker_manager.get_router())

# register worker tasks
worker_manager.register_task(maintenance_manager.custom_task, "custom")
worker_manager.register_task(maintenance_manager.ocr, "ocr")
worker_manager.register_task(
    maintenance_manager.update_text_embeddings, "text_embedding"
)
worker_manager.register_task(maintenance_manager.sync_terms, "sync_terms")
worker_manager.register_task(maintenance_manager.update_pdate_from_url, "sync_pdate")
worker_manager.register_task(maintenance_manager.sync_sources, "sync_sources")

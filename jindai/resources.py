import hashlib
import os
from typing import Any, Optional, Type

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.sql import func, select
import uuid

from .app import get_current_admin, router, storage
from .models import (
    Base,
    Dataset,
    History,
    Paragraph,
    TaskDBO,
    TextEmbeddings,
    UserInfo,
    is_uuid_literal,
    get_db,
    AsyncSession
)
from .worker import add_task, clear_tasks, get_task_result, get_task_stats, revoke_task


def get_pagination(offset: int = 0, limit: int = 100):
    return {"offset": offset, "limit": limit}


async def paginate(session, stmt, model_cls, offset, limit):
    total = await session.execute(select(func.count()).select_from(stmt.subquery()))
    total = total.scalar()
    results = await session.execute(stmt.offset(offset).limit(limit))
    results = results.scalars().all()
    return {"total": total, "results": [r.as_dict() for r in results]}


def create_generic_router(model_cls: Type[Base], prefix: str, tags: list):
    generic_router = APIRouter(prefix=prefix, tags=tags)

    @generic_router.get("/")
    async def list_items(params: dict = Depends(get_pagination), session = Depends(get_db)):
        stmt = select(model_cls)
        return await paginate(
            session, stmt, model_cls, params["offset"], params["limit"]
        )
            
    @generic_router.put("/{resource_id}")
    async def put_item(resource_id: str, data: dict = Body(...), session = Depends(get_db)):
        item = await session.get(
            model_cls,
            resource_id if is_uuid_literal(resource_id) else resource_id,
        )
        if not item:
            raise HTTPException(404, detail="Resource not found")
        for k, v in data.items():
            setattr(item, k, v)
                        
    @generic_router.get("/{resource_id}")
    async def get_item(resource_id: str, session = Depends(get_db)):
        item = await session.get(
            model_cls,
            resource_id if is_uuid_literal(resource_id) else resource_id,
        )
        if not item:
            raise HTTPException(404, detail="Resource not found")
        return item.as_dict()

    @generic_router.delete("/{resource_id}")
    async def delete_item(
        resource_id: str, admin: UserInfo = Depends(get_current_admin), session = Depends(get_db)
    ):
        item = await session.get(model_cls, uuid.UUID(resource_id))
        if not item:
            raise HTTPException(404)
        await session.delete(item)
        return {"message": "Deleted"}

    return generic_router


# 注册基础资源 (UserInfo, History, TaskDBO)
router.include_router(create_generic_router(UserInfo, "/users", ["Users"]))
router.include_router(create_generic_router(History, "/histories", ["Histories"]))
router.include_router(create_generic_router(TaskDBO, "/tasks", ["Tasks"]))


@router.get("/datasets", tags=["Datasets"])
async def get_datasets(resource_id: Optional[str] = None, session = Depends(get_db)):
    if not resource_id:
        return {"results": await Dataset.get_hierarchy()}
    
    res = await Dataset.get(resource_id)
    if not res:
        raise HTTPException(404)
    return res.as_dict()


@router.put("/datasets/{resource_id}", tags=["Datasets"])
async def rename_dataset(resource_id: str, name: str = Body(..., embed=True)):
    ds = await Dataset.get(resource_id, True)
    if not ds:
        raise HTTPException(404)
    return await ds.rename_dataset(name)


@router.get("/embeddings")
async def stat_embeddings(session : AsyncSession = Depends(get_db)):
    result = await session.execute(
        select(TextEmbeddings).filter(TextEmbeddings.chunk_id == 1).with_only_columns(func.count(1))
    )
    return result.scalar()


@router.post("/embeddings/{resource_id}", status_code=201)
async def update_embedding(resource_id: str, session : AsyncSession = Depends(get_db)):
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
    return {"id": resource_id}


@router.post("/paragraphs", tags=["Paragraphs"])
async def post_paragraphs(
    data: dict = Body(...), params: dict = Depends(get_pagination), session : AsyncSession = Depends(get_db)
):
    if data.get("search"):
        query = await Paragraph.build_query(data)
        return await paginate(
            session, query, Paragraph, params["offset"], params["limit"]
        )

    new_para = Paragraph.from_dict(data)
    session.add(new_para)
    await session.flush()
    return new_para.as_dict()


@router.get("/files/{file_path:path}")
async def get_file(
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
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.post("/files/{file_path:path}")
def upload_file(
    file_path: str = "",
    is_directory: bool = Body(False),
    name: str = Body(""),
    file: Optional[UploadFile] = File(None),
):
    if is_directory and name:
        return storage.mkdir(file_path, name)

    if not file:
        raise HTTPException(400, detail="No file")
    return storage.save(file.file, file_path)


@router.get("/worker/{task_id}")
def task_status(task_id: str = ""):
    return get_task_result(task_id) if task_id else get_task_stats()


@router.post("/worker")
def start_task(payload: dict = Body(...)):
    return add_task(**payload)


@router.delete("/worker")
def clear_task():
    return clear_tasks()


@router.delete("/worker/{task_id}")
def task_status(task_id: str = ""):
    return revoke_task(task_id)


@router.get("/worker")
def task_stats():
    return get_task_stats()
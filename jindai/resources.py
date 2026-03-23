"""Resource management and API routes for Jindai application.

This module provides:
- ResourceRegistry: Base class for CRUD resource endpoints
- TaskResource: Task resource with shared access support
- EmbeddingManager: Embedding processing and polling
- ContentManager: Dataset and paragraph management
- StorageManager: File upload/download operations
"""

import asyncio
import json
import base64
import struct
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

# Import existing components
from .app import get_current_admin, get_current_username, router, app
from .config import config
from .maintenance import maintenance_manager
from .worker import worker_manager
from .models import (
    APIKey,
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
    get_db,
    generate_api_key,
    hash_api_key
)
from .plugin import plugins
from .storage import storage

# --- Basic utilities and dependencies ---

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# --- Resource management base class ---

class ResourceRegistry:
    """Base class for automatic APIRouter registration with permissions.

    Provides CRUD endpoints for database models with automatic
    authentication and authorization filtering.
    """

    def __init__(self, model: Type[Base], prefix: str, tags: List[str]):
        """Initialize resource registry.

        Args:
            model: SQLAlchemy model class.
            prefix: URL prefix for endpoints.
            tags: Tags for API documentation.
        """
        self.model = model
        self.prefix = prefix
        self.tags = tags

    async def get_auth_filters(self, username: str, permission: str = "r") -> Any:
        """Get authentication filters for user access.

        Default: Admin has full access, regular users only see their own data.

        Args:
            username: Current username.
            permission: Permission type ('r' for read, 'w' for write).

        Returns:
            SQLAlchemy filter expression.
        """
        
        # Admin has full access
        if await get_current_admin({}, username):
            return True
        
        # Check if model has user relationship
        if hasattr(self.model, 'user'):
            return self.model.user.username == username
        elif hasattr(self.model, 'user_id'):
            return self.model.user_id == UserInfo.id
        else:
            # For models without user relationship, allow access
            return True

    async def paginate(
        self,
        session: AsyncSession,
        stmt,
        sort: str,
        offset: int,
        limit: int
    ) -> dict:
        """Paginate query results.

        Args:
            session: Database session.
            stmt: SQLAlchemy select statement.
            sort: Sort string.
            offset: Offset for pagination.
            limit: Limit for pagination.

        Returns:
            Dictionary with total count and results.
        """
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
        """Register CRUD endpoints with parent router.

        Args:
            parent_router: Parent APIRouter to include endpoints.
            dependencies: Additional dependencies for endpoints.
        """
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
            """List items with pagination.

            Args:
                offset: Offset for pagination.
                limit: Limit for pagination.
                sort: Sort string.
                session: Database session.
                username: Current username.

            Returns:
                Dictionary with total and results.
            """
            stmt = select(self.model).filter(await self.get_auth_filters(username))
            return await self.paginate(session, stmt, sort, offset, limit)

        @res_router.get("/{resource_id}")
        async def get_item(
            resource_id: str,
            session: AsyncSession = Depends(get_db),
            username: str = Depends(get_current_username),
        ):
            """Get single item by ID.

            Args:
                resource_id: Item ID.
                session: Database session.
                username: Current username.

            Returns:
                Item as dictionary.

            Raises:
                HTTPException: If item not found.
            """
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
            """Update item.

            Args:
                resource_id: Item ID.
                data: Update data.
                session: Database session.
                username: Current username.

            Returns:
                Updated item as dictionary.

            Raises:
                HTTPException: If item not found.
            """
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
            """Delete item.

            Args:
                resource_id: Item ID.
                session: Database session.
                username: Current username.

            Returns:
                Deletion confirmation.

            Raises:
                HTTPException: If item not found.
            """
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
    """Task resource with shared access support."""

    def __init__(self):
        """Initialize task resource."""
        super().__init__(TaskDBO, "/tasks", ["Tasks"])

    async def get_auth_filters(self, username: str, permission: str = "r") -> Any:
        """Get authentication filters for task access.

        Extends base to allow viewing shared tasks.

        Args:
            username: Current username.
            permission: Permission type.

        Returns:
            SQLAlchemy filter expression.
        """
        filters = await super().get_auth_filters(username, permission)
        if filters is not True and permission == "r":
            # Extended permission: allow viewing shared tasks
            filters = (filters) | (TaskDBO.shared == True)
        return filters


# --- Core business logic managers ---

class EmbeddingManager:
    """Handles embedding statistics and background polling."""

    def __init__(self):
        self.polling_task: Optional[asyncio.Task] = None

    async def polling_loop(self):
        """Background loop for processing embedding queue."""
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
        """Lifespan context for embedding polling task.

        Args:
            app: FastAPI application.

        Yields:
            None.
        """
        self.polling_task = asyncio.create_task(self.polling_loop())
        yield
        if self.polling_task:
            self.polling_task.cancel()
            await asyncio.gather(self.polling_task, return_exceptions=True)

    def register_routes(self, router: APIRouter):
        """Register embedding-related routes.

        Args:
            router: APIRouter to register routes with.
        """
        @router.get("/embeddings/", tags=["Embeddings"])
        async def stat_embeddings(session: AsyncSession = Depends(get_db)):
            """Get embedding statistics.

            Args:
                session: Database session.

            Returns:
                Dictionary with finished and queued counts.
            """
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

        @router.post("/embeddings", tags=["Embeddings"])
        async def create_embeddings(
            data: dict = Body(...),
            session: AsyncSession = Depends(get_db),
        ):
            """Create embeddings for input text(s) using configured model.
            
            This endpoint follows OpenAI's embeddings API style:
            https://platform.openai.com/docs/api-reference/embeddings
            
            Args:
                data: Request body with:
                    - model: Model name to use for embeddings (optional, uses config model)
                    - input: Text or list of texts to embed
                    - encoding_format: Output format ("float" or "base64")
                session: Database session.

            Returns:
                Dictionary with embeddings in OpenAI format:
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": [float, float, ...],
                            "index": 0
                        }
                    ],
                    "model": "model-name",
                    "usage": {"prompt_tokens": int, "total_tokens": int}
                }
            """
            # Extract input texts (can be string or list of strings)
            input_data = data.get("input", [])
            if isinstance(input_data, str):
                input_texts = [input_data]
            elif isinstance(input_data, list):
                input_texts = input_data
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid input: must be a string or list of strings"
                )
            
            if not input_texts:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid input: empty list"
                )
            
            # Get model name (use config if not provided)
            model_name = data.get("model", config.embedding_model)
            
            # Compute embeddings using the configured model
            embeddings = []
            for text in input_texts:
                if not isinstance(text, str):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid input: expected string, got {type(text).__name__}"
                    )
                # Use the existing get_embedding method from TextEmbeddings
                embedding = await TextEmbeddings.get_embedding(text)
                embeddings.append(embedding)
            
            # Determine encoding format (default to float)
            encoding_format = data.get("encoding_format", "float")
            
            # Build response in OpenAI format
            data_entries = []
            for i, embedding in enumerate(embeddings):
                # Convert to base64 if requested
                if encoding_format == "base64":
                    # Pack floats as binary and encode as base64
                    packed = struct.pack(f'{len(embedding)}f', *embedding)
                    embedding_b64 = base64.b64encode(packed).decode('utf-8')
                    data_entries.append({
                        "object": "embedding",
                        "embedding": embedding_b64,
                        "index": i
                    })
                else:
                    # Default: return as float list
                    data_entries.append({
                        "object": "embedding",
                        "embedding": embedding,
                        "index": i
                    })
            
            # Calculate token usage (approximate: 1 token ~ 4 characters)
            total_tokens = sum(len(text) // 4 for text in input_texts)
            
            return {
                "object": "list",
                "data": data_entries,
                "model": model_name,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens
                }
            }

        @router.post("/embeddings/{resource_id}", status_code=201, tags=["Embeddings"])
        async def update_single_embedding(
            resource_id: str, session: AsyncSession = Depends(get_db)
        ):
            """Update embedding for a single paragraph.

            Args:
                resource_id: Paragraph ID.
                session: Database session.

            Returns:
                Dictionary with paragraph ID.

            Raises:
                HTTPException: If paragraph not found.
            """
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


class APIKeyManager:
    """Handles API key management operations."""

    def register_routes(self, router: APIRouter):
        """Register API key management routes.

        Args:
            router: APIRouter to register routes with.
        """
        @router.get("/apikeys", tags=["API Keys"])
        async def list_api_keys(
            session: AsyncSession = Depends(get_db),
            username: str = Depends(get_current_username),
        ):
            """List all API keys for the current user.
            
            Args:
                session: Database session.
                username: Current username.
                
            Returns:
                List of API keys with metadata.
            """
            # Get user ID from username
            result = await session.execute(
                select(UserInfo).filter(UserInfo.username == username)
            )
            user = result.scalar_one_or_none()
            if not user:
                raise HTTPException(404, detail="User not found")
            
            # Query API keys for this user
            result = await session.execute(
                select(APIKey).filter(APIKey.user_id == user.id)
            )
            api_keys = result.scalars().all()
            
            return {
                "object": "list",
                "data": [
                    {
                        "id": str(api_key.id),
                        "object": "api_key",
                        "name": api_key.name,
                        "is_active": api_key.is_active,
                        "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                        "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                        "created_at": api_key.created_at.isoformat(),
                    }
                    for api_key in api_keys
                ],
            }

        @router.post("/apikeys", status_code=201, tags=["API Keys"])
        async def create_api_key(
            data: dict = Body(...),
            session: AsyncSession = Depends(get_db),
            username: str = Depends(get_current_username),
        ):
            """Create a new API key for the current user.
            
            Args:
                data: Request body with optional 'name' field.
                session: Database session.
                username: Current username.
                
            Returns:
                Dictionary with the plain API key (only shown once!) and metadata.
            """
                        
            # Get user ID from username
            result = await session.execute(
                select(UserInfo).filter(UserInfo.username == username)
            )
            user = result.scalar_one_or_none()
            if not user:
                raise HTTPException(404, detail="User not found")
            
            # Generate new API key
            plain_key = generate_api_key()
            key_hash = hash_api_key(plain_key)
            
            # Create API key record
            api_key = APIKey(
                user_id=user.id,
                key_hash=key_hash,
                name=data.get("name"),
                is_active=True,
            )
            session.add(api_key)
            await session.commit()
            
            return {
                "id": str(api_key.id),
                "object": "api_key",
                "name": api_key.name,
                "is_active": api_key.is_active,
                "created_at": api_key.created_at.isoformat(),
                # Only show the plain key once!
                "plain_key": plain_key,
            }

        @router.delete("/apikeys/{resource_id}", tags=["API Keys"])
        async def delete_api_key(
            resource_id: str,
            session: AsyncSession = Depends(get_db),
            username: str = Depends(get_current_username),
        ):
            """Delete an API key for the current user.
            
            Args:
                resource_id: API key ID.
                session: Database session.
                username: Current username.
                
            Returns:
                Deletion confirmation.
                
            Raises:
                HTTPException: If API key not found or doesn't belong to user.
            """ 
            # Get user ID from username
            result = await session.execute(
                select(UserInfo).filter(UserInfo.username == username)
            )
            user = result.scalar_one_or_none()
            if not user:
                raise HTTPException(404, detail="User not found")
            
            # Get API key
            api_key = await session.get(APIKey, resource_id)
            if not api_key:
                raise HTTPException(404, detail="API key not found")
            
            # Verify ownership
            if api_key.user_id != user.id:
                raise HTTPException(403, detail="API key does not belong to user")
            
            # Soft delete by setting is_active to False
            await session.delete(api_key)
            await session.commit()
            
            return {"message": "API key deleted"}

        @router.post("/apikeys/{resource_id}/revoke", tags=["API Keys"])
        async def revoke_api_key(
            resource_id: str,
            session: AsyncSession = Depends(get_db),
            username: str = Depends(get_current_username),
        ):
            """Revoke an API key (soft delete).
            
            Args:
                resource_id: API key ID.
                session: Database session.
                username: Current username.
                
            Returns:
                Revocation confirmation.
                
            Raises:
                HTTPException: If API key not found or doesn't belong to user.
            """
            # Get user ID from username
            result = await session.execute(
                select(UserInfo).filter(UserInfo.username == username)
            )
            user = result.scalar_one_or_none()
            if not user:
                raise HTTPException(404, detail="User not found")
            
            # Get API key
            api_key = await session.get(APIKey, resource_id)
            if not api_key:
                raise HTTPException(404, detail="API key not found")
            
            # Verify ownership
            if api_key.user_id != user.id:
                raise HTTPException(403, detail="API key does not belong to user")
            
            # Soft delete by setting is_active to False
            api_key.is_active = False
            await session.commit()
            
            return {"message": "API key revoked"}

        @router.post("/apikeys/{resource_id}/activate", tags=["API Keys"])
        async def activate_api_key(
            resource_id: str,
            session: AsyncSession = Depends(get_db),
            username: str = Depends(get_current_username),
        ):
            """Activate a revoked API key.
            
            Args:
                resource_id: API key ID.
                session: Database session.
                username: Current username.
                
            Returns:
                Activation confirmation.
                
            Raises:
                HTTPException: If API key not found or doesn't belong to user.
            """
            # Get user ID from username
            result = await session.execute(
                select(UserInfo).filter(UserInfo.username == username)
            )
            user = result.scalar_one_or_none()
            if not user:
                raise HTTPException(404, detail="User not found")
            
            # Get API key
            api_key = await session.get(APIKey, resource_id)
            if not api_key:
                raise HTTPException(404, detail="API key not found")
            
            # Verify ownership
            if api_key.user_id != user.id:
                raise HTTPException(403, detail="API key does not belong to user")
            
            # Activate the key
            api_key.is_active = True
            await session.commit()
            
            return {"message": "API key activated"}


class ContentManager(ResourceRegistry):
    """Handles dataset and paragraph logic."""

    def __init__(self):
        """Initialize content manager."""
        super().__init__(Paragraph, '', [])

    def register_routes(self, router: APIRouter):
        """Register content-related routes.

        Args:
            router: APIRouter to register routes with.
        """
        @router.get("/datasets", tags=["Datasets"])
        @router.get("/datasets/{resource_id}", tags=["Datasets"])
        async def get_datasets(resource_id: Optional[str] = None):
            """Get datasets list or single dataset.

            Args:
                resource_id: Optional dataset ID.

            Returns:
                Dataset hierarchy or single dataset.
            """
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
            """Rename a dataset.

            Args:
                resource_id: Dataset ID.
                name: New dataset name.

            Returns:
                New dataset ID.

            Raises:
                HTTPException: If dataset not found.
            """
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
            """Delete a dataset.

            Args:
                resource_id: Dataset ID.
                session: Database session.

            Returns:
                Deletion confirmation.

            Raises:
                HTTPException: If dataset not found.
            """
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
            """Create a new paragraph.

            Args:
                data: Paragraph data.
                session: Database session.

            Returns:
                Created paragraph.
            """
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
            """Search paragraphs with filters.

            Args:
                filters: QueryFilters with search parameters.
                session: Database session.
                username: Current username.

            Returns:
                Search results with pagination.
            """
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
            """Get filter values for a column.

            Args:
                column: Column name.
                filters: QueryFilters.
                session: Database session.

            Returns:
                List of values with counts.
            """
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
    """Handles file storage operations."""

    def register_routes(self, router: APIRouter):
        """Register file storage routes.

        Args:
            router: APIRouter to register routes with.
        """
        @router.get("/files/{file_path:path}", tags=["Files"])
        def get_file(
            file_path: str = "",
            search: str = "",
            metadata: bool = False,
            page: Optional[int] = None,
            format: Optional[str] = None,
        ):
            """Get file or list directory contents.

            Args:
                file_path: File path.
                search: Search pattern.
                metadata: Include metadata.
                page: Page number for PDF.
                format: Output format.

            Returns:
                File content or directory listing.

            Raises:
                HTTPException: If file not found or access denied.
            """
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
            """Upload a file or create directory.

            Args:
                file_path: Target path.
                is_directory: Create directory if True.
                name: Directory name.
                file: Uploaded file.

            Returns:
                File or directory info.

            Raises:
                HTTPException: If upload fails.
            """
            if is_directory and name:
                return storage.mkdir(file_path, name)
            if not file:
                raise HTTPException(400, detail="No file")
            return storage.save(file.file, file_path)


# --- Router assembly and lifecycle binding ---

emb_manager = EmbeddingManager()

# Combined lifespan that runs both embedding polling and worker
@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    """Combined lifespan for embedding polling and worker management."""
    # Start embedding polling task
    emb_manager.polling_task = asyncio.create_task(emb_manager.polling_loop())
    
    # Start worker
    await worker_manager.start_worker()
    
    yield
    
    # Stop worker
    worker_manager.stop_worker()
    
    # Stop embedding polling task
    if emb_manager.polling_task:
        emb_manager.polling_task.cancel()
        await asyncio.gather(emb_manager.polling_task, return_exceptions=True)

app.router.lifespan_context = combined_lifespan
api_key_manager = APIKeyManager()

# Register common CRUD resources
ResourceRegistry(UserInfo, "/users", ["Users"]).register(
    router, dependencies=[Depends(get_current_admin)]
)
ResourceRegistry(History, "/histories", ["Histories"]).register(router)
TaskResource().register(router)

# Register business logic modules
ContentManager().register_routes(router)
emb_manager.register_routes(router)
api_key_manager.register_routes(router)
StorageManager().register_routes(router)


# Auxiliary functions
@router.post("/translator", tags=["Translator"])
async def translator(params: dict = Body(...)):
    """Translate text using external API.

    Args:
        params: Dictionary with 'lang', 'text', and 'zhipu_api_key'.

    Returns:
        Translated text.
    """
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


# 4. External manager integration
router.include_router(plugins.get_router())
router.include_router(maintenance_manager.get_router())
app.include_router(router)

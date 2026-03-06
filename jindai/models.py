"""Database models for Jindai application.

This module provides SQLAlchemy ORM models for:
- Dataset: Data organization unit
- UserInfo: User authentication and authorization
- History: User query history
- Paragraph: Text content storage
- Terms: Vocabulary/keyword storage
- EmbeddingPendingQueue: Embedding processing queue
- FileMetadata: File metadata tracking
- TaskDBO: Task definition storage
- TextEmbeddings: Vector embeddings for semantic search
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import wraps
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import redis
import regex as re
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, Field
from sqlalchemy import (BigInteger, Boolean, Column, DateTime, ForeignKey, Index, Integer,
                        PrimaryKeyConstraint, String, Text, UniqueConstraint,
                        asc, desc, exists, or_, select, text, update)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID, insert
from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.orm import (Mapped, declarative_base, mapped_column,
                            relationship, validates)
from sqlalchemy.sql import func

from .config import config
from .helpers import AutoUnloadSentenceTransformer, jieba

engine = create_async_engine(config.database)
session_factory = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Get an async database session with automatic commit/rollback.

    Yields:
        AsyncSession: Database session.

    Raises:
        Exception: Any exception during session execution.
    """
    session = session_factory()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise e
    finally:
        await session.close()


MBase = declarative_base()


class Base(MBase):
    """Base class for all ORM models.

    Provides common functionality:
    - UUID primary keys
    - created_at/modified_at timestamps
    - as_dict() method for serialization
    - parse_sort_string() for query sorting
    """

    # Add default created_at field for all tables (ignored if table already has it)
    __abstract__ = True
    __table_args__ = {"schema": "public"}

    # Common field configuration: Primary key ID uses UUID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="Primary Key ID"
    )

    def as_dict(self) -> dict:
        """Convert model instance to dictionary.

        Returns:
            Dictionary with column names as keys and values.
        """
        data = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            # Handle UUID type
            if value is None:
                data[column.name] = None
            elif isinstance(value, uuid.UUID):
                data[column.name] = str(value)
            # Handle datetime type
            elif isinstance(value, datetime):
                data[column.name] = value.isoformat()
            # Handle nested Base models
            elif isinstance(value, Base):
                data[column.name] = value.as_dict()
            elif isinstance(value, (int, str, float, list)):
                data[column.name] = value
            elif isinstance(column.type, JSONB):
                data[column.name] = value
            else:
                data[column.name] = f"<{value} of {column.type}>"
        return data

    @classmethod
    def parse_sort_string(cls, sort_by: Any) -> List[Any]:
        """Parse sort string into SQLAlchemy order_by expressions.

        Args:
            sort_by: Sort specification as string or list.

        Returns:
            List of SQLAlchemy order_by expressions.
        """
        if isinstance(sort_by, str):
            sort_by = sort_by.split(",")

        sorts = []
        for sort_part in sort_by:
            if not isinstance(sort_part, str):
                sorts.append(sort_part)
                continue

            sort_part = sort_part.strip()
            if not sort_part:
                continue

            if sort_part.startswith("-"):
                col_name, sort_func = sort_part[1:], desc
            else:
                col_name, sort_func = sort_part, asc

            col = getattr(cls, col_name, None)
            if col is not None:
                sorts.append(sort_func(col))

        return sorts


# Base model class
class Dataset(Base):
    """Dataset model for organizing paragraphs.

    Datasets are used to group related paragraphs together.
    Supports hierarchical organization using "--" separator in names.
    """

    __tablename__ = "dataset"
    __table_args__ = (
        UniqueConstraint("name", name="dataset_name_key"),  # Unique constraint
        {
            "comment": "Dataset information table",
        },  # Table comment + schema specification
    )

    order_weight: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="Sort weight"
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="Dataset name")
    tags: Mapped[List[str] | None] = mapped_column(ARRAY(Text), comment="Tag list")

    # Relationship: One dataset has many paragraphs
    paragraphs: Mapped[List["Paragraph"]] = relationship(
        "Paragraph", back_populates="dataset_obj", cascade="all, delete-orphan"
    )

    @staticmethod
    async def get(name: str, auto_create: bool = True) -> "Dataset":
        """Get or create a dataset by name.

        Args:
            name: Dataset name or UUID.
            auto_create: Create dataset if not found.

        Returns:
            Dataset instance.
        """
        async with get_db_session() as session:
            if is_uuid_literal(name):
                ds = await session.get(Dataset, name)
            else:
                ds = await session.execute(select(Dataset).filter(Dataset.name == name))
                ds = ds.scalar_one_or_none()
            if ds is None and auto_create:
                ds = Dataset(name=name)
                session.add(ds)
                await session.flush()

        return ds

    async def rename_dataset(self, new_name: str) -> Optional[uuid.UUID]:
        """Rename dataset and update all related paragraphs.

        Args:
            new_name: New dataset name.

        Returns:
            Dataset ID of the new dataset if renamed, or current ID if unchanged.
        """
        if self.name == new_name:
            return None
        ds = await Dataset.get(new_name)
        if ds:
            stmt = (
                update(Paragraph)
                .where(Paragraph.dataset == self.id)
                .values({"dataset": ds.id})
            )
            async with get_db_session() as session:
                await session.execute(stmt)
                await session.delete(self)
        else:
            self.name = new_name
        return ds.id if ds else self.id

    @staticmethod
    async def get_hierarchy() -> list:
        """Get hierarchical structure of datasets.

        Returns:
            Nested list representing dataset hierarchy with children.
        """
        def _dataset_sort_key(ds: Dataset):
            return len(ds.name.split("--")), ds.order_weight, ds.name

        async with get_db_session() as session:
            datasets = await session.execute(select(Dataset))

        sorted_datasets = sorted(datasets.scalars().all(), key=_dataset_sort_key)
        hierarchy = []
        for dataset in sorted_datasets:
            current_level = hierarchy
            parts = dataset.name.split("--")
            for parti, part in enumerate(parts):
                found = False
                for item in current_level:
                    if item["title"] == part:
                        current_level = item.setdefault("children", [])
                        found = True
                        break
                if not found:
                    new_item = {
                        "title": part,
                        "children": [],
                        "order_weight": dataset.order_weight,
                        "record_id": (
                            str(dataset.id) if parti == len(parts) - 1 else None
                        ),
                        "value": "--".join(parts[: parti + 1]),
                    }
                    current_level.append(new_item)
                    current_level = new_item["children"]
        return hierarchy


class UserInfo(Base):
    """User information model for authentication and authorization.

    Stores user credentials, roles, and dataset permissions.
    """

    __tablename__ = "user_info"
    __table_args__ = (
        UniqueConstraint("username", name="user_info_username_key"),  # Username unique
        {
            "comment": "User table",
        },
    )

    username: Mapped[str] = mapped_column(String(64), nullable=False, comment="Username")
    roles: Mapped[List[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False, comment="User role list"
    )
    datasets: Mapped[List[UUID] | None] = mapped_column(
        ARRAY(Text), default=list, comment="List of accessible dataset IDs"
    )

    # Relationships
    histories: Mapped[List["History"]] = relationship(
        "History", back_populates="user", cascade="all, delete-orphan"
    )
    tasks: Mapped[List["TaskDBO"]] = relationship(
        "TaskDBO", back_populates="user", cascade="all, delete-orphan"
    )


class History(Base):
    """User query history model.

    Stores user search queries for audit and analytics purposes.
    """

    __tablename__ = "history"
    __table_args__ = {
        "comment": "User operation history table",
    }

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user_info.id", ondelete="CASCADE"),
        nullable=False,
        comment="Related user ID",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.current_timestamp(), nullable=False, comment="Creation time"
    )
    queries: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB, default=dict, nullable=False, comment="Query records (JSON)"
    )

    # Relationship to user
    user: Mapped["UserInfo"] = relationship("UserInfo", back_populates="histories")


class QueryFilters(BaseModel):
    """Query filters for paragraph search.

    Provides a flexible filtering interface for searching paragraphs
    with support for:
    - Text search
    - ID/dataset/source filtering
    - Author/language filtering
    - Vector embedding search
    - Grouping and sorting
    - Pagination
    """

    # Search keyword, default "*"
    q: str = "*"

    # Basic filtering
    ids: Optional[List[str|uuid.UUID]] = None
    datasets: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    sourcePage: Optional[int] = None
    outline: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    author: Optional[str] = None
    lang: Optional[List[str]] = None

    # Control logic
    embeddings: Optional[bool] = None  # False excludes data with vectors
    groupBy: Optional[str] = None  # Corresponds to Paragraph field name

    # Sorting and pagination
    sort: Optional[Union[str, List[str]]] = ""
    offset: int = 0
    limit: Optional[int] = None


class Paragraph(Base):
    """Paragraph model for storing text content.

    Represents individual text segments with metadata including
    source information, author, date, and keywords.
    """

    __tablename__ = "paragraph"
    __table_args__ = (
        # Index definitions (same as original table)
        PrimaryKeyConstraint("id", "dataset", name="paragraph_part_pk"),
        Index("fki_dataset", "dataset"),
        Index("idx_paragraph_author", "author"),
        Index("idx_paragraph_keywords", "keywords", postgresql_using="gin"),
        Index("idx_paragraph_outline", "outline"),
        Index("idx_paragraph_pagenum", "pagenum"),
        Index("idx_paragraph_pdate", "pdate"),
        Index("idx_paragraph_source", "source_url", "source_page", "pagenum"),
        {
            "comment": "Paragraph table",
        },
    )

    source_url: Mapped[str | None] = mapped_column(String(1024), comment="Source URL")
    source_page: Mapped[int | None] = mapped_column(
        Integer, comment="Source file page number (e.g., PDF page)"
    )
    author: Mapped[str | None] = mapped_column(String(128), comment="Author")
    pdate: Mapped[datetime | None] = mapped_column(DateTime, comment="Publication date")
    outline: Mapped[str] = mapped_column(Text, comment="Outline/summary")
    content: Mapped[str] = mapped_column(Text, comment="Paragraph content")
    pagenum: Mapped[str | None] = mapped_column(Text, comment="Page number")
    lang: Mapped[str] = mapped_column(String(16), default="zh", comment="Language")
    extdata: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Extended metadata (custom JSON field)",
    )
    dataset: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset.id"),
        nullable=False,
        primary_key=True,
        comment="Related dataset ID",
    )
    keywords: Mapped[List[str]] = mapped_column(
        ARRAY(Text), comment="Keyword list"
    )

    dataset_obj: Mapped["Dataset"] = relationship(
        "Dataset", back_populates="paragraphs", lazy="joined"
    )
    text_embeddings: Mapped[List["TextEmbeddings"]] = relationship(
        "TextEmbeddings", back_populates="paragraph", cascade="all, delete-orphan"
    )

    @validates("content")
    def normalize_content(self, key: str, val: str) -> str:
        """Normalize content by removing null bytes.

        Args:
            key: Field name.
            val: Content value.

        Returns:
            Normalized content.
        """
        return val.replace("\x00", "")

    @validates("keywords")
    def normalize_keywords(self, key: str, val: list) -> list:
        """Normalize keywords by stripping whitespace and lowercasing.

        Args:
            key: Field name.
            val: List of keywords.

        Returns:
            Normalized list of unique keywords.
        """
        if val:
            val = list({v.strip().lower() for v in val if v.strip()})
        return val

    async def set_dataset_name(self, new_name: str) -> None:
        """Set dataset by name.

        Args:
            new_name: New dataset name.
        """
        self.dataset = (await Dataset.get(new_name)).id

    @staticmethod
    def from_dict(data: dict, ignored_fields: list = None, **kwargs) -> "Paragraph":
        """Create Paragraph instance from dictionary.

        Args:
            data: Dictionary with field values.
            ignored_fields: Fields to ignore (default: ["id", "dataset_name"]).
            **kwargs: Additional field values.

        Returns:
            Paragraph instance.
        """
        if ignored_fields is None:
            ignored_fields = ["id", "dataset_name"]
        p = Paragraph()
        data.update(kwargs)
        for k, v in data.items():
            if k in ignored_fields:
                continue
            if hasattr(p, k):
                setattr(p, k, v)
            else:
                if p.extdata is None:
                    p.extdata = {}
                p.extdata[k] = v
        return p

    def __getitem__(self, key: str) -> Any:
        """Get attribute or extdata value.

        Args:
            key: Field name.

        Returns:
            Field value or extdata value.
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            if self.extdata is None:
                self.extdata = {}
            return self.extdata.get(key)

    def __setitem__(self, key: str, val: Any) -> None:
        """Set attribute or extdata value.

        Args:
            key: Field name.
            val: Value to set.
        """
        if hasattr(self, key):
            setattr(self, key, val)
        else:
            if self.extdata is None:
                self.extdata = {}
            self.extdata[key] = val

    def __delitem__(self, key: str) -> None:
        """Delete extdata key.

        Args:
            key: Key to delete from extdata.
        """
        if key in self.extdata:
            del self.extdata[key]

    def as_dict(self) -> dict:
        """Convert to dictionary with dataset name.

        Returns:
            Dictionary with all fields including dataset_name.
        """
        data = super().as_dict()
        data["dataset_name"] = self.dataset_obj.name if self.dataset_obj else None
        return data

    @staticmethod
    async def build_query(query_filters: QueryFilters):
        """Build SQLAlchemy query from QueryFilters.

        Args:
            query_filters: QueryFilters instance with search parameters.

        Returns:
            SQLAlchemy select query with filters applied.
        """
        query = select(Paragraph)
        filters = []
        query_embedding = None
        search = query_filters.q

        # Primary Key / ID Filters
        if ids := query_filters.ids:
            filters.append(Paragraph.id.in_(ids))

        # Dataset Filters (Optimized logic)
        if datasets := query_filters.datasets:
            dataset_filters = [Dataset.name.in_(datasets)]
            for dataset_prefix in datasets:
                dataset_filters.append(Dataset.name.ilike(f"{dataset_prefix}--%"))

            datasets = select(Dataset.id).where(or_(*dataset_filters))
            async with get_db_session() as session:
                dataset_ids = await session.execute(datasets)
            filters.append(Paragraph.dataset.in_(dataset_ids.scalars()))

        # Source URL Filters
        if sources := query_filters.sources:
            source_filters = [Paragraph.source_url.in_(sources)]
            for source in sources:
                source_filters.append(Paragraph.source_url.ilike(f"{source}%"))
            filters.append(or_(*source_filters))

        if source_page := query_filters.sourcePage:
            filters.append(Paragraph.source_page == source_page)

        # Outline Filter
        if outline := query_filters.outline:
            filters.append(Paragraph.outline.in_(outline))

        if lang := query_filters.lang:
            filters.append(Paragraph.lang.in_(lang))

        # Author Filter
        if authors := query_filters.authors:
            filters.append(Paragraph.author.in_(authors))
        if author := query_filters.author:
            filters.append(Paragraph.author == author)

        # Embedding Existence Filter
        if query_filters.embeddings is False:
            filters.append(~exists().where(TextEmbeddings.id == Paragraph.id))

        # Search Logic
        if search.startswith("?"):
            filters.append(text(search[1:]))
        elif search.startswith("*"):
            search_term = search.strip("*")
            if search_term:
                filters.append(Paragraph.content.ilike(f"%{search_term}%"))
        elif search.startswith(":") or query_filters.embeddings:
            query_embedding = await TextEmbeddings.get_embedding(search.strip(":"))
        else:
            words = [_.strip().lower() for _ in jieba.cut_query(search) if _.strip()]
            for word in words:
                candidates = [word.strip("^%")]
                if word.startswith("^"):
                    candidates.extend(await Terms.starting_with(word.strip("^")))
                filters.append(
                    or_(*(Paragraph.keywords.contains([c]) for c in candidates))
                )

        query = query.filter(*filters)

        # Vector Search Join
        if query_embedding is not None:
            query = (
                query.join(
                    TextEmbeddings,
                    (Paragraph.id == TextEmbeddings.id)
                    & (Paragraph.dataset == TextEmbeddings.dataset),  # Chunking requires
                )
                .distinct(
                    Paragraph.id,
                    Paragraph.dataset,
                )
                .order_by(
                    Paragraph.id,
                    Paragraph.dataset,
                    TextEmbeddings.embedding.cosine_distance(query_embedding),
                )
                .add_columns(TextEmbeddings.embedding)
                .subquery()
            )
            query = (
                select(query.c.id, query.c.dataset)
                .order_by(query.c.embedding.cosine_distance(query_embedding))
                .subquery()
            )
            query = select(Paragraph).join(
                query,
                (Paragraph.dataset == query.c.dataset) & (Paragraph.id == query.c.id),
            )

        if group_field_name := query_filters.groupBy:
            group_column = (
                getattr(Paragraph, group_field_name, None) if group_field_name else None
            )
            if group_column is not None:
                query = query.distinct(group_column).order_by(
                    group_column
                )  # DISTINCT ON requires order_by to start with group column

        # Sorting
        if sort_by := query_filters.sort:
            sorts = Paragraph.parse_sort_string(sort_by)
            query = query.order_by(*sorts)

        # Pagination
        if offset := query_filters.offset:
            query = query.offset(offset)
        if limit := query_filters.limit:
            query = query.limit(limit)

        return query


class Terms(MBase):  # terms has no `id`, and cannot as_dict()
    """Terms/vocabulary table.

    Stores unique terms extracted from paragraphs for search indexing.
    """

    __tablename__ = "terms"
    __table_args__ = {
        "comment": "Vocabulary table",
    }

    term: Mapped[str] = mapped_column(
        String, nullable=False, comment="Term", primary_key=True
    )

    @staticmethod
    async def starting_with(prefix: str) -> List[str]:
        """Get terms starting with prefix.

        Args:
            prefix: Term prefix to match.

        Returns:
            List of matching terms.
        """
        if not prefix:
            return []
        async with get_db_session() as session:
            q = await session.execute(
                select(Terms)
                .filter(Terms.term.startswith(prefix))
                .with_only_columns(Terms.term)
            )
            return list(q.scalars())

    @staticmethod
    async def store(words: list[str]) -> list[str]:
        """Store words in terms table.

        Args:
            words: List of words to store.

        Returns:
            List of stored words.
        """
        # Clean the input to ensure uniqueness in the batch
        data = [{"term": w} for w in words if w and len(w) > 1]
        if not data:
            return []
        async with get_db_session() as session:
            async with session.begin():  # Start a transaction

                # Execute a bulk insert
                # Note: This assumes 'term' is a unique column.
                # If not, use standard session.execute(insert(Terms), data)
                stmt = insert(Terms).values(data)

                # If you have a UNIQUE constraint on 'term', use this to skip duplicates:
                stmt = stmt.on_conflict_do_nothing(index_elements=["term"])

                await session.execute(stmt)
                # No need for manual commit if using 'async with session.begin()'

        return words


class EmbeddingPendingQueue(Base):
    """Queue for pending text embeddings.

    Tracks paragraphs that need embedding processing.
    """

    __tablename__ = 'embedding_pending_queue'

    # Define composite primary key
    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True)
    dataset: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True)

    # Record creation time for debugging delays or ordered processing
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FileMetadata(Base):
    """File metadata model for tracking stored files.

    Provides content-addressable storage tracking using SHA-1 hashes.
    """

    __tablename__ = "file_metadata"

    # Original filename (with extension)
    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        primary_key=True,
        doc="Original filename as uploaded / stored"
    )

    # Primary key – using SHA-1 hash as natural/business key (common in content-addressable systems)
    sha1: Mapped[str] = mapped_column(
        String(40),          # SHA-1 is exactly 40 hex characters
        index=True,
        doc="SHA-1 hash of the file content (hex digest)"
    )

    # File extension (normalized, lowercase, without dot)
    extension: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        doc="File extension in lowercase without leading dot (pdf, jpg, docx, ...)"
    )

    # File size in bytes
    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        doc="Size of the file in bytes"
    )

    # For PDFs mostly – number of pages
    page_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of pages (mainly for PDFs & images with multiple frames)"
    )

    # Flexible metadata stored in JSONB (very powerful with PostgreSQL)
    extdata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        doc="""Flexible JSON metadata – examples:
        {
          "title": "...",
          "author": "...",
          "keywords": ["finance", "2024"],
          "ocr_text": "...",
          "has_form_fields": true,
          "color_space": "CMYK",
          "embedded_fonts": 12,
          "is_scanned": true,
          "scan_dpi": 300
        }"""
    )

    # Audit timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True
    )

    modified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        index=True
    )

    # Optional: if you want automatic "version" on update (incrementing counter)
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="1",
        doc="Incremented on each update (optimistic locking possible)",
        onupdate=text("version + 1")
    )

    def __repr__(self):
        return f"<File {self.filename}  sha1:{self.sha1[:12]}…  {self.page_count or '?'} pages>"

    @property
    def is_pdf(self) -> bool:
        """Check if file is a PDF.

        Returns:
            True if file extension is 'pdf'.
        """
        return self.extension == "pdf"


class TaskDBO(Base):
    """Task definition model.

    Stores task configurations including pipeline stages and parameters.
    """

    __tablename__ = "task_dbo"
    __table_args__ = {
        "comment": "Task table",
    }

    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="Task name")
    pipeline: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False, comment="Task pipeline (JSON)"
    )
    resume_next: Mapped[bool] = mapped_column(
        Boolean, default=False, comment="Whether to continue on error"
    )
    last_run: Mapped[datetime | None] = mapped_column(DateTime, comment="Last run time")
    concurrent: Mapped[int] = mapped_column(Integer, default=3, comment="Concurrency level")
    shortcut_map: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, default=dict, nullable=False, comment="Shortcut mapping (JSON)"
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user_info.id", ondelete="CASCADE"),
        nullable=False,
        comment="Related user ID",
    )
    shared: Mapped[bool] = mapped_column(Boolean, default=False, comment="Whether shared")

    # Relationship to user
    user: Mapped["UserInfo"] = relationship("UserInfo", back_populates="tasks")

    @staticmethod
    async def get(id_or_name: str) -> Optional["TaskDBO"]:
        """Get TaskDBO by ID or name.

        Args:
            id_or_name: UUID string or name to search for.

        Returns:
            TaskDBO instance or None.
        """
        async with get_db_session() as session:
            if is_uuid_literal(id_or_name):
                return await session.get(TaskDBO, uuid.UUID(id_or_name))
            else:
                result = await session.execute(
                    select(TaskDBO).filter(TaskDBO.name.contains(id_or_name))
                )
                return result.scalar_one_or_none()


redis_client = redis.Redis.from_url(config.redis + "/2")


# Cache core configuration (can be flexibly modified according to business needs)
CACHE_EXPIRE_SECONDS = 60 * 10  # Cache default expiration: 5 minutes
CACHE_KEY_PREFIX = "query_cache:"  # Cache key prefix for Redis


def redis_auto_renew_cache(cache_key=None):
    """Decorator factory for automatic Redis cache renewal.

    Provides automatic caching with access-based expiration.
    Key features:
    - Automatically refreshes cache expiration on each access
    - Hot data never expires, cold data auto-expires
    - Supports custom cache key generation

    Args:
        cache_key: Optional custom cache key function receiving *args and **kwargs.

    Returns:
        Decorator function for the decorated function.
    """

    def decorator(func):
        func.cache_key_method = cache_key

        @wraps(func)  # Preserve original function attributes (name, docstring, etc.)
        def wrapper(*args, **kwargs):
            if func.cache_key_method is not None:
                cache_key_str = f"{func.cache_key_method(*args, **kwargs) or ''}"
            else:
                args_str = "_".join(map(str, args))
                kwargs_str = "_".join([f"{k}_{v}" for k, v in sorted(kwargs.items())])
                cache_key_str = f"{args_str}_{kwargs_str}"

            if cache_key_str:
                cache_key_full = f"{CACHE_KEY_PREFIX}{func.__name__}_{cache_key_str}"
                cached_data = redis_client.get(cache_key_full)
                if cached_data is not None:
                    redis_client.expire(cache_key_full, CACHE_EXPIRE_SECONDS)
                    return json.loads(cached_data)

            result = func(*args, **kwargs)

            if result is not None and cache_key_str:
                # Write to cache + set expiration,实现「自动过期」
                redis_client.setex(
                    name=cache_key_full,
                    time=CACHE_EXPIRE_SECONDS,
                    value=json.dumps(result),  # Unified to string, supports any JSON-serializable result
                )

            return result

        return wrapper

    return decorator


class TextEmbeddings(Base):
    """Text embeddings model for semantic search.

    Stores vector embeddings of text chunks for similarity search.
    """

    embedding_model = AutoUnloadSentenceTransformer(config.embedding_model)

    __tablename__ = "text_embeddings"
    __table_args__ = (
        Index(
            "idx_embedding_cosine",
            "embedding",
            postgresql_using="vchordrq",
            # postgresql_with={"m": 16},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        {
            "comment": "Text embeddings table",
        },
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("paragraph.id", ondelete="CASCADE"),
        primary_key=True,
        comment="Paragraph ID",
    )

    dataset: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset.id", ondelete="CASCADE"),
        primary_key=True,
        comment="Dataset ID",
    )

    chunk_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        comment="Chunk ID",
    )

    embedding: Mapped[Vector] = mapped_column(
        Vector(config.embedding_dims), nullable=False, comment="Text embedding vector"
    )

    paragraph: Mapped["Paragraph"] = relationship(
        "Paragraph", back_populates="text_embeddings"
    )

    @staticmethod
    @redis_auto_renew_cache()
    def get_embedding_sync(text: str) -> list:
        """Get embedding for text synchronously.

        Args:
            text: Input text to embed.

        Returns:
            Embedding as list of floats.
        """
        embedding = TextEmbeddings.embedding_model.encode(
            text.strip(),
            convert_to_numpy=True,  # Return numpy array for easier processing
            normalize_embeddings=True,  # Normalize vectors for better retrieval
        )

        return embedding.tolist()

    @staticmethod
    async def get_embedding(text: str) -> list:
        """Get embedding for text asynchronously.

        Args:
            text: Input text to embed (supports 100+ languages).

        Returns:
            Embedding as list of floats (dimension: 384).
        """
        return await asyncio.to_thread(TextEmbeddings.get_embedding_sync, text)

    @staticmethod
    def get_chunks(text: str, chunk_length: int, overlap: int) -> list:
        """Split long text into chunks with overlapping windows.

        Args:
            text: Long text to split (paragraph/page text).
            chunk_length: Character length per chunk.
            overlap: Overlapping character length between chunks.

        Returns:
            List of text chunks.
        """
        # Basic validation
        if not text.strip():
            return []
        if chunk_length <= 0 or overlap < 0 or overlap >= chunk_length:
            return []

        text = text.strip()
        chunks = []
        start_idx = 0
        text_total_len = len(text)

        # Sliding window split: core overlapping chunk logic
        while start_idx < text_total_len:
            # Calculate end index for current chunk
            end_idx = start_idx + chunk_length
            chunk = text[start_idx:end_idx]
            chunks.append(chunk)
            # Step = chunk_length - overlap,实现滑动重叠
            start_idx += chunk_length - overlap
            # Fallback: handle last chunk that doesn't reach full length
            if start_idx + chunk_length > text_total_len and start_idx < text_total_len:
                chunk = text[-chunk_length:]
                chunks.append(chunk)
                break

        return chunks

    @staticmethod
    async def batch_encode(batch: list) -> list:
        """Encode a batch of texts.

        Args:
            batch: List of texts to encode.

        Returns:
            List of embeddings as lists of floats.
        """
        # Batch encode, same parameters as get_embedding for consistent vector format
        embeddings = await asyncio.to_thread(
            TextEmbeddings.embedding_model.encode,
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Keep normalized for consistent retrieval
        )

        # Unified return as list format, consistent with get_embedding's return embedding.tolist()
        return embeddings.tolist()


def is_uuid_literal(val: str) -> bool:
    """Check if string is a valid UUID literal.

    Args:
        val: String to check.

    Returns:
        True if string is valid UUID format.
    """
    return (
        re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            val.lower(),
        )
        is not None
    )

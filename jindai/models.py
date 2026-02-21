"""DB Objects"""

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
from sqlalchemy import (Boolean, DateTime, ForeignKey, Index, Integer,
                        PrimaryKeyConstraint, String, Text, UniqueConstraint,
                        asc, desc, exists, or_, select, text, update)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID, insert
from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.orm import (Mapped, declarative_base, mapped_column,
                            relationship, validates)
from sqlalchemy.sql import func

from .config import instance as config
from .helpers import AutoUnloadSentenceTransformer, jieba

engine = create_async_engine(config.database)
session_factory = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
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
    """所有 ORM 模型的基类"""

    # 为所有表添加默认的 created_at 字段（如果表中没有则自动忽略）
    __abstract__ = True
    __table_args__ = {"schema": "public"}

    # 通用字段配置：主键 ID 统一使用 UUID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="主键ID"
    )

    def as_dict(self) -> dict:
        data = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            # 处理 UUID 类型
            if value is None:
                data[column.name] = None
            elif isinstance(value, uuid.UUID):
                data[column.name] = str(value)
            # 处理 datetime 类型
            elif isinstance(value, datetime):
                data[column.name] = value.isoformat()
            # 处理数组/JSONB/向量类型
            elif isinstance(value, Base):
                data[column.name] = value.as_dict()
            elif isinstance(value, (int, str, float, list)):
                data[column.name] = value
            elif isinstance(value, datetime):
                data[column.name] = value.isoformat()
            elif isinstance(column.type, JSONB):
                data[column.name] = value
            else:
                data[column.name] = f"<{value} of {column.type}>"
        return data


# 基础模型类
class Dataset(Base):
    __tablename__ = "dataset"
    __table_args__ = (
        UniqueConstraint("name", name="dataset_name_key"),  # 唯一约束
        {
            "comment": "数据集信息表",
        },  # 表注释 + 指定模式
    )

    order_weight: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="排序权重"
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="数据集名称")
    tags: Mapped[List[str] | None] = mapped_column(ARRAY(Text), comment="标签列表")

    # 关联关系：一个数据集对应多个段落
    paragraphs: Mapped[List["Paragraph"]] = relationship(
        "Paragraph", back_populates="dataset_obj", cascade="all, delete-orphan"
    )

    @staticmethod
    async def get(name, auto_create=True):
        async with get_db_session() as session:
            ds = await session.execute(select(Dataset).filter(Dataset.name == name))
            ds = ds.scalar_one_or_none()
            if ds is None and auto_create:
                ds = Dataset(name=name)
                session.add(ds)
                await session.flush()

        return ds

    async def rename_dataset(self, new_name):
        """Rename dataset and update all related paragraphs

        :param new_name: New dataset name
        :type new_name: str
        :return: Dataset ID
        :rtype: uuid.UUID
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
        return self.id

    @staticmethod
    async def get_hierarchy() -> list:
        """Get hierarchical structure of datasets

        :return: Nested list representing dataset hierarchy
        :rtype: list
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
    __tablename__ = "user_info"
    __table_args__ = (
        UniqueConstraint("username", name="user_info_username_key"),  # 用户名唯一
        {
            "comment": "用户表",
        },
    )

    username: Mapped[str] = mapped_column(String(64), nullable=False, comment="用户名")
    roles: Mapped[List[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False, comment="用户角色列表"
    )
    datasets: Mapped[List[UUID] | None] = mapped_column(
        ARRAY(Text), default=list, comment="有权限的数据集列表"
    )

    # 关联关系：一个用户对应多个操作历史/任务
    histories: Mapped[List["History"]] = relationship(
        "History", back_populates="user", cascade="all, delete-orphan"
    )
    tasks: Mapped[List["TaskDBO"]] = relationship(
        "TaskDBO", back_populates="user", cascade="all, delete-orphan"
    )


class History(Base):
    __tablename__ = "history"
    __table_args__ = {
        "comment": "用户操作历史表",
    }

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user_info.id", ondelete="CASCADE"),
        nullable=False,
        comment="关联用户ID",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.current_timestamp(), nullable=False, comment="创建时间"
    )
    queries: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False, comment="查询记录列表（JSON）"
    )

    # 关联关系：关联到用户
    user: Mapped["UserInfo"] = relationship("UserInfo", back_populates="histories")


class QueryFilters(BaseModel):
    # 搜索关键字，默认为 "*"
    q: str = "*"

    # 基础过滤
    ids: Optional[List[int]] = None
    datasets: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    sourcePage: Optional[int] = None
    outline: Optional[List[str]] = None
    authors: Optional[List[str]] = None

    # 控制逻辑
    embeddings: Optional[bool] = None  # False 时排除有向量的数据
    groupBy: Optional[str] = None  # 对应 Paragraph 的字段名

    # 排序与分页
    sort: Optional[Union[str, List[str]]] = ""
    offset: int = 0
    limit: Optional[int] = None


class Paragraph(Base):
    __tablename__ = "paragraph"
    __table_args__ = (
        # 索引定义（与原表一致）
        PrimaryKeyConstraint("id", "dataset", name="paragraph_part_pk"),
        Index("fki_dataset", "dataset"),
        Index("idx_paragraph_author", "author"),
        Index("idx_paragraph_keywords", "keywords", postgresql_using="gin"),
        Index("idx_paragraph_outline", "outline"),
        Index("idx_paragraph_pagenum", "pagenum"),
        Index("idx_paragraph_pdate", "pdate"),
        Index("idx_paragraph_source", "source_url", "source_page", "pagenum"),
        {
            "comment": "段落表",
        },
    )

    source_url: Mapped[str | None] = mapped_column(String(1024), comment="源URL地址")
    source_page: Mapped[int | None] = mapped_column(
        Integer, comment="源文件页码（如PDF页码）"
    )
    author: Mapped[str | None] = mapped_column(String(128), comment="作者")
    pdate: Mapped[datetime | None] = mapped_column(DateTime, comment="发布日期")
    outline: Mapped[str | None] = mapped_column(Text, comment="大纲/摘要")
    content: Mapped[str | None] = mapped_column(Text, comment="段落内容")
    pagenum: Mapped[str | None] = mapped_column(Text, comment="页码")
    lang: Mapped[str] = mapped_column(String(16), default="zh", comment="语言")
    extdata: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="扩展元数据（自定义JSON字段）",
    )
    dataset: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset.id"),
        nullable=False,
        primary_key=True,
        comment="关联数据集ID",
    )
    keywords: Mapped[List[str] | None] = mapped_column(
        ARRAY(Text), comment="关键词列表"
    )

    dataset_obj: Mapped["Dataset"] = relationship(
        "Dataset", back_populates="paragraphs", lazy="joined"
    )
    text_embeddings: Mapped[List["TextEmbeddings"]] = relationship(
        "TextEmbeddings", back_populates="paragraph", cascade="all, delete-orphan"
    )

    @validates("content")
    def normalize_content(self, key, val):
        return val.replace("\x00", "")

    @validates("keywords")
    def normalize_keywords(self, key, val) -> list:
        if val:
            val = list({v.strip().lower() for v in val if v.strip()})
        return val

    @staticmethod
    def from_dict(data, ignored_fields=["id", "dataset_name"], **kwargs) -> "Paragraph":
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

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            if self.extdata is None:
                self.extdata = {}
            return self.extdata.get(key)

    def __setitem__(self, key, val) -> None:
        if hasattr(self, key):
            setattr(self, key, val)
        else:
            if self.extdata is None:
                self.extdata = {}
            self.extdata[key] = val

    def __delitem__(self, key) -> None:
        if key in self.extdata:
            del self.extdata[key]

    def as_dict(self) -> dict:
        data = super().as_dict()
        data["dataset_name"] = self.dataset_obj.name if self.dataset_obj else None
        return data

    @staticmethod
    async def build_query(query_filters: QueryFilters):
        query = select(Paragraph)
        filters = []
        query_embedding = None
        search = query_filters.q

        if group_field_name := query_filters.groupBy:
            group_column = (
                getattr(Paragraph, group_field_name, None) if group_field_name else None
            )
            if group_column is not None:
                query = query.distinct(group_column).order_by(
                    group_column
                )  # DISTINCT ON requires order_by to start with group column

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

        # Author Filter
        if authors := query_filters.authors:
            filters.append(Paragraph.author.in_(authors))

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
                    & (Paragraph.dataset == TextEmbeddings.dataset),  # 分片需要
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

        # Sorting
        if sort_by := query_filters.sort:
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

                col = getattr(Paragraph, col_name, None)
                if col is not None:
                    sorts.append(sort_func(col))

            query = query.order_by(*sorts)

        # Pagination
        if offset := query_filters.offset:
            query = query.offset(offset)
        if limit := query_filters.limit:
            query = query.limit(limit)

        return query


class Terms(MBase):  # terms has no `id`, and cannot as_dict()
    __tablename__ = "terms"
    __table_args__ = {
        "comment": "词汇表",
    }

    term: Mapped[str] = mapped_column(
        String, nullable=False, comment="词汇", primary_key=True
    )

    @staticmethod
    async def starting_with(prefix: str) -> List[str]:
        if not prefix:
            return []
        async with get_db_session() as session:
            q = await session.execute(
                select(Terms)
                .filter(Terms.term.startswith(prefix))
                .with_only_columns(Terms.term)
            )
            return list(q.scalars())

    async def store(
        words: list[str],
    ):  # Clean the input to ensure uniqueness in the batch
        async with get_db_session() as session:
            async with session.begin():  # Start a transaction
                # Prepare the data dictionaries
                data = [{"term": w} for w in words]

                # Execute a bulk insert
                # Note: This assumes 'term' is a unique column.
                # If not, use standard session.execute(insert(Terms), data)
                stmt = insert(Terms).values(data)

                # If you have a UNIQUE constraint on 'term', use this to skip duplicates:
                stmt = stmt.on_conflict_do_nothing(index_elements=["term"])

                await session.execute(stmt)
                # No need for manual commit if using 'async with session.begin()'

        return words


class TaskDBO(Base):
    __tablename__ = "task_dbo"
    __table_args__ = {
        "comment": "任务表",
    }

    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="任务名称")
    pipeline: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False, comment="任务流水线（JSON）"
    )
    resume_next: Mapped[bool] = mapped_column(
        Boolean, default=False, comment="是否忽略错误继续运行"
    )
    last_run: Mapped[datetime | None] = mapped_column(DateTime, comment="最后运行时间")
    concurrent: Mapped[int] = mapped_column(Integer, default=3, comment="并发数")
    shortcut_map: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, default=dict, nullable=False, comment="快捷方式映射（JSON）"
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user_info.id", ondelete="CASCADE"),
        nullable=False,
        comment="关联用户ID",
    )
    shared: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否共享")

    # 关联关系：关联到用户
    user: Mapped["UserInfo"] = relationship("UserInfo", back_populates="tasks")

    @staticmethod
    async def get(id_or_name):
        """Get TaskDBO by ID or name

        :param id_or_name: UUID string or name to search for
        :type id_or_name: str
        :return: TaskDBO instance or None
        :rtype: TaskDBO | None
        """
        async with get_db_session() as session:
            if is_uuid_literal(id_or_name):
                return await session.get(TaskDBO, uuid.UUID(id_or_name))
            else:
                result = await session.execute(
                    select(TaskDBO).filter(TaskDBO.name.contains(id_or_name))
                )
                return result.scalar_one_or_none()


redis_client = redis.Redis.from_url(config.redis.rstrip("/") + "/2")


# 缓存核心配置（可根据业务灵活修改）
CACHE_EXPIRE_SECONDS = 60 * 10  # 缓存默认过期时间：5分钟
CACHE_KEY_PREFIX = "query_cache:"  # 缓存key前缀，方便redis中区分缓存数据


def redis_auto_renew_cache(cache_key=None):
    """
    Redis 自动续期缓存的装饰器工厂函数，为业务函数提供自动缓存/访问续期/自动过期的缓存能力。
    核心特性：每次命中缓存时，自动刷新缓存的过期时间，实现「热点数据永不失效、冷数据自动过期」。

    :param cache_key: 可选，自定义缓存key的生成方法/函数，接收被装饰函数的*args和**kwargs作为入参，
                      需返回字符串类型的缓存key片段；为None时，自动拼接函数入参生成缓存key
    :type cache_key: callable | None

    :return: 实际作用于业务函数的装饰器函数
    :rtype: function

    缓存key生成规则优先级 & 拼接规则：
        1. 传参指定cache_key函数 → 缓存key = 全局前缀 + 被装饰函数名 + 自定义函数返回的key片段
        2. 未指定cache_key → 缓存key = 全局前缀 + 被装饰函数名 + 有序拼接的位置参数+关键字参数
        3. 空key场景：若生成的key片段为空，则不会执行任何缓存相关逻辑，直接执行原函数

    核心执行逻辑流程：
        1. 调用被装饰函数前，先根据规则生成完整缓存key；
        2. 若缓存key有效，从Redis查询缓存数据，命中则自动续期并直接返回缓存结果；
        3. 缓存未命中/无有效key时，执行原业务函数获取执行结果；
        4. 若函数返回有效结果+存在有效缓存key，将结果序列化后写入Redis并设置过期时间；
        5. 统一返回业务函数执行结果/缓存结果。

    依赖说明：
        - 全局常量 CACHE_KEY_PREFIX: 所有缓存key的统一前缀，用于Redis key的命名隔离
        - 全局常量 CACHE_EXPIRE_SECONDS: 缓存默认过期时长，单位为秒
        - 全局实例 redis_client: 已初始化的Redis客户端，需支持 get/expire/setex 方法
        - 序列化方式：统一使用json.dumps/json.loads，支持所有可JSON序列化的返回结果

    使用限制：
        1. 被装饰函数的返回值必须是可JSON序列化的对象（dict/list/str/int等）；
        2. 关键字参数拼接时会做排序，保证入参顺序不同但内容一致时生成相同key；
        3. 若自定义cache_key函数返回空值，则跳过缓存逻辑，直接执行原函数。
    """

    def decorator(func):
        func.cache_key_method = cache_key

        @wraps(func)  # 保留原函数的属性（函数名、注释等）
        def wrapper(*args, **kwargs):
            if func.cache_key_method is not None:
                cache_key = f"{func.cache_key_method(*args, **kwargs) or ''}"
            else:
                args_str = "_".join(map(str, args))
                kwargs_str = "_".join([f"{k}_{v}" for k, v in sorted(kwargs.items())])
                cache_key = f"{args_str}_{kwargs_str}"

            if cache_key:
                cache_key = f"{CACHE_KEY_PREFIX}{func.__name__}_{cache_key}"
                cached_data = redis_client.get(cache_key)
                if cached_data is not None:
                    redis_client.expire(cache_key, CACHE_EXPIRE_SECONDS)
                    return json.loads(cached_data)

            result = func(*args, **kwargs)

            if result is not None and cache_key:
                # 写入缓存+设置过期时间，实现「自动过期」
                redis_client.setex(
                    name=cache_key,
                    time=CACHE_EXPIRE_SECONDS,
                    value=json.dumps(result),  # 统一转字符串，支持任意可序列化结果
                )

            return result

        return wrapper

    return decorator


class TextEmbeddings(Base):
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
            "comment": "文本嵌入表",
        },
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("paragraph.id", ondelete="CASCADE"),
        primary_key=True,
        comment="段落ID",
    )

    dataset: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset.id", ondelete="CASCADE"),
        primary_key=True,
        comment="数据集ID",
    )

    chunk_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        comment="分块ID",
    )

    embedding: Mapped[Vector] = mapped_column(
        Vector(config.embedding_dims), nullable=False, comment="文本嵌入向量"
    )

    paragraph: Mapped["Paragraph"] = relationship(
        "Paragraph", back_populates="text_embeddings"
    )

    @staticmethod
    @redis_auto_renew_cache()
    def get_embedding_sync(text: str):
        embedding = TextEmbeddings.embedding_model.encode(
            text.strip(),
            convert_to_numpy=True,  # 返回numpy数组，方便后续处理
            normalize_embeddings=True,  # 归一化向量，提升检索效果
        )

        return embedding.tolist()

    @staticmethod
    async def get_embedding(text: str):
        """
        将多语言文本转换为embedding向量

        Args:
            text: 输入的多语言文本（支持中文、英文、日文等100+种语言）

        Returns:
            np.ndarray: 生成的embedding向量（维度为384）

        Raises:
            ValueError: 输入文本为空时抛出异常
        """
        return await asyncio.to_thread(TextEmbeddings.get_embedding_sync, text)

    @staticmethod
    def get_chunks(text: str, chunk_length: int, overlap: int) -> list:
        """
        长文本切分，带重叠窗口避免语义割裂
        Args:
            text: 待切分的长文本（语段/整页文本）
            chunk_length: 每个文本块的字符长度
            overlap: 相邻文本块的重叠字符长度
        Returns:
            list: 二维列表，每个元素为 单文本块的embedding向量(list格式)，与你get_embedding返回格式一致
        Raises:
            ValueError: 输入文本为空/切分参数非法时抛出异常
        """
        # 基础校验
        if not text.strip():
            return []
        if chunk_length <= 0 or overlap < 0 or overlap >= chunk_length:
            return []

        text = text.strip()
        chunks = []
        start_idx = 0
        text_total_len = len(text)

        # 滑动窗口切分长文本：核心重叠分块逻辑
        while start_idx < text_total_len:
            # 计算当前块的结束下标
            end_idx = start_idx + chunk_length
            chunk = text[start_idx:end_idx]
            chunks.append(chunk)
            # 步进 = 块长度 - 重叠长度，实现滑动重叠
            start_idx += chunk_length - overlap
            # 兜底：处理最后一个不足长度的块，避免遗漏文本
            if start_idx + chunk_length > text_total_len and start_idx < text_total_len:
                chunk = text[-chunk_length:]
                chunks.append(chunk)
                break

        return chunks

    @staticmethod
    async def batch_encode(batch):

        # 批量encode，参数与get_embedding完全一致，保证向量格式统一
        embeddings = await asyncio.to_thread(
            TextEmbeddings.embedding_model.encode,
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 保持归一化，和单句向量一致性检索
        )

        # 统一返回list格式，与get_embedding的return embedding.tolist()完全对齐
        return embeddings.tolist()


def is_uuid_literal(val: str) -> bool:
    """Check if string is a valid UUID literal

    :param val: String to check
    :type val: str
    :return: True if string is valid UUID format
    :rtype: bool
    """
    return (
        re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            val.lower(),
        )
        is not None
    )

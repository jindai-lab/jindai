"""DB Objects"""

from functools import wraps
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List

import json
import jieba
import redis
from pgvector.sqlalchemy import Vector
from sqlalchemy import (Boolean, DateTime, ForeignKey, Index, Integer, String,
                        Text, UniqueConstraint, asc, create_engine, desc,
                        exists, or_, select, text, update)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import (Mapped, declarative_base, mapped_column,
                            relationship, scoped_session, sessionmaker)
from sqlalchemy.sql import func

from .config import instance as config
from .helpers import AutoUnloadSentenceTransformer


engine = create_engine(config.database)
session_factory = sessionmaker(bind=engine)
db_session = scoped_session(session_factory)
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

    def as_dict(self):
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
    def get(name, auto_create=True):
        ds = db_session.query(Dataset).filter(Dataset.name == name).first()
        if ds is None and auto_create:
            ds = Dataset(name=name)
            db_session.add(ds)
            db_session.commit()
        return ds

    def rename_dataset(self, new_name):
        if self.name == new_name:
            return
        ds = Dataset.get(new_name)
        if ds:
            stmt = (
                update(Paragraph)
                .where(Paragraph.dataset == self.id)
                .values({"dataset": ds.id})
            )
            db_session.execute(stmt)
            db_session.delete(self)
        else:
            self.name = new_name
        db_session.commit()
        return self.id

    @staticmethod
    def get_hierarchy():

        def _dataset_sort_key(ds: Dataset):
            return len(ds.name.split("--")), ds.order_weight, ds.name

        datasets = db_session.query(Dataset).all()
        sorted_datasets = sorted(datasets, key=_dataset_sort_key)
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


class Paragraph(Base):
    __tablename__ = "paragraph"
    __table_args__ = (
        # 索引定义（与原表一致）
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
        comment="关联数据集ID",
    )
    keywords: Mapped[List[str] | None] = mapped_column(
        ARRAY(Text), comment="关键词列表"
    )

    dataset_obj: Mapped["Dataset"] = relationship(
        "Dataset", back_populates="paragraphs"
    )
    text_embeddings: Mapped[List["TextEmbeddings"]] = relationship(
        "TextEmbeddings", back_populates="paragraph", cascade="all, delete-orphan"
    )

    def as_dict(self):
        data = super().as_dict()
        data["dataset"] = self.dataset_obj.name
        return data

    @staticmethod
    def build_query(query_data):
        query = db_session.query(Paragraph)
        filters = []

        search = query_data["search"]

        if datasets := query_data.get("datasets"):
            dataset_filters = [Dataset.name.in_(datasets)]
            for dataset_prefix in datasets:
                dataset_filters.append(Dataset.name.ilike(f"{dataset_prefix}--%"))
            filters.append(
                Paragraph.dataset.in_(
                    [
                        _[0]
                        for _ in db_session.query(Dataset)
                        .filter(or_(*dataset_filters))
                        .with_entities(Dataset.id)
                        .all()
                    ]
                )
            )

        if sources := query_data.get("sources"):
            source_filters = [Paragraph.source_url.in_(sources)]
            for source in sources:
                source_filters.append(Paragraph.source_url.ilike(f"{source}%"))
            filters.append(or_(*source_filters))

        if query_data.get("embeddings") == False:
            param = (
                ~exists()
                .where(TextEmbeddings.id == Paragraph.id)
                .correlate(TextEmbeddings)
            )
            filters.append(param)

        if search.startswith("?"):
            param = text(search[1:])
            filters.append(param)
        elif search.startswith("*"):
            param = Paragraph.content.ilike(f"%{search.strip('*')}%")
            filters.append(param)
        elif search.startswith(":") or query_data.get("embeddings"):
            if 'total' not in query_data:
                query_embedding = TextEmbeddings.get_embedding(search.strip(":"))
                sub_stmt = (
                    select(
                        TextEmbeddings.id,
                        TextEmbeddings.embedding.cosine_distance(query_embedding).label(
                            "dist"
                        ),
                    )
                    .order_by(TextEmbeddings.embedding.cosine_distance(query_embedding))
                    .subquery("embs")
                )
                query = query.join(sub_stmt, Paragraph.id == sub_stmt.c.id).order_by(
                    sub_stmt.c.dist
                )
        else:
            param = Paragraph.keywords.contains(
                [_.strip().lower() for _ in jieba.cut(search) if _.strip()]
            )
            filters.append(param)

        query = query.filter(*filters)

        if sort_string := query_data.get("sort", ""):
            assert isinstance(
                sort_string, (list, str)
            ), "Sort must be list of strings or a string seperated by commas"
            if isinstance(sort_string, list):
                sort_string = ",".join(sort_string)

            order_params = []
            # 1. 拆分字符串
            parts = sort_string.split(",")

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # 2. 判断排序方向
                if part.startswith("-"):
                    column_name = part[1:]
                    sort_func = desc
                else:
                    column_name = part
                    sort_func = asc

                # 3. 获取模型属性并生成排序对象
                column = getattr(Paragraph, column_name, None)
                if column is not None:
                    order_params.append(sort_func(column))

            query = query.order_by(*order_params)

        if offset := query_data.get("offset", 0):
            query = query.offset(offset)

        if limit := query_data.get("limit", 0):
            query = query.limit(limit)

        return query


class Terms(Base):
    __tablename__ = "terms"
    __table_args__ = {
        "comment": "词汇表",
    }

    term: Mapped[str] = mapped_column(String, nullable=False, comment="词汇")


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
    def get(id_or_name):
        query = db_session.query(TaskDBO)
        if is_uuid_literal(id_or_name):
            return query.get(uuid.UUID(id_or_name))
        else:
            return query.filter(TaskDBO.name.contains(id_or_name)).first()


redis_client = redis.Redis(**config.redis)


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
                cache_key = f'{CACHE_KEY_PREFIX}{func.__name__}_{cache_key}'
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
            postgresql_using="ivfflat",
            postgresql_with={"lists": "5000"},
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
    def get_embedding(text: str):
        """
        将多语言文本转换为embedding向量

        Args:
            text: 输入的多语言文本（支持中文、英文、日文等100+种语言）

        Returns:
            np.ndarray: 生成的embedding向量（维度为384）

        Raises:
            ValueError: 输入文本为空时抛出异常
        """
        embedding = TextEmbeddings.embedding_model.encode(
            text.strip(),
            convert_to_numpy=True,  # 返回numpy数组，方便后续处理
            normalize_embeddings=True,  # 归一化向量，提升检索效果
        )

        return embedding.tolist()

    @staticmethod
    def get_embedding_chunks(text: str, chunk_length: int, overlap: int):
        """
        长文本切分+批量生成embedding向量，带重叠窗口避免语义割裂
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

        # 批量encode，参数与get_embedding完全一致，保证向量格式统一
        embeddings = TextEmbeddings.embedding_model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 保持归一化，和单句向量一致性检索
        )

        # 统一返回list格式，与get_embedding的return embedding.tolist()完全对齐
        return embeddings.tolist()


def is_uuid_literal(val: str):
    return (
        re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            val.lower(),
        )
        is not None
    )

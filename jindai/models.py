"""DB Objects"""

import uuid
from datetime import datetime
from typing import Any, Dict, List

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .app import db


class Base(db.Model):
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
                data[column.name] = f'<{value} of {column.type}>'
        print(data)
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
        JSONB, default=list, nullable=False, comment="用户角色列表（JSON）"
    )
    datasets: Mapped[List[UUID] | None] = mapped_column(
        JSONB, default=dict, comment="有权限的数据集列表（JSON）"
    )
    token: Mapped[str | None] = mapped_column(Text, comment="用户令牌")

    # 关联关系：一个用户对应多个操作历史/令牌
    histories: Mapped[List["History"]] = relationship(
        "History", back_populates="user", cascade="all, delete-orphan"
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
        data['dataset'] = self.dataset_obj.name
        return data


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
    creator: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="创建者用户名"
    )
    shared: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否共享")


class TextEmbeddings(Base):
    __tablename__ = "text_embeddings"
    __table_args__ = (
        Index(
            "idx_embedding_uuid_cosine",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": "100"},
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

    collection: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="嵌入集合名称"
    )
    embedding: Mapped[Vector] = mapped_column(
        Vector(384), nullable=False, comment="文本嵌入向量"  # 384 维向量
    )

    paragraph: Mapped["Paragraph"] = relationship(
        "Paragraph", back_populates="text_embeddings"
    )


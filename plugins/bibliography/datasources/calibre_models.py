# Generated with sqlacodegen
from typing import Any, Optional, List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import datetime

from sqlalchemy import Boolean, CheckConstraint, Column, ForeignKey, Index, Integer, LargeBinary, REAL, TIMESTAMP, Table, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql.sqltypes import NullType

class Base(DeclarativeBase):
    pass


class Annotations(Base):
    __tablename__ = 'annotations'
    __table_args__ = (
        UniqueConstraint('book', 'user_type', 'user', 'format', 'annot_type', 'annot_id'),
        Index('annot_idx', 'book')
    )

    book: Mapped[int] = mapped_column(Integer, nullable=False)
    format: Mapped[str] = mapped_column(Text, nullable=False)
    user_type: Mapped[str] = mapped_column(Text, nullable=False)
    user: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[float] = mapped_column(REAL, nullable=False)
    annot_id: Mapped[str] = mapped_column(Text, nullable=False)
    annot_type: Mapped[str] = mapped_column(Text, nullable=False)
    annot_data: Mapped[str] = mapped_column(Text, nullable=False)
    searchable_text: Mapped[str] = mapped_column(Text, nullable=False, server_default=text('""'))
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class AnnotationsDirtied(Base):
    __tablename__ = 'annotations_dirtied'

    book: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


t_annotations_fts = Table(
    'annotations_fts', Base.metadata,
    Column('searchable_text', NullType)
)


class AnnotationsFtsConfig(Base):
    __tablename__ = 'annotations_fts_config'

    k: Mapped[str] = mapped_column(NullType, primary_key=True)
    v: Mapped[Optional[str]] = mapped_column(NullType)


class AnnotationsFtsData(Base):
    __tablename__ = 'annotations_fts_data'

    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
    block: Mapped[Optional[bytes]] = mapped_column(LargeBinary)


class AnnotationsFtsDocsize(Base):
    __tablename__ = 'annotations_fts_docsize'

    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
    sz: Mapped[Optional[bytes]] = mapped_column(LargeBinary)


class AnnotationsFtsIdx(Base):
    __tablename__ = 'annotations_fts_idx'

    segid: Mapped[str] = mapped_column(NullType, primary_key=True)
    term: Mapped[str] = mapped_column(NullType, primary_key=True)
    pgno: Mapped[Optional[str]] = mapped_column(NullType)


t_annotations_fts_stemmed = Table(
    'annotations_fts_stemmed', Base.metadata,
    Column('searchable_text', NullType)
)


class AnnotationsFtsStemmedConfig(Base):
    __tablename__ = 'annotations_fts_stemmed_config'

    k: Mapped[str] = mapped_column(NullType, primary_key=True)
    v: Mapped[Optional[str]] = mapped_column(NullType)


class AnnotationsFtsStemmedData(Base):
    __tablename__ = 'annotations_fts_stemmed_data'

    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
    block: Mapped[Optional[bytes]] = mapped_column(LargeBinary)


class AnnotationsFtsStemmedDocsize(Base):
    __tablename__ = 'annotations_fts_stemmed_docsize'

    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
    sz: Mapped[Optional[bytes]] = mapped_column(LargeBinary)


class AnnotationsFtsStemmedIdx(Base):
    __tablename__ = 'annotations_fts_stemmed_idx'

    segid: Mapped[str] = mapped_column(NullType, primary_key=True)
    term: Mapped[str] = mapped_column(NullType, primary_key=True)
    pgno: Mapped[Optional[str]] = mapped_column(NullType)


class Authors(Base):
    __tablename__ = 'authors'

    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    link: Mapped[str] = mapped_column(Text, nullable=False, server_default=text('""'))
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
    sort: Mapped[Optional[str]] = mapped_column(Text)


class Books(Base):
    __tablename__ = 'books'
    __table_args__ = (
        Index('authors_idx', 'author_sort'),
        Index('books_idx', 'sort')
    )

    title: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'Unknown'"))
    series_index: Mapped[float] = mapped_column(REAL, nullable=False, server_default=text('1.0'))
    path: Mapped[str] = mapped_column(Text, nullable=False, server_default=text('""'))
    last_modified: Mapped[datetime.datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=text('"2000-01-01 00:00:00+00:00"'))
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
    sort: Mapped[Optional[str]] = mapped_column(Text)
    timestamp: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))
    pubdate: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))
    author_sort: Mapped[Optional[str]] = mapped_column(Text)
    uuid: Mapped[Optional[str]] = mapped_column(Text)
    has_cover: Mapped[Optional[bool]] = mapped_column(Boolean, server_default=text('0'))


class BooksAuthorsLink(Base):
    __tablename__ = 'books_authors_link'
    __table_args__ = (
        UniqueConstraint('book', 'author'),
        Index('books_authors_link_aidx', 'author'),
        Index('books_authors_link_bidx', 'book')
    )

    book: Mapped[int] = mapped_column(Integer, nullable=False)
    author: Mapped[int] = mapped_column(Integer, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class BooksLanguagesLink(Base):
    __tablename__ = 'books_languages_link'
    __table_args__ = (
        UniqueConstraint('book', 'lang_code'),
        Index('books_languages_link_aidx', 'lang_code'),
        Index('books_languages_link_bidx', 'book')
    )

    book: Mapped[int] = mapped_column(Integer, nullable=False)
    lang_code: Mapped[int] = mapped_column(Integer, nullable=False)
    item_order: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class BooksPluginData(Base):
    __tablename__ = 'books_plugin_data'
    __table_args__ = (
        UniqueConstraint('book', 'name'),
    )

    book: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    val: Mapped[str] = mapped_column(Text, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class BooksPublishersLink(Base):
    __tablename__ = 'books_publishers_link'
    __table_args__ = (
        Index('books_publishers_link_aidx', 'publisher'),
        Index('books_publishers_link_bidx', 'book')
    )

    book: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    publisher: Mapped[int] = mapped_column(Integer, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class BooksRatingsLink(Base):
    __tablename__ = 'books_ratings_link'
    __table_args__ = (
        UniqueConstraint('book', 'rating'),
        Index('books_ratings_link_aidx', 'rating'),
        Index('books_ratings_link_bidx', 'book')
    )

    book: Mapped[int] = mapped_column(Integer, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class BooksSeriesLink(Base):
    __tablename__ = 'books_series_link'
    __table_args__ = (
        Index('books_series_link_aidx', 'series'),
        Index('books_series_link_bidx', 'book')
    )

    book: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    series: Mapped[int] = mapped_column(Integer, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class BooksTagsLink(Base):
    __tablename__ = 'books_tags_link'
    __table_args__ = (
        UniqueConstraint('book', 'tag'),
        Index('books_tags_link_aidx', 'tag'),
        Index('books_tags_link_bidx', 'book')
    )

    book: Mapped[int] = mapped_column(Integer, nullable=False)
    tag: Mapped[int] = mapped_column(Integer, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class Comments(Base):
    __tablename__ = 'comments'
    __table_args__ = (
        Index('comments_idx', 'book'),
    )

    book: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    text_: Mapped[str] = mapped_column('text', Text, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class ConversionOptions(Base):
    __tablename__ = 'conversion_options'
    __table_args__ = (
        UniqueConstraint('format', 'book'),
        Index('conversion_options_idx_a', 'format'),
        Index('conversion_options_idx_b', 'book')
    )

    format: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
    book: Mapped[Optional[int]] = mapped_column(Integer)


class CustomColumn1(Base):
    __tablename__ = 'custom_column_1'
    __table_args__ = (
        Index('custom_column_1_idx', 'book'),
    )

    value: Mapped[str] = mapped_column(Text, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
    book: Mapped[Optional[int]] = mapped_column(Integer, unique=True)


class CustomColumns(Base):
    __tablename__ = 'custom_columns'
    __table_args__ = (
        Index('custom_columns_idx', 'label'),
    )

    label: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    datatype: Mapped[str] = mapped_column(Text, nullable=False)
    mark_for_delete: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('0'))
    editable: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('1'))
    display: Mapped[str] = mapped_column(Text, nullable=False, server_default=text('"{}"'))
    is_multiple: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('0'))
    normalized: Mapped[bool] = mapped_column(Boolean, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class Data(Base):
    __tablename__ = 'data'
    __table_args__ = (
        UniqueConstraint('book', 'format'),
        Index('data_idx', 'book'),
        Index('formats_idx', 'format')
    )

    book: Mapped[int] = mapped_column(Integer, nullable=False)
    format: Mapped[str] = mapped_column(Text, nullable=False)
    uncompressed_size: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class Feeds(Base):
    __tablename__ = 'feeds'

    title: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    script: Mapped[str] = mapped_column(Text, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class Identifiers(Base):
    __tablename__ = 'identifiers'
    __table_args__ = (
        UniqueConstraint('book', 'type'),
    )

    book: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False, server_default=text('"isbn"'))
    val: Mapped[str] = mapped_column(Text, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class Languages(Base):
    __tablename__ = 'languages'
    __table_args__ = (
        Index('languages_idx', 'lang_code'),
    )

    lang_code: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    link: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class LastReadPositions(Base):
    __tablename__ = 'last_read_positions'
    __table_args__ = (
        UniqueConstraint('user', 'device', 'book', 'format'),
        Index('lrp_idx', 'book')
    )

    book: Mapped[int] = mapped_column(Integer, nullable=False)
    format: Mapped[str] = mapped_column(Text, nullable=False)
    user: Mapped[str] = mapped_column(Text, nullable=False)
    device: Mapped[str] = mapped_column(Text, nullable=False)
    cfi: Mapped[str] = mapped_column(Text, nullable=False)
    epoch: Mapped[float] = mapped_column(REAL, nullable=False)
    pos_frac: Mapped[float] = mapped_column(REAL, nullable=False, server_default=text('0'))
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class LibraryId(Base):
    __tablename__ = 'library_id'

    uuid: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class MetadataDirtied(Base):
    __tablename__ = 'metadata_dirtied'

    book: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class Preferences(Base):
    __tablename__ = 'preferences'

    key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    val: Mapped[str] = mapped_column(Text, nullable=False)
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class Publishers(Base):
    __tablename__ = 'publishers'
    __table_args__ = (
        Index('publishers_idx', 'name'),
    )

    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    link: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
    sort: Mapped[Optional[str]] = mapped_column(Text)


class Ratings(Base):
    __tablename__ = 'ratings'
    __table_args__ = (
        CheckConstraint('rating > -1 AND rating < 11'),
    )

    link: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
    rating: Mapped[Optional[int]] = mapped_column(Integer, unique=True)


class Series(Base):
    __tablename__ = 'series'
    __table_args__ = (
        Index('series_idx', 'name'),
    )

    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    link: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
    sort: Mapped[Optional[str]] = mapped_column(Text)


class Tags(Base):
    __tablename__ = 'tags'
    __table_args__ = (
        Index('tags_idx', 'name'),
    )

    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    link: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)


class BooksPagesLink(Books):
    __tablename__ = 'books_pages_link'
    __table_args__ = (
        Index('books_pages_link_pidx', 'needs_scan'),
    )

    pages: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    algorithm: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    format: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    format_size: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    needs_scan: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('0'))
    book: Mapped[Optional[int]] = mapped_column(ForeignKey('books.id', ondelete='CASCADE'), primary_key=True)
    timestamp: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))

class Format(str, Enum):
    """电子书格式枚举"""
    EPUB = "EPUB"
    MOBI = "MOBI"
    PDF = "PDF"
    AZW3 = "AZW3"
    DOCX = "DOCX"
    TXT = "TXT"
    CBZ = "CBZ"
    CBR = "CBR"

@dataclass
class AuthorInfo:
    """作者信息"""
    id: int
    name: str
    sort: Optional[str] = None
    link: str = ""

@dataclass
class SeriesInfo:
    """丛书信息"""
    id: int
    name: str
    sort: Optional[str] = None
    link: str = ""

@dataclass
class PublisherInfo:
    """出版商信息"""
    id: int
    name: str
    sort: Optional[str] = None
    link: str = ""

@dataclass
class TagInfo:
    """标签信息"""
    id: int
    name: str
    link: str = ""

@dataclass
class LanguageInfo:
    """语言信息"""
    id: int
    code: str
    name: Optional[str] = None  # 语言名称（如果需要）

@dataclass
class RatingInfo:
    """评分信息"""
    id: int
    rating: Optional[int] = None  # 1-10分
    stars: float = 0.0  # 转换为星标（0-5星）

@dataclass
class IdentifierInfo:
    """标识符信息"""
    id: int
    type: str  # isbn, doi, etc.
    value: str

@dataclass
class FileFormat:
    """文件格式信息"""
    format: str
    size: int  # 字节
    name: str  # 文件名
    path: Optional[str] = None  # 文件路径

@dataclass
class CommentInfo:
    """评论/简介信息"""
    id: int
    text: str

@dataclass
class CustomColumnValue:
    """自定义列值"""
    column_id: int
    column_name: str
    column_label: str
    value: Any
    datatype: str

@dataclass
class PageInfo:
    """页码信息"""
    pages: int = 0
    algorithm: int = 0
    format: str = ""
    format_size: int = 0
    needs_scan: bool = False
    timestamp: Optional[datetime] = None

@dataclass
class ReadingProgress:
    """阅读进度"""
    format: str
    user: str
    device: str
    cfi: str  # 内容片段标识符
    position_frac: float  # 阅读进度百分比
    timestamp: float  # 时间戳
    epoch: float

@dataclass
class Annotation:
    """批注信息"""
    id: int
    format: str
    user_type: str
    user: str
    timestamp: float
    annot_id: str
    annot_type: str
    annot_data: str
    searchable_text: str = ""

@dataclass
class CompleteBookInfo:
    """完整的书籍信息表"""
    
    # 基本信息
    id: int
    title: str
    uuid: Optional[str] = None
    sort: Optional[str] = None  # 排序用的标题
    path: str = ""  # 书籍文件路径
    has_cover: bool = False  # 是否有封面
    
    # 时间信息
    timestamp: Optional[datetime] = None  # 添加时间
    pubdate: Optional[datetime] = None  # 出版日期
    last_modified: Optional[datetime] = None  # 最后修改时间
    
    # 作者信息
    authors: List[AuthorInfo] = field(default_factory=list)
    author_sort: Optional[str] = None  # 作者排序字符串
    
    # 丛书信息
    series: Optional[SeriesInfo] = None
    series_index: float = 1.0  # 在丛书中的序号
    
    # 出版信息
    publisher: Optional[PublisherInfo] = None
    
    # 标签
    tags: List[TagInfo] = field(default_factory=list)
    
    # 语言
    languages: List[LanguageInfo] = field(default_factory=list)
    
    # 评分
    rating: Optional[RatingInfo] = None
    
    # 标识符
    identifiers: List[IdentifierInfo] = field(default_factory=list)
    
    # 文件格式
    formats: Dict[str, FileFormat] = field(default_factory=dict)  # format -> FileFormat
    
    # 简介
    comment: Optional[CommentInfo] = None
    
    # 自定义列
    custom_columns: Dict[str, CustomColumnValue] = field(default_factory=dict)
    
    # 页码信息
    pages: Optional[PageInfo] = None
    
    # 阅读进度
    reading_progress: List[ReadingProgress] = field(default_factory=list)
    
    # 批注
    annotations: List[Annotation] = field(default_factory=list)
    
    def get_main_format(self) -> Optional[str]:
        """获取主要格式"""
        if not self.formats:
            return None
        # 优先返回 EPUB 或 PDF
        for fmt in ['EPUB', 'PDF', 'MOBI', 'AZW3']:
            if fmt in self.formats:
                return fmt
        return list(self.formats.keys())[0]
    
    def get_file_size(self, format: Optional[str] = None) -> int:
        """获取文件大小"""
        if format and format in self.formats:
            return self.formats[format].size
        elif self.formats:
            # 返回所有格式的总大小
            return sum(f.size for f in self.formats.values())
        return 0
    
    def get_authors_string(self) -> str:
        """获取作者字符串"""
        return ' & '.join(a.name for a in self.authors)
    
    def get_tags_string(self) -> str:
        """获取标签字符串"""
        return ', '.join(t.name for t in self.tags)
    
    def get_languages_string(self) -> str:
        """获取语言字符串"""
        return ', '.join(l.code for l in self.languages)
    
    def get_rating_stars(self) -> float:
        """获取星级评分（0-5）"""
        if self.rating and self.rating.rating:
            return self.rating.rating / 2.0
        return 0.0
    
    def get_reading_progress(self, user: str = "default", device: str = "main") -> Optional[float]:
        """获取指定用户的阅读进度"""
        for progress in self.reading_progress:
            if progress.user == user and progress.device == device:
                return progress.position_frac
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'title': self.title,
            'uuid': self.uuid,
            'sort': self.sort,
            'path': self.path,
            'has_cover': self.has_cover,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'pubdate': self.pubdate.isoformat() if self.pubdate else None,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None,
            'authors': [{'id': a.id, 'name': a.name, 'sort': a.sort} for a in self.authors],
            'author_sort': self.author_sort,
            'series': {
                'id': self.series.id,
                'name': self.series.name,
                'sort': self.series.sort
            } if self.series else None,
            'series_index': self.series_index,
            'publisher': {
                'id': self.publisher.id,
                'name': self.publisher.name
            } if self.publisher else None,
            'tags': [{'id': t.id, 'name': t.name} for t in self.tags],
            'languages': [{'code': l.code} for l in self.languages],
            'rating': {
                'value': self.rating.rating,
                'stars': self.get_rating_stars()
            } if self.rating else None,
            'identifiers': [{'type': i.type, 'value': i.value} for i in self.identifiers],
            'formats': {
                fmt: {
                    'size': f.size,
                    'name': f.name,
                    'path': f.path
                } for fmt, f in self.formats.items()
            },
            'comment': self.comment.text if self.comment else None,
            'custom_columns': {
                label: {
                    'value': col.value,
                    'datatype': col.datatype
                } for label, col in self.custom_columns.items()
            },
            'pages': {
                'count': self.pages.pages,
                'needs_scan': self.pages.needs_scan
            } if self.pages else None,
            'reading_progress': [
                {
                    'format': p.format,
                    'user': p.user,
                    'progress': p.position_frac,
                    'cfi': p.cfi
                } for p in self.reading_progress
            ],
            'annotations_count': len(self.annotations)
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        authors_str = self.get_authors_string()
        return f"《{self.title}》" + (f" by {authors_str}" if authors_str else "")

# 使用示例
def create_book_info_from_orm_models(book: Books, session) -> CompleteBookInfo:
    """从 ORM 模型创建完整的书籍信息"""
    
    # 获取作者
    authors = []
    author_links = session.query(BooksAuthorsLink).filter(BooksAuthorsLink.book == book.id).all()
    for link in author_links:
        author = session.query(Authors).filter(Authors.id == link.author).first()
        if author:
            authors.append(AuthorInfo(
                id=author.id,
                name=author.name,
                sort=author.sort,
                link=author.link
            ))
    
    # 获取丛书
    series_info = None
    series_link = session.query(BooksSeriesLink).filter(BooksSeriesLink.book == book.id).first()
    if series_link:
        series = session.query(Series).filter(Series.id == series_link.series).first()
        if series:
            series_info = SeriesInfo(
                id=series.id,
                name=series.name,
                sort=series.sort,
                link=series.link
            )
    
    # 获取出版商
    publisher_info = None
    publisher_link = session.query(BooksPublishersLink).filter(BooksPublishersLink.book == book.id).first()
    if publisher_link:
        publisher = session.query(Publishers).filter(Publishers.id == publisher_link.publisher).first()
        if publisher:
            publisher_info = PublisherInfo(
                id=publisher.id,
                name=publisher.name,
                sort=publisher.sort,
                link=publisher.link
            )
    
    # 获取标签
    tags = []
    tag_links = session.query(BooksTagsLink).filter(BooksTagsLink.book == book.id).all()
    for link in tag_links:
        tag = session.query(Tags).filter(Tags.id == link.tag).first()
        if tag:
            tags.append(TagInfo(
                id=tag.id,
                name=tag.name,
                link=tag.link
            ))
    
    # 获取语言
    languages = []
    lang_links = session.query(BooksLanguagesLink).filter(BooksLanguagesLink.book == book.id).all()
    for link in lang_links:
        lang = session.query(Languages).filter(Languages.id == link.lang_code).first()
        if lang:
            languages.append(LanguageInfo(
                id=lang.id,
                code=lang.lang_code
            ))
    
    # 获取评分
    rating_info = None
    rating_link = session.query(BooksRatingsLink).filter(BooksRatingsLink.book == book.id).first()
    if rating_link:
        rating = session.query(Ratings).filter(Ratings.id == rating_link.rating).first()
        if rating:
            rating_info = RatingInfo(
                id=rating.id,
                rating=rating.rating
            )
    
    # 获取标识符
    identifiers = []
    identifier_records = session.query(Identifiers).filter(Identifiers.book == book.id).all()
    for id_rec in identifier_records:
        identifiers.append(IdentifierInfo(
            id=id_rec.id,
            type=id_rec.type,
            value=id_rec.val
        ))
    
    # 获取文件格式
    formats = {}
    data_records = session.query(Data).filter(Data.book == book.id).all()
    for data in data_records:
        formats[data.format] = FileFormat(
            format=data.format,
            size=data.uncompressed_size,
            name=data.name
        )
    
    # 获取简介
    comment_info = None
    comment = session.query(Comments).filter(Comments.book == book.id).first()
    if comment:
        comment_info = CommentInfo(
            id=comment.id,
            text=comment.text_
        )
    
    # 获取页码信息
    pages_info = None
    pages_link = session.query(BooksPagesLink).filter(BooksPagesLink.book == book.id).first()
    if pages_link:
        pages_info = PageInfo(
            pages=pages_link.pages,
            algorithm=pages_link.algorithm,
            format=pages_link.format,
            format_size=pages_link.format_size,
            needs_scan=pages_link.needs_scan,
            timestamp=pages_link.timestamp
        )
    
    # 获取阅读进度
    reading_progress = []
    progress_records = session.query(LastReadPositions).filter(LastReadPositions.book == book.id).all()
    for prog in progress_records:
        reading_progress.append(ReadingProgress(
            format=prog.format,
            user=prog.user,
            device=prog.device,
            cfi=prog.cfi,
            position_frac=prog.pos_frac,
            timestamp=prog.epoch,
            epoch=prog.epoch
        ))
    
    # 获取批注
    annotations = []
    annotation_records = session.query(Annotations).filter(Annotations.book == book.id).all()
    for ann in annotation_records:
        annotations.append(Annotation(
            id=ann.id,
            format=ann.format,
            user_type=ann.user_type,
            user=ann.user,
            timestamp=ann.timestamp,
            annot_id=ann.annot_id,
            annot_type=ann.annot_type,
            annot_data=ann.annot_data,
            searchable_text=ann.searchable_text
        ))
    
    # 创建完整信息对象
    return CompleteBookInfo(
        id=book.id,
        title=book.title,
        uuid=book.uuid,
        sort=book.sort,
        path=book.path,
        has_cover=bool(book.has_cover),
        timestamp=book.timestamp,
        pubdate=book.pubdate,
        last_modified=book.last_modified,
        authors=authors,
        author_sort=book.author_sort,
        series=series_info,
        series_index=book.series_index,
        publisher=publisher_info,
        tags=tags,
        languages=languages,
        rating=rating_info,
        identifiers=identifiers,
        formats=formats,
        comment=comment_info,
        pages=pages_info,
        reading_progress=reading_progress,
        annotations=annotations
    )

from typing import List, Dict, Optional, Generator
from sqlalchemy.orm import Session, joinedload, contains_eager
from sqlalchemy import func
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_books_complete_info(
    session: Session, 
    batch_size: int = 100,
    include_annotations: bool = True,
    include_reading_progress: bool = True,
    include_custom_columns: bool = True
) -> List[CompleteBookInfo]:
    """
    获取数据库中所有书籍的完整信息
    
    Args:
        session: SQLAlchemy 数据库会话
        batch_size: 批量处理的大小，避免一次性加载过多数据
        include_annotations: 是否包含批注信息
        include_reading_progress: 是否包含阅读进度
        include_custom_columns: 是否包含自定义列
        
    Returns:
        List[CompleteBookInfo]: 所有书籍的完整信息列表
    """
    
    # 获取书籍总数
    total_books = session.query(func.count(Books.id)).scalar()
    logger.info(f"开始获取 {total_books} 本书的完整信息")
    
    all_books_info = []
    processed = 0
    
    # 分批处理
    for offset in range(0, total_books, batch_size):
        # 获取当前批次的书籍
        books_batch = session.query(Books)\
            .order_by(Books.id)\
            .offset(offset)\
            .limit(batch_size)\
            .all()
        
        logger.info(f"处理第 {offset//batch_size + 1} 批，共 {len(books_batch)} 本书")
        
        # 获取当前批次所有书籍的ID
        book_ids = [book.id for book in books_batch]
        
        # 批量预加载所有关联数据
        preloaded_data = preload_all_related_data(
            session, 
            book_ids,
            include_annotations,
            include_reading_progress
        )
        
        # 为每本书创建完整信息
        for book in books_batch:
            try:
                book_info = create_book_info_from_preloaded(
                    book=book,
                    preloaded=preloaded_data,
                    include_custom_columns=include_custom_columns
                )
                all_books_info.append(book_info)
                processed += 1
                
                if processed % 100 == 0:
                    logger.info(f"已处理 {processed}/{total_books} 本书")
                    
            except Exception as e:
                logger.error(f"处理书籍 ID {book.id} 时出错: {e}")
                continue
    
    logger.info(f"完成！成功获取 {len(all_books_info)} 本书的信息")
    return all_books_info


def preload_all_related_data(
    session: Session,
    book_ids: List[int],
    include_annotations: bool = True,
    include_reading_progress: bool = True
) -> Dict:
    """
    预加载所有相关的数据，避免N+1查询问题
    """
    preloaded = {}
    
    # 1. 预加载作者关联
    author_links = session.query(BooksAuthorsLink)\
        .filter(BooksAuthorsLink.book.in_(book_ids))\
        .all()
    
    author_ids = [link.author for link in author_links]
    authors = session.query(Authors)\
        .filter(Authors.id.in_(author_ids))\
        .all()
    
    preloaded['author_links'] = author_links
    preloaded['authors'] = {a.id: a for a in authors}
    
    # 2. 预加载丛书关联
    series_links = session.query(BooksSeriesLink)\
        .filter(BooksSeriesLink.book.in_(book_ids))\
        .all()
    
    series_ids = [link.series for link in series_links]
    series = session.query(Series)\
        .filter(Series.id.in_(series_ids))\
        .all()
    
    preloaded['series_links'] = series_links
    preloaded['series'] = {s.id: s for s in series}
    
    # 3. 预加载出版商关联
    publisher_links = session.query(BooksPublishersLink)\
        .filter(BooksPublishersLink.book.in_(book_ids))\
        .all()
    
    publisher_ids = [link.publisher for link in publisher_links]
    publishers = session.query(Publishers)\
        .filter(Publishers.id.in_(publisher_ids))\
        .all()
    
    preloaded['publisher_links'] = publisher_links
    preloaded['publishers'] = {p.id: p for p in publishers}
    
    # 4. 预加载标签关联
    tag_links = session.query(BooksTagsLink)\
        .filter(BooksTagsLink.book.in_(book_ids))\
        .all()
    
    tag_ids = [link.tag for link in tag_links]
    tags = session.query(Tags)\
        .filter(Tags.id.in_(tag_ids))\
        .all()
    
    preloaded['tag_links'] = tag_links
    preloaded['tags'] = {t.id: t for t in tags}
    
    # 5. 预加载语言关联
    lang_links = session.query(BooksLanguagesLink)\
        .filter(BooksLanguagesLink.book.in_(book_ids))\
        .all()
    
    lang_ids = [link.lang_code for link in lang_links]
    languages = session.query(Languages)\
        .filter(Languages.id.in_(lang_ids))\
        .all()
    
    preloaded['lang_links'] = lang_links
    preloaded['languages'] = {l.id: l for l in languages}
    
    # 6. 预加载评分关联
    rating_links = session.query(BooksRatingsLink)\
        .filter(BooksRatingsLink.book.in_(book_ids))\
        .all()
    
    rating_ids = [link.rating for link in rating_links]
    ratings = session.query(Ratings)\
        .filter(Ratings.id.in_(rating_ids))\
        .all()
    
    preloaded['rating_links'] = rating_links
    preloaded['ratings'] = {r.id: r for r in ratings}
    
    # 7. 预加载标识符
    identifiers = session.query(Identifiers)\
        .filter(Identifiers.book.in_(book_ids))\
        .all()
    preloaded['identifiers'] = identifiers
    
    # 8. 预加载文件格式
    data_files = session.query(Data)\
        .filter(Data.book.in_(book_ids))\
        .all()
    preloaded['data_files'] = data_files
    
    # 9. 预加载评论/简介
    comments = session.query(Comments)\
        .filter(Comments.book.in_(book_ids))\
        .all()
    preloaded['comments'] = comments
    
    # 10. 预加载页码信息
    pages_links = session.query(BooksPagesLink)\
        .filter(BooksPagesLink.book.in_(book_ids))\
        .all()
    preloaded['pages_links'] = pages_links
    
    # 11. 预加载阅读进度（如果需要）
    if include_reading_progress:
        reading_progress = session.query(LastReadPositions)\
            .filter(LastReadPositions.book.in_(book_ids))\
            .all()
        preloaded['reading_progress'] = reading_progress
    
    # 12. 预加载批注（如果需要）
    if include_annotations:
        annotations = session.query(Annotations)\
            .filter(Annotations.book.in_(book_ids))\
            .all()
        preloaded['annotations'] = annotations
    
    # 13. 预加载自定义列（如果需要）
    custom_columns_data = preload_custom_columns(session, book_ids)
    preloaded['custom_columns'] = custom_columns_data
    
    return preloaded


def preload_custom_columns(session: Session, book_ids: List[int]) -> Dict:
    """
    预加载所有自定义列的数据
    """
    # 获取所有自定义列的定义
    custom_columns = session.query(CustomColumns).all()
    
    custom_data = {}
    
    for col in custom_columns:
        # 根据列名获取对应的表和数据
        table_name = f"custom_column_{col.id}"
        try:
            # 动态获取自定义列的表
            if hasattr(session, 'query'):
                # 这里需要根据实际的列类型来查询
                # 假设自定义列的值存储在对应的表中
                table_class = globals().get(f'CustomColumn{col.id}')
                if table_class:
                    records = session.query(table_class)\
                        .filter(table_class.book.in_(book_ids))\
                        .all()
                    custom_data[col.id] = {
                        'column': col,
                        'records': records
                    }
        except Exception as e:
            logger.warning(f"获取自定义列 {col.label} 数据时出错: {e}")
    
    return custom_data


def create_book_info_from_preloaded(
    book: Books,
    preloaded: Dict,
    include_custom_columns: bool = True
) -> CompleteBookInfo:
    """
    使用预加载的数据创建书籍完整信息
    """
    
    # 1. 处理作者
    authors = []
    author_links = [link for link in preloaded['author_links'] if link.book == book.id]
    for link in author_links:
        author = preloaded['authors'].get(link.author)
        if author:
            authors.append(AuthorInfo(
                id=author.id,
                name=author.name,
                sort=author.sort,
                link=author.link
            ))
    
    # 2. 处理丛书
    series_info = None
    series_link = next((link for link in preloaded['series_links'] if link.book == book.id), None)
    if series_link:
        series = preloaded['series'].get(series_link.series)
        if series:
            series_info = SeriesInfo(
                id=series.id,
                name=series.name,
                sort=series.sort,
                link=series.link
            )
    
    # 3. 处理出版商
    publisher_info = None
    publisher_link = next((link for link in preloaded['publisher_links'] if link.book == book.id), None)
    if publisher_link:
        publisher = preloaded['publishers'].get(publisher_link.publisher)
        if publisher:
            publisher_info = PublisherInfo(
                id=publisher.id,
                name=publisher.name,
                sort=publisher.sort,
                link=publisher.link
            )
    
    # 4. 处理标签
    tags = []
    tag_links = [link for link in preloaded['tag_links'] if link.book == book.id]
    for link in tag_links:
        tag = preloaded['tags'].get(link.tag)
        if tag:
            tags.append(TagInfo(
                id=tag.id,
                name=tag.name,
                link=tag.link
            ))
    
    # 5. 处理语言
    languages = []
    lang_links = [link for link in preloaded['lang_links'] if link.book == book.id]
    for link in lang_links:
        lang = preloaded['languages'].get(link.lang_code)
        if lang:
            languages.append(LanguageInfo(
                id=lang.id,
                code=lang.lang_code
            ))
    
    # 6. 处理评分
    rating_info = None
    rating_link = next((link for link in preloaded['rating_links'] if link.book == book.id), None)
    if rating_link:
        rating = preloaded['ratings'].get(rating_link.rating)
        if rating:
            rating_info = RatingInfo(
                id=rating.id,
                rating=rating.rating
            )
    
    # 7. 处理标识符
    identifiers = []
    for id_rec in preloaded['identifiers']:
        if id_rec.book == book.id:
            identifiers.append(IdentifierInfo(
                id=id_rec.id,
                type=id_rec.type,
                value=id_rec.val
            ))
    
    # 8. 处理文件格式
    formats = {}
    for data in preloaded['data_files']:
        if data.book == book.id:
            formats[data.format] = FileFormat(
                format=data.format,
                size=data.uncompressed_size,
                name=data.name,
                path=f"{book.path}/{data.name}.{data.format.lower()}"
            )
    
    # 9. 处理评论
    comment_info = None
    comment = next((c for c in preloaded['comments'] if c.book == book.id), None)
    if comment:
        comment_info = CommentInfo(
            id=comment.id,
            text=comment.text_
        )
    
    # 10. 处理页码信息
    pages_info = None
    pages_link = next((p for p in preloaded['pages_links'] if p.book == book.id), None)
    if pages_link:
        pages_info = PageInfo(
            pages=pages_link.pages,
            algorithm=pages_link.algorithm,
            format=pages_link.format,
            format_size=pages_link.format_size,
            needs_scan=pages_link.needs_scan,
            timestamp=pages_link.timestamp
        )
    
    # 11. 处理阅读进度
    reading_progress = []
    if 'reading_progress' in preloaded:
        for prog in preloaded['reading_progress']:
            if prog.book == book.id:
                reading_progress.append(ReadingProgress(
                    format=prog.format,
                    user=prog.user,
                    device=prog.device,
                    cfi=prog.cfi,
                    position_frac=prog.pos_frac,
                    timestamp=prog.epoch,
                    epoch=prog.epoch
                ))
    
    # 12. 处理批注
    annotations = []
    if 'annotations' in preloaded:
        for ann in preloaded['annotations']:
            if ann.book == book.id:
                annotations.append(Annotation(
                    id=ann.id,
                    format=ann.format,
                    user_type=ann.user_type,
                    user=ann.user,
                    timestamp=ann.timestamp,
                    annot_id=ann.annot_id,
                    annot_type=ann.annot_type,
                    annot_data=ann.annot_data,
                    searchable_text=ann.searchable_text
                ))
    
    # 13. 处理自定义列
    custom_columns = {}
    if include_custom_columns and 'custom_columns' in preloaded:
        for col_id, col_data in preloaded['custom_columns'].items():
            column = col_data['column']
            records = [r for r in col_data['records'] if r.book == book.id]
            if records:
                # 这里需要根据实际的列类型来处理值
                record = records[0]
                value = getattr(record, 'value', None)
                custom_columns[column.label] = CustomColumnValue(
                    column_id=column.id,
                    column_name=column.name,
                    column_label=column.label,
                    value=value,
                    datatype=column.datatype
                )
    
    # 创建完整信息对象
    return CompleteBookInfo(
        id=book.id,
        title=book.title,
        uuid=book.uuid,
        sort=book.sort,
        path=book.path,
        has_cover=bool(book.has_cover),
        timestamp=book.timestamp,
        pubdate=book.pubdate,
        last_modified=book.last_modified,
        authors=authors,
        author_sort=book.author_sort,
        series=series_info,
        series_index=book.series_index,
        publisher=publisher_info,
        tags=tags,
        languages=languages,
        rating=rating_info,
        identifiers=identifiers,
        formats=formats,
        comment=comment_info,
        custom_columns=custom_columns,
        pages=pages_info,
        reading_progress=reading_progress,
        annotations=annotations
    )


def get_all_books_generator(
    session: Session,
    batch_size: int = 100
) -> Generator[CompleteBookInfo, None, None]:
    """
    使用生成器方式逐批获取书籍信息，适合处理大量数据
    """
    total_books = session.query(func.count(Books.id)).scalar()
    
    for offset in range(0, total_books, batch_size):
        books_batch = session.query(Books)\
            .order_by(Books.id)\
            .offset(offset)\
            .limit(batch_size)\
            .all()
        
        book_ids = [book.id for book in books_batch]
        preloaded = preload_all_related_data(session, book_ids)
        
        for book in books_batch:
            try:
                yield create_book_info_from_preloaded(book, preloaded)
            except Exception as e:
                logger.error(f"处理书籍 ID {book.id} 时出错: {e}")
                continue


def get_books_by_filter(
    session: Session,
    author_name: Optional[str] = None,
    tag_name: Optional[str] = None,
    series_name: Optional[str] = None,
    publisher_name: Optional[str] = None,
    language_code: Optional[str] = None,
    format_type: Optional[str] = None,
    title_contains: Optional[str] = None,
    min_rating: Optional[int] = None,
    has_cover: Optional[bool] = None,
    limit: Optional[int] = None
) -> List[CompleteBookInfo]:
    """
    根据条件筛选书籍
    """
    query = session.query(Books).distinct()
    
    # 应用过滤条件
    if author_name:
        query = query.join(BooksAuthorsLink)\
            .join(Authors)\
            .filter(Authors.name.ilike(f'%{author_name}%'))
    
    if tag_name:
        query = query.join(BooksTagsLink)\
            .join(Tags)\
            .filter(Tags.name.ilike(f'%{tag_name}%'))
    
    if series_name:
        query = query.join(BooksSeriesLink)\
            .join(Series)\
            .filter(Series.name.ilike(f'%{series_name}%'))
    
    if publisher_name:
        query = query.join(BooksPublishersLink)\
            .join(Publishers)\
            .filter(Publishers.name.ilike(f'%{publisher_name}%'))
    
    if language_code:
        query = query.join(BooksLanguagesLink)\
            .join(Languages)\
            .filter(Languages.lang_code == language_code)
    
    if format_type:
        query = query.join(Data).filter(Data.format == format_type.upper())
    
    if title_contains:
        query = query.filter(Books.title.ilike(f'%{title_contains}%'))
    
    if min_rating:
        query = query.join(BooksRatingsLink)\
            .join(Ratings)\
            .filter(Ratings.rating >= min_rating)
    
    if has_cover is not None:
        query = query.filter(Books.has_cover == has_cover)
    
    if limit:
        query = query.limit(limit)
    
    books = query.all()
    book_ids = [book.id for book in books]
    
    if not book_ids:
        return []
    
    preloaded = preload_all_related_data(session, book_ids)
    
    return [
        create_book_info_from_preloaded(book, preloaded)
        for book in books
    ]


def get_books_statistics(session: Session) -> Dict:
    """
    获取书籍统计信息
    """
    stats = {
        'total_books': session.query(func.count(Books.id)).scalar(),
        'total_authors': session.query(func.count(Authors.id)).scalar(),
        'total_tags': session.query(func.count(Tags.id)).scalar(),
        'total_series': session.query(func.count(Series.id)).scalar(),
        'total_publishers': session.query(func.count(Publishers.id)).scalar(),
        'total_formats': session.query(func.count(Data.id)).scalar(),
        'books_with_cover': session.query(Books).filter(Books.has_cover == True).count(),
        'books_without_cover': session.query(Books).filter(Books.has_cover == False).count(),
    }
    
    # 各格式的数量
    format_counts = session.query(
        Data.format, 
        func.count(Data.book).label('count')
    ).group_by(Data.format).all()
    
    stats['formats'] = {f: c for f, c in format_counts}
    
    # 评分分布
    rating_dist = session.query(
        Ratings.rating,
        func.count(BooksRatingsLink.book).label('count')
    ).join(BooksRatingsLink).group_by(Ratings.rating).all()
    
    stats['rating_distribution'] = {r: c for r, c in rating_dist if r}
    
    return stats

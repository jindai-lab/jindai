import datetime
import sqlite3
import os
import urllib.parse
from uuid import UUID

from sqlalchemy import update

from jindai.storage import storage
from jindai.models import Dataset, Paragraph, get_db_session
from jindai.pipeline import DataSourceStage, PipelineStage


class CalibreLibraryDataSource(DataSourceStage):
    """
    Import paragraphs from PDF
    @zhs 从 PDF 中导入语段
    """

    def get_calibre_books_safe(self, library_path: str) -> list[Paragraph]:
        # 将路径转换为 SQLite URI 格式，以确保只读打开
        db_path = os.path.abspath(os.path.join(library_path, "metadata.db"))
        if not os.path.exists(db_path):
            return []

        # 构造只读 URI (Windows 下需要处理驱动器号)
        db_uri = f"file:{urllib.parse.quote(db_path)}?mode=ro"

        try:
            # 使用 uri=True 开启只读连接
            conn = sqlite3.connect(db_uri, uri=True)
            cursor = conn.cursor()

            # SQL 查询：
            # 1. 增加了 WHERE 子句过滤后缀
            # 2. 使用 LOWER() 确保不区分大小写
            query = """
            SELECT 
                b.id,
                b.title, 
                b.path, 
                d.name, 
                d.format, 
                b.pubdate,
                (SELECT GROUP_CONCAT(a.name, ' & ') 
                FROM authors a 
                JOIN books_authors_link bal ON a.id = bal.author 
                WHERE bal.book = b.id) as author_names
            FROM books b
            JOIN data d ON b.id = d.book
            WHERE LOWER(d.format) IN ('pdf', 'epub')
            """

            cursor.execute(query)
            rows = cursor.fetchall()

            books_info = []
            for row in rows:
                book_id, title, folder_path, file_name, ext, pub_date, authors = row

                # 处理年份：Calibre 的空年份通常是 '0101-01-01'
                year = (
                    int(pub_date[:4])
                    if pub_date and not pub_date.startswith("0101")
                    else ""
                )

                # 拼接相对路径
                # Calibre 存储的 path 是相对于书库根目录的目录路径
                relative_file_path = os.path.join(
                    folder_path, f"{file_name}.{ext.lower()}"
                )

                books_info.append(
                    Paragraph(
                        author=authors,
                        pdate=datetime.datetime(year, 1, 1) if year else None,
                        outline=title,
                        content=relative_file_path,
                        extdata={"book_id": book_id},
                    )
                )

            conn.close()
            return books_info

        except sqlite3.Error as e:
            print(f"读取数据库时出错（请检查路径或权限）: {e}")
            return []

    def apply_params(
        self,
        dataset_name="",
        lang="auto",
        content="",
        formats="epub,pdf",
    ) -> None:
        """
        Args:
            dataset_name (DATASET):
                Dataset name
                @zhs 数据集名称
            lang (LANG):
                Language
                @zhs 语言标识
            content (FILES):
                Calibre Library Path
                @zhs Calibre 书库路径
            formats (str):
                File format
                @zhs 允许的文件格式
        """
        self.dataset_name = dataset_name
        self.lang = lang
        self.paths = content
        self.formats = tuple(formats.lower().split(","))

    async def fetch(self):
        paths = await PipelineStage.parse_paths(self.paths)
        dsid = (await Dataset.get(self.dataset_name)).id

        async with get_db_session() as session:
            for path in paths:
                books = self.get_calibre_books_safe(storage.safe_join(path))
                for book in books:
                    if self.formats and book.content.lower().endswith(self.formats):
                        book.content = os.path.join(path, book.content)
                        book.dataset = dsid
                        # update moved books
                        await session.execute(
                            update(Paragraph)
                            .filter(
                                Paragraph.source_url.contains(f' ({book.extdata["book_id"]})/'),
                                Paragraph.extdata["book_id"] == book.extdata["book_id"],
                                Paragraph.source_url != book.content,
                            )
                            .values(source_url=book.content)
                        )
                        # assign outline and author info
                        # await session.execute(
                        #     update(Paragraph).filter(Paragraph.source_url == book.content)
                        #     .values(outline=book.outline, author=book.author)
                        # )
                        yield book

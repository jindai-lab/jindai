"""PGVector Extension"""

from jindai import Plugin, PipelineStage
from jindai.models import Paragraph
import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2 import sql
from sentence_transformers import SentenceTransformer
from bson import ObjectId
import uuid
from PyMongoWrapper import F

psycopg2.extras.register_uuid()


class PGVectorPlugin(Plugin):
    """PGVector Extension"""

    def __init__(
        self, pmanager, model_name="paraphrase-multilingual-MiniLM-L12-v2", **pgconfig
    ):
        super().__init__(pmanager)
        self.pgconfig = pgconfig
        self.model = SentenceTransformer(model_name)
        self.init_database()
        self.register_pipelines(self.get_pipelines())

    @property
    def pgconn(self):
        conn = psycopg2.connect(**self.pgconfig)
        cursor = conn.cursor()
        return conn, cursor
    
    def init_database(self):
        """
        初始化PostgreSQL数据库，创建必要的表和索引

        Args:
            None

        Returns:
            None
        """
        conn, cursor = self.pgconn
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            create_table_sql = sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS text_embeddings (
                    id UUID PRIMARY KEY,
                    collection VARCHAR(255) NOT NULL,
                    embedding vector({dim}) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            ).format(dim=sql.Literal(384))  # 假设embedding向量的维度为384
            cursor.execute(create_table_sql)

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_uuid_cosine 
                ON text_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
                """
            )

            conn.commit()
        except Exception as e:
            print(f"Error during initialization: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def save_embedding(self, collection, unique_id, vector: np.ndarray):
        """
        保存embedding向量到PostgreSQL数据库

        Args:
            collection: 要保存的embedding向量的集合名称
            unique_id: 要保存的embedding向量的唯一ID
            vector: 要保存的embedding向量（numpy数组）

        Returns:
            None
        """
        conn, cursor = self.pgconn
        if isinstance(unique_id, ObjectId):
            unique_id = uuid.UUID(bytes=unique_id.binary + b"\0" * 4)

        try:
            insert_sql = "INSERT INTO text_embeddings (id, collection, embedding) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;"
            cursor.execute(
                insert_sql,
                (
                    unique_id,
                    collection,
                    vector.tolist(),
                ),
            )

            conn.commit()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_embedding(self, text: str) -> np.ndarray:
        """
        将多语言文本转换为embedding向量

        Args:
            text: 输入的多语言文本（支持中文、英文、日文等100+种语言）

        Returns:
            np.ndarray: 生成的embedding向量（维度为384）

        Raises:
            ValueError: 输入文本为空时抛出异常
        """
        embedding = self.model.encode(
            text.strip(),
            convert_to_numpy=True,  # 返回numpy数组，方便后续处理
            normalize_embeddings=True,  # 归一化向量，提升检索效果
        )

        return embedding

    def search_embedding(self, query_text: str, top_k: int = 5) -> list:
        query_embedding = self.get_embedding(query_text)
        conn, cursor = self.pgconn
        try:
            search_sql = "SELECT collection, substring(replace(id::text, '-', ''), 1, 24) FROM text_embeddings ORDER BY embedding <=> %s::vector LIMIT %s;"
            cursor.execute(search_sql, (query_embedding.tolist(), top_k))
            results = cursor.fetchall()
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_pipelines(self_plugin):

        class PGVectorSearch(PipelineStage):

            def __init__(self, top_k: int = 5):
                """
                Args:
                    top_k (int): Number of top results to return
                        @zhs 返回的结果数量
                """
                super().__init__()
                self.top_k = top_k

            def resolve(self, paragraph):
                results = self_plugin.search_embedding(paragraph.content, self.top_k)
                for collection, id in results:
                    yield Paragraph.get_coll(collection).first(F.id == ObjectId(id))

        class PGVectorSave(PipelineStage):
            
            def resolve(self, paragraph):
                embedding = self_plugin.get_embedding(paragraph.content)
                self_plugin.save_embedding(
                    paragraph.db.name, paragraph._id, embedding
                )
                yield paragraph

        return [PGVectorSave, PGVectorSearch]

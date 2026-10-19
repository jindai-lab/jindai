
import sys, os
sys.path.insert(0, '/app')
os.chdir('/app')
from sqlalchemy import select, exists
from sqlalchemy.dialects import postgresql
from jindai.models import Paragraph, EmbeddingPendingQueue, TextEmbeddings
stmt = (
    select(Paragraph)
    .join(EmbeddingPendingQueue,
          (EmbeddingPendingQueue.id == Paragraph.id)
          & (EmbeddingPendingQueue.source_id == Paragraph.source_id))
    .where(~exists(select(1).where(TextEmbeddings.id == Paragraph.id)))
    .with_only_columns(Paragraph.id, Paragraph.source_id, Paragraph.content)
    .order_by(Paragraph.id)
    .limit(10000)
)
print(str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})))

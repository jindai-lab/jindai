"""Shared helpers for jindai batch_operation scripts.

All scripts run INSIDE the jindai container so they reuse the app's own
config, ORM models and chunking rules (single source of truth):

    docker exec -w /app jindai /app/.venv/bin/python \
        /app/batch_operation/<script>.py [args]

Host paths: this folder is ~/jindai-lab/jindai/batch_operation (bind of /app).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

# --- bootstrap: make `import jindai...` work and locate config.yaml -------
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
os.chdir(APP_ROOT)  # jindai.config resolves config.yaml from cwd unless -c given

import asyncpg  # noqa: E402
from jindai.config import config  # noqa: E402
from jindai.models import TextEmbeddings  # noqa: E402  (for get_chunks parity)

# --- embedding endpoint (from config.yaml, reporting only) ----------------
# The HTTP calls themselves are made by jindai.embedutils.EmbeddingClient.
EMBED_BASE = config.embedding_api_base
EMBED_MODEL = config.embedding_model
EMBED_KEY = config.embedding_api_key
DIMS = config.embedding_dims

CHUNK_LEN = 200
CHUNK_OVERLAP = 50
ONEDRIVE_LIKE = "OneDrive/%"

INSERT_EMBEDDING_SQL = """
    INSERT INTO text_embeddings (id, source_id, chunk_id, embedding)
    VALUES ($1, $2, $3, $4::text::halfvec)
    ON CONFLICT DO NOTHING
"""


def get_chunks(content: Optional[str]) -> list[str]:
    """Chunk a paragraph exactly like the backend does."""
    return TextEmbeddings.get_chunks(content or "", CHUNK_LEN, CHUNK_OVERLAP)


def pg_dsn() -> str:
    """Convert the SQLAlchemy URL in config.yaml to an asyncpg DSN."""
    return config.database.replace("postgresql+asyncpg://", "postgresql://", 1)


async def get_conn() -> asyncpg.Connection:
    return await asyncpg.connect(dsn=pg_dsn())


def setup_logging(name: str, logfile: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False  # keep our own format; root has no useful handler
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class Progress:
    """Throughput / ETA tracker for long batch runs."""

    def __init__(self, logger: logging.Logger, unit: str = "chunks"):
        self.log = logger
        self.unit = unit
        self.t0 = time.time()
        self.done = 0
        self.total = 0
        self._last = self.t0

    def set_total(self, n: int) -> None:
        self.total = n

    def add(self, n: int) -> None:
        self.done += n

    def line(self, extra: str = "") -> str:
        now = time.time()
        el = max(now - self.t0, 1e-6)
        rate = self.done / el
        eta = ""
        if self.total > self.done and rate > 0:
            eta = f", ETA {(self.total - self.done) / rate / 3600:.1f}h"
        self._last = now
        return f"{self.done:,} {self.unit} in {el / 60:.1f}min ({rate:,.0f}/s){eta}"


def dump_json(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fo:
        json.dump(obj, fo, ensure_ascii=False, indent=1, default=str)


def human(n: float) -> str:
    for unit in ("", "K", "M", "G"):
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"

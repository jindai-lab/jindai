"""Embedding client utils

``EmbeddingClient`` is the single entry point for embeddings: it owns the
HTTP call to an OpenAI-compatible ``/embeddings`` endpoint (or dispatches
to the in-process ONNX backend when ``embedding_api_base`` is ``local``),
the client-side pacing, the per-request token splitting and the retries.
``jindai.models.TextEmbeddings`` only stores vectors and never sends
requests itself.

The gateway serving BAAI/bge-m3 (SiliconFlow via newapi) enforces
2000 RPM / 500k TPM / <=8k tokens per request (measured 2026-10-24).
This module paces outbound ``/embeddings`` requests so we never rely on
429s: every caller reserves its token budget BEFORE sending.

Token counting uses the local bge-m3 tokenizer (sentencepiece in the
HF cache) so the 8k per-request cap and the TPM window are exact, not
heuristic; a conservative char-count fallback applies if the tokenizer
cannot be loaded.

Usage (async, process-wide singletons):

    client = embedutils.get_client()
    vec = await client.encode_one("some text")
    vecs = await client.encode(["a", "b", "c"])

    limiter = embedutils.get_limiter()          # pacing only
    groups = limiter.split_by_tokens(texts, max_tokens)  # [(sub_texts, n_tokens)]
    for sub_texts, ntok in groups:
        await limiter.acquire(ntok)              # blocks until RPM+TPM allow
        ... POST ...

Safe for concurrent tasks (asyncio.Lock guards the window; the wait
happens outside the lock).

注：本模块与 ``ocrutils``（``PaddleOCRClient``）同一命名族、同一职责边界——
对外只暴露 client，``jindai.models`` 不再直接发 embedding 请求。
"""
from __future__ import annotations

import asyncio
import glob
import logging
import os
import time
from collections import deque
from typing import List, Optional, Tuple

import httpx

from .config import config

log = logging.getLogger(__name__)

_tokenizer = None
_tokenizer_tried = False


def _get_tokenizer():
    """Lazy-load the bge-m3 tokenizer from the local HF cache (Rust, fast)."""
    global _tokenizer, _tokenizer_tried
    if _tokenizer_tried:
        return _tokenizer
    _tokenizer_tried = True
    try:
        from tokenizers import Tokenizer
        hf_home = os.environ.get("HF_HOME") or str(
            os.path.expanduser("~/.cache/huggingface"))
        cands = glob.glob(os.path.join(
            hf_home, "hub", "models--BAAI--bge-m3", "snapshots", "*",
            "tokenizer.json"))
        if not cands:
            log.warning("bge-m3 tokenizer.json not found under %s", hf_home)
            return None
        _tokenizer = Tokenizer.from_file(cands[0])
        log.info("rate_limit: loaded bge-m3 tokenizer from %s", cands[0])
    except Exception as ex:  # noqa: BLE001 - fall back to heuristic
        log.warning("rate_limit: tokenizer load failed (%s: %s), "
                    "falling back to char-count estimate",
                    type(ex).__name__, ex)
        _tokenizer = None
    return _tokenizer


def count_tokens(texts: List[str]) -> List[int]:
    """Exact token count per text via the bge-m3 tokenizer.

    Fallback (tokenizer unavailable): len(text) chars, a conservative
    upper bound for XLM-R sentencepiece (pieces merge >= 1 char).
    """
    tok = _get_tokenizer()
    if tok is not None:
        return [max(1, len(e.ids)) for e in tok.encode_batch(texts)]
    return [max(1, len(t)) for t in texts]


class AsyncRateLimiter:
    """Sliding-window limiter over RPM and TPM simultaneously."""

    def __init__(self, rpm: int = 0, tpm: int = 0, window: float = 60.0):
        """Args:
            rpm: max requests per window (0 = unlimited).
            tpm: max tokens per window (0 = unlimited).
            window: window length in seconds.
        """
        self.rpm = rpm
        self.tpm = tpm
        self.window = window
        self._entries: deque = deque()   # (monotonic_ts, tokens)
        self._tokens = 0
        self._lock = asyncio.Lock()
        self.stats = {"requests": 0, "tokens": 0, "waits": 0, "wait_s": 0.0}

    def _trim(self, now: float) -> None:
        cutoff = now - self.window
        while self._entries and self._entries[0][0] <= cutoff:
            _, ntok = self._entries.popleft()
            self._tokens -= ntok

    async def acquire(self, n_tokens: int, deadline: float = 300.0) -> None:
        """Reserve n_tokens + one request slot; blocks until both fit.

        A single reservation larger than tpm is clamped to tpm (a ~2k-token
        paragraph can never exceed the 500k window, but the clamp keeps a
        misconfigured small tpm from deadlocking).
        """
        if self.rpm <= 0 and self.tpm <= 0:
            return
        if self.tpm > 0:
            n_tokens = min(n_tokens, self.tpm)
        t_start = time.monotonic()
        while True:
            async with self._lock:
                now = time.monotonic()
                self._trim(now)
                rpm_ok = self.rpm <= 0 or len(self._entries) < self.rpm
                tpm_ok = self.tpm <= 0 or self._tokens + n_tokens <= self.tpm
                if rpm_ok and tpm_ok:
                    self._entries.append((now, n_tokens))
                    self._tokens += n_tokens
                    waited = now - t_start
                    if waited > 0.5:
                        self.stats["waits"] += 1
                        self.stats["wait_s"] += waited
                    self.stats["requests"] += 1
                    self.stats["tokens"] += n_tokens
                    return
                # wait until the OLDEST entry expires (both limits share
                # the same window, so oldest-expiry makes progress on both)
                wait = self._entries[0][0] + self.window - now
            if time.monotonic() - t_start > deadline:
                raise TimeoutError(
                    f"rate limiter: could not reserve {n_tokens} tokens in "
                    f"{deadline:.0f}s (rpm={self.rpm} tpm={self.tpm})")
            await asyncio.sleep(max(wait, 0.05))

    def split_by_tokens(
        self, texts: List[str], max_tokens: int
    ) -> List[Tuple[List[str], int]]:
        """Greedily pack texts into sub-batches of <= max_tokens.

        Returns [(sub_texts, sub_token_count), ...]. A single text over
        max_tokens becomes its own group (paragraphs are capped at 2000
        chars, so this never trips the 8k gateway limit in practice).
        """
        if not texts:
            return []
        counts = count_tokens(texts)
        groups: List[Tuple[List[str], int]] = []
        cur: List[str] = []
        cur_n = 0
        for text, n in zip(texts, counts):
            if cur and cur_n + n > max_tokens:
                groups.append((cur, cur_n))
                cur, cur_n = [], 0
            cur.append(text)
            cur_n += n
        if cur:
            groups.append((cur, cur_n))
        return groups

    def snapshot(self) -> dict:
        now = time.monotonic()
        self._trim(now)
        return {
            "rpm_limit": self.rpm, "tpm_limit": self.tpm,
            "in_window_requests": len(self._entries),
            "in_window_tokens": self._tokens,
            **self.stats,
        }


_limiter: Optional[AsyncRateLimiter] = None


def get_limiter() -> AsyncRateLimiter:
    """Process-wide limiter built from config.yaml (read once per boot)."""
    global _limiter
    if _limiter is None:
        _limiter = AsyncRateLimiter(
            rpm=int(getattr(config, "embedding_rate_rpm", 2000) or 0),
            tpm=int(getattr(config, "embedding_rate_tpm", 500000) or 0),
        )
        log.info("embedding rate limiter: rpm=%s tpm=%s window=60s",
                 _limiter.rpm, _limiter.tpm)
    return _limiter


# ---------------------------------------------------------------------------
# Embedding client (single entry point: HTTP / local backend + pacing)
# ---------------------------------------------------------------------------

# Sentinel value: embedding_api_base == "local" selects the in-process backend.
LOCAL = "local"

_local_backend = None


def local_backend():
    """Return the in-process ONNX backend module (lazy, cached).

    The backend lives in the built-in embedding plugin
    (``plugins/embedding/local_backend.py``); importing it eagerly would
    pull the whole plugin package in at import time.
    """
    global _local_backend
    if _local_backend is None:
        from importlib import import_module
        _local_backend = import_module("plugins.embedding.local_backend")
    return _local_backend


def is_local(base: Optional[str]) -> bool:
    """True when the given embedding API base selects the local backend."""
    return (base or "").strip().lower().rstrip("/") == LOCAL


def _retry_after(response: httpx.Response, attempt: int) -> float:
    """Seconds to wait after a 429: Retry-After when present, else backoff."""
    try:
        return min(max(float(response.headers.get("Retry-After", "")), 1.0), 120.0)
    except ValueError:
        return min(2.0 ** attempt, 120.0)


class EmbeddingClient:
    """Client for the configured embedding backend.

    Remote mode POSTs to ``{base_url}/embeddings`` (OpenAI-compatible)
    with client-side pacing through :func:`get_limiter` and per-request
    token splitting; ``local`` mode runs the in-process ONNX backend and
    skips pacing (there is no gateway to protect).

    Attributes:
        base_url: Embedding API base URL, or the literal ``local``.
        api_key: Bearer token for the remote endpoint (may be empty).
        model: Model name sent in the request payload.
        timeout: Default per-request timeout in seconds.
        retries: Attempts per sub-request (429 / transport errors).
        max_request_tokens: Per-request input cap (gateway: 8k tokens).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        retries: int = 6,
    ) -> None:
        """Initialize the client from config unless overridden."""
        self.base_url = (base_url or config.embedding_api_base or "").strip()
        self.local = is_local(self.base_url)
        self.base_url = self.base_url.rstrip("/")
        self.api_key = config.embedding_api_key if api_key is None else api_key
        self.model = model or config.embedding_model
        self.timeout = timeout
        self.retries = retries
        self.max_request_tokens = int(
            getattr(config, "embedding_max_request_tokens", 8000) or 8000)
        # One AsyncClient per client instance: the polling loop and the API
        # routes reuse connections instead of paying a TCP/TLS handshake per
        # request. Created lazily so constructing a client stays cheap.
        self._client: Optional[httpx.AsyncClient] = None
        self._dims: Optional[int] = None

    # -- transport ---------------------------------------------------------
    def _http(self) -> httpx.AsyncClient:
        """Return the shared AsyncClient, recreating it if it was closed."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient()
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _check_dims(self, vector: List[float]) -> None:
        """Warn once if the endpoint returns vectors of an unexpected size."""
        if self._dims is None:
            self._dims = len(vector)
            expected = getattr(config, "embedding_dims", 0) or 0
            if expected and expected != self._dims:
                log.warning(
                    "embedding API returned %d dims, config.embedding_dims "
                    "is %d", self._dims, expected)

    async def _post(self, sub_texts: List[str], n_tokens: int,
                    timeout: float) -> List[List[float]]:
        """POST one token-bounded sub-batch, pacing + retrying as needed."""
        limiter = get_limiter()
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": sub_texts}
        headers = self._headers()
        for attempt in range(self.retries):
            await limiter.acquire(n_tokens)
            try:
                response = await self._http().post(
                    url, headers=headers, json=payload, timeout=timeout)
                if response.status_code == 429:
                    # Gateway still throttled (another consumer, or the
                    # window closed mid-flight): honor Retry-After when
                    # present, else exponential backoff.
                    wait = _retry_after(response, attempt)
                    log.warning(
                        "embedding 429 rate-limited (ntok=%d), waiting "
                        "%.1fs (attempt %d/%d)", n_tokens, wait,
                        attempt + 1, self.retries)
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                vectors = [item["embedding"] for item in response.json()["data"]]
            except (httpx.HTTPError, asyncio.TimeoutError) as ex:
                wait = min(2.0 ** attempt, 60.0)
                log.warning(
                    "embedding call failed (%s: %s), retry %d/%d in %.0fs",
                    type(ex).__name__, ex, attempt + 1, self.retries, wait)
                await asyncio.sleep(wait)
                continue
            if len(vectors) != len(sub_texts):
                raise RuntimeError(
                    f"embedding API returned {len(vectors)}/"
                    f"{len(sub_texts)} vectors")
            self._check_dims(vectors[0])
            return vectors
        raise RuntimeError(
            f"embedding request failed after {self.retries} attempts "
            f"(ntok={n_tokens}, texts={len(sub_texts)})")

    # -- public API --------------------------------------------------------
    async def encode(self, texts: List[str],
                     timeout: Optional[float] = None) -> List[List[float]]:
        """Embed a batch of texts; one vector per input, in order.

        Args:
            texts: Texts to embed (stripped before sending).
            timeout: Per-request timeout override (default ``self.timeout``).

        Returns:
            List of embeddings as lists of floats.
        """
        if not texts:
            return []
        stripped = [text.strip() for text in texts]
        if self.local:
            return await local_backend().encode(stripped)
        limiter = get_limiter()
        groups = limiter.split_by_tokens(stripped, self.max_request_tokens)
        vectors: List[List[float]] = []
        for sub_texts, n_tokens in groups:
            vectors.extend(await self._post(
                sub_texts, n_tokens, timeout or self.timeout))
        return vectors

    async def encode_one(self, text: str,
                         timeout: Optional[float] = None) -> List[float]:
        """Embed a single text (query path)."""
        return (await self.encode([text], timeout=timeout))[0]

    def _clone(self) -> "EmbeddingClient":
        """Copy of this client sharing the connection settings, not the pool."""
        clone = EmbeddingClient(
            base_url=self.base_url, api_key=self.api_key, model=self.model,
            timeout=self.timeout, retries=self.retries)
        clone.max_request_tokens = self.max_request_tokens
        return clone

    def encode_one_sync(self, text: str) -> List[float]:
        """Blocking single-text embedding for sync pipeline stages.

        ``asyncio.run`` spins up a throw-away event loop, so the pooled
        AsyncClient of this instance (bound to the loop it was first used
        on) must not be reused: the call runs on a short-lived clone.
        """
        clone = self._clone()

        async def _run() -> List[float]:
            try:
                return await clone.encode_one(text)
            finally:
                await clone.aclose()

        return asyncio.run(_run())


_client: Optional[EmbeddingClient] = None


def get_client() -> EmbeddingClient:
    """Process-wide embedding client built from config (once per boot)."""
    global _client
    if _client is None:
        _client = EmbeddingClient()
        log.info("embedding client: %s (model=%s)",
                 "local ONNX backend" if _client.local
                 else _client.base_url + "/embeddings", _client.model)
    return _client


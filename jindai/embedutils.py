"""Embedding 网关客户端侧工具：token 计数 + RPM/TPM 滑窗节流（2026-10-24；
2026-10-29 由 ``rate_limit.py`` 更名，与 ocrutils / pdfutils 同一命名族）。

The gateway serving BAAI/bge-m3 (SiliconFlow via newapi) enforces
2000 RPM / 500k TPM / <=8k tokens per request (measured 2026-10-24).
This module paces outbound /embeddings requests so we never rely on
429s: every caller reserves its token budget BEFORE sending.

Token counting uses the local bge-m3 tokenizer (sentencepiece in the
HF cache) so the 8k per-request cap and the TPM window are exact, not
heuristic; a conservative char-count fallback applies if the tokenizer
cannot be loaded.

Usage (async, process-wide singleton):

    limiter = embedutils.get_limiter()
    groups = limiter.split_by_tokens(texts, max_tokens)  # [(sub_texts, n_tokens)]
    for sub_texts, ntok in groups:
        await limiter.acquire(ntok)              # blocks until RPM+TPM allow
        ... POST ...

Safe for concurrent tasks (asyncio.Lock guards the window; the wait
happens outside the lock).

注：本模块只管"节奏与切批"，HTTP 调用仍在 ``jindai.models.TextEmbeddings``
（encode_one / encode_batch）；若日后把 embedding HTTP 客户端也收进来，请一并在
此文件暴露 ``EmbeddingClient``，保持 "ocrutils:PaddleOCRClient" 对称。
"""
from __future__ import annotations

import asyncio
import glob
import os
import time
from collections import deque
from typing import List, Optional, Tuple

import logging

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
        from jindai.config import config
        _limiter = AsyncRateLimiter(
            rpm=int(getattr(config, "embedding_rate_rpm", 2000) or 0),
            tpm=int(getattr(config, "embedding_rate_tpm", 500000) or 0),
        )
        log.info("embedding rate limiter: rpm=%s tpm=%s window=60s",
                 _limiter.rpm, _limiter.tpm)
    return _limiter

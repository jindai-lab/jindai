"""PaddleOCR remote service client for Jindai application.

This module provides a client for the remote PaddleOCR HTTP service
that runs on other machines. It is used as an OCR fallback when a PDF
page contains no extractable text (e.g. scanned documents).

Service protocol (matching ocrmypdf-paddleocr-remote plugin):
    - Endpoint: {base_url}/ocr
    - Request: JSON POST with base64-encoded JPEG image (fileType=1)
      or base64-encoded PDF bytes (fileType=0)
    - Response: result.ocrResults[] -- ONE entry per page for PDF input,
      entry i = page i; prunedResult.rec_texts holds the line texts

PDF transport (preferred, 2026-10-16):
    - `ocr_pdf()` ships raw page bytes instead of a client-side render.
      The service rasterizes with pypdfium2 at scale=2.0 -- the same zoom
      `render_pdf_page` used -- so quality is equal or better (the old
      client path also passed images through `convert('1')`, whose
      dithering erased thin strokes on woodblock/line-art scans).
    - Page copies are ~10x smaller than the rendered JPEG payload and
      cost ~0.005s to build vs ~0.6s/page of client rendering.
    - The service caps input at `max_num_input_imgs` (default 10 pages,
      see OCR.yaml `Serving.extra`) -- callers MUST chunk accordingly
      and index results by position.

Multi-replica support:
    - config.paddle_remotes (list of base URLs) -> requests round-robin
      across replicas so page-level parallel OCR fans out over GPUs.
    - Fallback: base_url arg, then config.paddle_remote (single endpoint).
"""

from __future__ import annotations

import base64
import itertools
import logging
from io import BytesIO
from typing import List, Optional

import httpx
from PIL import Image

from .config import config

log = logging.getLogger(__name__)

# Round-robin counter is PROCESS-GLOBAL on purpose: the datasource builds one
# PaddleOCRClient per file, and each PDF batch is only 1-2 requests, so a
# per-instance counter would leave every file hitting urls[0] (measured:
# 6/6 requests to 8080 with a second replica idle). Shared counter => the
# rotation continues across per-file clients. Modulo handles 1..N replicas.
_RR = itertools.count()


class PaddleOCRClient:
    """Client for the remote PaddleOCR HTTP service.

    Attributes:
        base_url: First configured base URL (kept for compatibility).
        timeout: Request timeout in seconds.
    """

    # Map Jindai/OCRmyPDF language codes to PaddleOCR language codes.
    LANGUAGE_MAP = {
        'en': 'en',
        'zh': 'ch',
        'chs': 'ch',
        'zhs': 'ch',
        'chi_sim': 'ch',
        'zht': 'chinese_cht',
        'cht': 'chinese_cht',
        'chi_tra': 'chinese_cht',
        'fr': 'fr',
        'de': 'german',
        'ja': 'japan',
        'jpn': 'japan',
        'ko': 'korean',
        'kor': 'korean',
        'es': 'spanish',
        'spa': 'spanish',
        'ru': 'ru',
        'rus': 'ru',
        'ar': 'ar',
        'ara': 'ar',
        'hi': 'hi',
        'hin': 'hi',
        'pt': 'pt',
        'por': 'pt',
        'it': 'it',
        'ita': 'it',
        'tr': 'tr',
        'tur': 'tr',
        'vi': 'vi',
        'vie': 'vi',
        'th': 'th',
        'tha': 'th',
    }

    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0) -> None:
        """Initialize the PaddleOCR client.

        Args:
            base_url: Base URL of the service. Overrides replica list.
            timeout: Request timeout in seconds.
        """
        urls = None
        if base_url is None:
            try:
                urls = getattr(config, 'paddle_remotes', None)
            except Exception:
                urls = None
        if not urls:
            urls = [base_url or config.paddle_remote]
        self._urls = [u.rstrip('/') + '/' for u in urls]
        self._rr = _RR  # shared across instances -- see module comment
        self.base_url = self._urls[0]
        self.timeout = timeout

    def _next_url(self) -> str:
        """Round-robin next replica base URL (thread-safe: itertools.count)."""
        return self._urls[next(self._rr) % len(self._urls)]

    def _prepare_image(self, image_bytes: bytes) -> str:
        """Preprocess image and return base64-encoded JPEG.

        Kept RGB (NOT 1-bit, 2026-10-16): the old ``convert('1')`` dithering
        erased thin strokes -- on a woodblock scan it returned 2 lines where
        RGB returned 16 and raw PDF transport 22 -- while *inflating* the
        JPEG (3.2MiB vs 474KiB: dither noise defeats compression). Output is
        still capped at 3000px so payloads stay bounded.

        Args:
            image_bytes: Raw image data (PNG/JPEG/etc.).

        Returns:
            Base64-encoded JPEG string.
        """
        buf = BytesIO()
        im = Image.open(BytesIO(image_bytes))
        im = im.convert('RGB')
        im.thumbnail((3000, 3000))
        im.save(buf, format='jpeg', quality=85)
        return base64.b64encode(buf.getvalue()).decode('ascii')

    @staticmethod
    def _join_texts(entry: dict) -> str:
        """One ocrResults entry -> newline-joined non-empty line texts."""
        texts = entry.get('prunedResult', {}).get('rec_texts', [])
        return '\n'.join(t for t in texts if t)

    @staticmethod
    def _pdf_payload(pdf_bytes: bytes) -> dict:
        """JSON payload shipping raw PDF bytes (fileType=0).

        No image preprocessing: the service decodes the PDF with pypdfium2
        itself (scale=2.0, same zoom the client-side render used).
        """
        return {
            "file": base64.b64encode(pdf_bytes).decode("ascii"),
            "fileType": 0,
            "returnWordBox": False,
            "visualize": False,
        }

    def _post(self, payload: dict) -> list:
        """POST to /ocr with failover across replicas; return ocrResults list.

        A dead/busy endpoint must not drop the page (an uncaught ConnectError
        used to empty the OCR result and the scanned page was silently
        skipped -> paragraph gap).
        """
        last_exc: Optional[Exception] = None
        for _ in range(len(self._urls)):
            url = self._next_url()
            try:
                response = httpx.post(url + 'ocr', json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json().get('result', {}).get('ocrResults', [])
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = exc
                log.warning("OCR endpoint %s failed (%s: %s); trying next replica",
                            url, type(exc).__name__, exc)
                continue
            return result
        raise last_exc if last_exc else RuntimeError("no OCR endpoint configured")

    def ocr_image(self, image_bytes: bytes, lang: str = 'en') -> str:
        """Recognize text in an image using the remote PaddleOCR service.

        Args:
            image_bytes: Raw image data.
            lang: Language code (Jindai/OCRmyPDF style, e.g. 'zh', 'en', 'de').

        Returns:
            Recognized text joined by newlines, or empty string on failure.

        Raises:
            httpx.HTTPError: If the service request fails.
        """
        paddle_lang = self.LANGUAGE_MAP.get(lang, lang)
        payload = {
            "file": self._prepare_image(image_bytes),
            "fileType": 1,
            "returnWordBox": False,
            "visualize": False,
            "lang": paddle_lang,
        }
        result = self._post(payload)
        if not result:
            return ""
        return self._join_texts(result[0])

    def ocr_pdf(self, pdf_bytes: bytes, lang: str = 'en') -> List[str]:
        """Recognize text in PDF pages sent as raw bytes (fileType=0).

        The service rasterizes the PDF itself (pypdfium2, scale=2.0) and
        returns one ``ocrResults`` entry per page, in page order -- so the
        returned list length must be checked against the pages sent: the
        service silently truncates at ``max_num_input_imgs`` (default 10).

        Args:
            pdf_bytes: Raw PDF bytes (a chunk of pages from
                ``jindai.pdfutils.extract_pages_pdf``).
            lang: Language code; kept for protocol symmetry only -- the
                PaddleX InferRequest schema has no ``lang`` field and
                ignores it (PP-OCRv5 models are multilingual).

        Returns:
            One newline-joined text string per page, in input page order.
            May be shorter than the pages sent (service cap) or empty.

        Raises:
            httpx.HTTPError: If every replica failed.
        """
        paddle_lang = self.LANGUAGE_MAP.get(lang, lang)
        payload = self._pdf_payload(pdf_bytes)
        payload["lang"] = paddle_lang
        results = self._post(payload)
        return [self._join_texts(r) for r in results]

    async def aocr_image(self, image_bytes: bytes, lang: str = 'en') -> str:
        """Asynchronously recognize text in an image.

        Args:
            image_bytes: Raw image data.
            lang: Language code (Jindai/OCRmyPDF style).

        Returns:
            Recognized text joined by newlines, or empty string on failure.

        Raises:
            httpx.HTTPError: If the service request fails.
        """
        paddle_lang = self.LANGUAGE_MAP.get(lang, lang)
        payload = {
            "file": self._prepare_image(image_bytes),
            "fileType": 1,
            "returnWordBox": False,
            "visualize": False,
            "lang": paddle_lang,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            last_exc: Optional[Exception] = None
            for _ in range(len(self._urls)):
                url = self._next_url()
                try:
                    response = await client.post(url + 'ocr', json=payload)
                    response.raise_for_status()
                    result = response.json().get('result', {}).get('ocrResults', [])
                except (httpx.HTTPError, ValueError) as exc:
                    last_exc = exc
                    log.warning("OCR endpoint %s failed (%s: %s); trying next replica",
                                url, type(exc).__name__, exc)
                    continue
                if not result:
                    return ""
                texts = result[0].get('prunedResult', {}).get('rec_texts', [])
                return '\n'.join(t for t in texts if t)
            raise last_exc if last_exc else RuntimeError("no OCR endpoint configured")

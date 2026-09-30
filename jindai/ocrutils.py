"""PaddleOCR remote service client for Jindai application.

This module provides a client for the remote PaddleOCR HTTP service
that runs on other machines. It is used as an OCR fallback when a PDF
page contains no extractable text (e.g. scanned documents).

Service protocol (matching ocrmypdf-paddleocr-remote plugin):
    - Endpoint: {base_url}/ocr
    - Request: JSON POST with base64-encoded JPEG image
    - Response: result.ocrResults[0].prunedResult.rec_texts (list of texts)

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
from typing import Optional

import httpx
from PIL import Image

from .config import config

log = logging.getLogger(__name__)


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
        self._rr = itertools.count()
        self.base_url = self._urls[0]
        self.timeout = timeout

    def _next_url(self) -> str:
        """Round-robin next replica base URL (thread-safe: itertools.count)."""
        return self._urls[next(self._rr) % len(self._urls)]

    def _prepare_image(self, image_bytes: bytes) -> str:
        """Preprocess image and return base64-encoded JPEG.

        Converts to 1-bit black & white, scales down to max 3000px,
        and encodes as JPEG (matching the ocrmypdf plugin behavior).

        Args:
            image_bytes: Raw image data (PNG/JPEG/etc.).

        Returns:
            Base64-encoded JPEG string.
        """
        buf = BytesIO()
        im = Image.open(BytesIO(image_bytes))
        im = im.convert('1')
        im.thumbnail((3000, 3000))
        im.save(buf, format='jpeg')
        return base64.b64encode(buf.getvalue()).decode('ascii')

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
        # Fail over across replicas: a dead/busy endpoint must not drop the
        # page (an uncaught ConnectError used to empty the OCR result and the
        # scanned page was silently skipped -> paragraph gap).
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
            if not result:
                return ""
            texts = result[0].get('prunedResult', {}).get('rec_texts', [])
            return '\n'.join(t for t in texts if t)
        raise last_exc if last_exc else RuntimeError("no OCR endpoint configured")

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

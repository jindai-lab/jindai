"""PaddleOCR remote service client for Jindai application.

This module provides a client for the remote PaddleOCR HTTP service
that runs on other machines. It is used as an OCR fallback when a PDF
page contains no extractable text (e.g. scanned documents).

Service protocol (matching ocrmypdf-paddleocr-remote plugin):
    - Endpoint: {base_url}/ocr
    - Request: JSON POST with base64-encoded JPEG image
    - Response: result.ocrResults[0].prunedResult.rec_texts (list of texts)
"""

from __future__ import annotations

import base64
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
        base_url: Base URL of the PaddleOCR remote service.
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
            base_url: Base URL of the service. Defaults to config.paddle_remote.
            timeout: Request timeout in seconds.
        """
        self.base_url = (base_url or config.paddle_remote).rstrip('/') + '/'
        self.timeout = timeout

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
        response = httpx.post(
            self.base_url + 'ocr',
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json().get('result', {}).get('ocrResults', [])
        if not result:
            return ""
        texts = result[0].get('prunedResult', {}).get('rec_texts', [])
        return '\n'.join(t for t in texts if t)

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
            response = await client.post(self.base_url + 'ocr', json=payload)
            response.raise_for_status()
            result = response.json().get('result', {}).get('ocrResults', [])
        if not result:
            return ""
        texts = result[0].get('prunedResult', {}).get('rec_texts', [])
        return '\n'.join(t for t in texts if t)
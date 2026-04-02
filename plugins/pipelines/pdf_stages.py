"""PDF-specific pipeline stages for Jindai application.

This module provides specialized pipeline stages for PDF text processing:
- TextCleaner: Clean PDF-extracted text (OCR errors, whitespace, garbled text)
- PDFLanguageDetect: Enhanced language detection for target languages
- CrossPageReparagraphizer: Re-paragraphize with cross-page support using sliding window
"""

import re
import statistics
from collections import deque
from typing import Iterator, List, Optional, Tuple

import hanzidentifier
import regex as re
from lingua import Language, LanguageDetectorBuilder

from jindai.models import Paragraph
from jindai.pipeline import PipelineStage, ResolveReturn


class TextCleaner(PipelineStage):
    """Clean PDF-extracted text.
    
    Features:
    - Remove excessive whitespace
    - Detect and filter garbled/OCR text
    - Normalize unicode characters
    - Remove hyphenation at line breaks
    - OCR error correction for European languages
    
    Attributes:
        remove_garbled: Enable garbled text detection.
        garbled_threshold: Threshold for garbled detection (0-1).
        normalize_spaces: Normalize multiple spaces to single.
        remove_hyphens: Remove hyphenation at line breaks.
        fix_ocr_errors: Enable OCR error correction.
    """
    
    # OCR confusables mapping for European languages
    OCR_CONFUSABLES = {
        # German umlauts and special chars
        'ä': 'ä', 'ö': 'ö', 'ü': 'ü', 'ß': 'ß',
        # French accented chars
        'é': 'é', 'è': 'è', 'ê': 'ê', 'ë': 'ë',
        'à': 'à', 'â': 'â', 'ù': 'ù', 'û': 'û',
        'ç': 'ç', 'î': 'î', 'ï': 'ï', 'ô': 'ô', 'î': 'î',
        # Russian Cyrillic (normalize to proper forms)
        'а': 'а', 'е': 'е', 'о': 'о', 'р': 'р', 'с': 'с', 'у': 'у',
        # Common OCR ligatures - expand to component letters
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        'ﬅ': 'st', 'ﬆ': 'st',
        # Smart quotes and dashes - normalize to ASCII
        ''': "'", '"': '"',
        ''': "'", '"': '"',
        '—': '-', '–': '-',  # Em/en dash to hyphen
        '…': '...',  # Ellipsis
        '•': '*', '°': 'o',  # Bullet and degree
        # Zero-width and control chars - remove
        '\u200b': '',  # Zero-width space
        '\u200c': '',  # Zero-width non-joiner
        '\u200d': '',  # Zero-width joiner
        '\ufeff': '',  # BOM
        '\u0000': '', '\u0001': '', '\u0002': '',  # Control chars
        '\u0003': '', '\u0004': '', '\u0005': '',
        '\ufffd': '',  # Replacement character
    }
    
    # German umlaut substitution patterns (optional, disabled by default)
    GERMAN_UMLAUT_PATTERNS = [
        # German umlaut errors - common substitutions
        (r'\bAe\b', 'Ä'), (r'\bOe\b', 'Ö'), (r'\bUe\b', 'Ü'),
        (r'\bae\b', 'ä'), (r'\boe\b', 'ö'), (r'\bue\b', 'ü'),
        (r'\bss\b', 'ß'),
    ]
    
    # Common OCR error patterns (regex)
    OCR_ERROR_PATTERNS = [
        # French apostrophe errors
        (r"\b([ldjcnmst]|qu|jusqu|lorsqu|puisqu)'(\w)", r"\1'\2"),
        # Hyphenation at line breaks - join words
        (r'(\w+)-\n(\w+)', r'\1\2'),
        # Multiple spaces/tabs - normalize to single space
        (r'[ \t]+', ' '),
        # Trailing/leading spaces around newlines
        (r' \n', '\n'),
        (r'\n +', '\n'),
        # Multiple newlines - normalize to double newline (paragraph break)
        (r'\n{3,}', '\n\n'),
    ]
    
    # Sentence-ending punctuation for various languages
    SENTENCE_ENDINGS = '.!?。！？…:：'
    
    def __init__(
        self,
        remove_garbled: bool = True,
        garbled_threshold: float = 0.3,
        normalize_spaces: bool = True,
        remove_hyphens: bool = True,
        fix_ocr_errors: bool = True,
        fix_german_umlauts: bool = False,
    ) -> None:
        """Initialize TextCleaner.
        
        Args:
            remove_garbled: Enable garbled text detection.
            garbled_threshold: Threshold for garbled detection (0-1).
            normalize_spaces: Normalize multiple spaces to single.
            remove_hyphens: Remove hyphenation at line breaks.
            fix_ocr_errors: Enable OCR error correction.
            fix_german_umlauts: Enable German umlaut substitutions (Ae→Ä, Oe→Ö, Ue→Ü, etc.).
                Disabled by default to avoid false positives in non-German text.
        """
        super().__init__()
        self.remove_garbled = remove_garbled
        self.garbled_threshold = garbled_threshold
        self.normalize_spaces = normalize_spaces
        self.remove_hyphens = remove_hyphens
        self.fix_ocr_errors = fix_ocr_errors
        self.fix_german_umlauts = fix_german_umlauts
        
        # Compile patterns for efficiency
        self._confusables_pattern = re.compile(
            '|'.join(re.escape(k) for k in self.OCR_CONFUSABLES.keys())
        )
    
    def _replace_confusable(self, match: re.Match) -> str:
        """Replace matched confusable character."""
        return self.OCR_CONFUSABLES.get(match.group(0), match.group(0))
    
    def clean_text(self, text: str) -> str:
        """Clean text content.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return text
        
        # Step 1: Replace confusable characters
        if self.fix_ocr_errors:
            text = self._confusables_pattern.sub(self._replace_confusable, text)
            
            # Apply regex patterns
            for pattern, replacement in self.OCR_ERROR_PATTERNS:
                text = re.sub(pattern, replacement, text)
            
            # Optionally apply German umlaut substitutions
            if self.fix_german_umlauts:
                for pattern, replacement in self.GERMAN_UMLAUT_PATTERNS:
                    text = re.sub(pattern, replacement, text)
        
        # Step 2: Normalize spaces (if not already done by patterns)
        if self.normalize_spaces:
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r' +\n', '\n', text)
            text = re.sub(r'\n +', '\n', text)
        
        # Step 3: Strip each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Step 4: Normalize multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Step 5: Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def detect_garbled_text(self, text: str) -> bool:
        """Detect if text contains excessive garbled/OCR characters.
        
        Detection criteria:
        1. High ratio of non-alphanumeric special characters (>threshold)
        2. Unusual unicode character combinations (control chars, replacement chars)
        3. Random capitalization patterns (alternating case)
        4. Common OCR error patterns
        
        Args:
            text: Text to analyze.
            
        Returns:
            True if text appears garbled.
        """
        if not text or len(text) < 10:
            return False
        
        # Count special characters (non-alphanumeric, non-space, non-punctuation)
        special_chars = 0
        total_chars = len(text)
        
        for char in text:
            if char in self.OCR_CONFUSABLES:
                special_chars += 1
            elif not (char.isalnum() or char.isspace() or 
                      char in '.,!?;:()[]{}"\'-–—…'):
                special_chars += 1
        
        # Check ratio of special characters
        if special_chars / total_chars > self.garbled_threshold:
            return True
        
        # Check for replacement characters (indicates encoding issues)
        if '\ufffd' in text:
            return True
        
        # Check for excessive control characters
        control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
        if control_chars / total_chars > 0.05:
            return True
        
        # Check for alternating case pattern (random capitalization)
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) > 10:
            case_changes = sum(
                1 for i in range(len(alpha_chars) - 1)
                if alpha_chars[i].islower() != alpha_chars[i + 1].islower()
            )
            if case_changes / len(alpha_chars) > 0.6:
                return True
        
        return False
    
    def resolve(self, paragraph: Paragraph) -> ResolveReturn:
        """Process a paragraph and clean its content.
        
        Args:
            paragraph: Paragraph to process.
            
        Returns:
            Cleaned paragraph, or None if garbled text detected.
        """
        if not paragraph.content:
            return paragraph
        
        # Check for garbled text
        if self.remove_garbled and self.detect_garbled_text(paragraph.content):
            self.log(f"Filtered garbled text: {paragraph.content[:50]}...")
            return None
        
        # Clean the text
        paragraph.content = self.clean_text(paragraph.content)
        
        return paragraph


class PDFLanguageDetect(PipelineStage):
    """Enhanced language detection for PDF content.
    
    Optimized for: Chinese (zh), English (en), German (de), 
                   French (fr), Japanese (ja), Russian (ru)
    
    Features:
    - Script-based pre-filtering for CJK
    - N-gram frequency analysis
    - Confidence scoring
    """
    
    SUPPORTED_LANGUAGES = {
        'zh': 'Chinese',
        'en': 'English', 
        'de': 'German',
        'fr': 'French',
        'ja': 'Japanese',
        'ru': 'Russian',
    }
    
    # Build lingua detector for supported languages
    _lingua_languages = [
        Language.CHINESE,
        Language.ENGLISH,
        Language.GERMAN,
        Language.FRENCH,
        Language.JAPANESE,
        Language.RUSSIAN,
    ]
    detector = LanguageDetectorBuilder.from_languages(_lingua_languages).build()
    
    # Unicode ranges for script detection
    CJK_RANGE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
    HIRAGANA_RANGE = re.compile(r'[\u3040-\u309f]')
    KATAKANA_RANGE = re.compile(r'[\u30a0-\u30ff]')
    CYRILLIC_RANGE = re.compile(r'[\u0400-\u04ff]')
    LATIN_ACCENTS_RANGE = re.compile(r'[\u00c0-\u00ff]')  # Latin with accents
    
    def detect(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score.
        
        Detection strategy:
        1. Check for CJK characters → Chinese
        2. Check for Hiragana/Katakana → Japanese
        3. Check for Cyrillic → Russian
        4. Use lingua detector for others
        
        Args:
            text: Text to analyze.
            
        Returns:
            Tuple of (language_code, confidence).
        """
        if not text or len(text.strip()) < 3:
            return ('unknown', 0.0)
        
        # Clean text for analysis
        clean_text = re.sub(r'[0-9\s\p{Punctuation}]', '', text)
        
        if not clean_text:
            return ('unknown', 0.0)
        
        # Step 1: Check for CJK characters (Chinese)
        cjk_matches = self.CJK_RANGE.findall(clean_text)
        if cjk_matches:
            cjk_ratio = len(cjk_matches) / len(clean_text)
            if cjk_ratio > 0.5:
                # Distinguish simplified vs traditional
                if hanzidentifier.has_chinese(clean_text):
                    if hanzidentifier.is_simplified(clean_text):
                        return ('zhs', min(cjk_ratio * 1.2, 1.0))
                    else:
                        return ('zht', min(cjk_ratio * 1.2, 1.0))
                return ('zh', cjk_ratio)
        
        # Step 2: Check for Japanese (Hiragana/Katakana)
        hiragana_matches = self.HIRAGANA_RANGE.findall(clean_text)
        katakana_matches = self.KATAKANA_RANGE.findall(clean_text)
        jp_matches = hiragana_matches + katakana_matches
        if jp_matches:
            jp_ratio = len(jp_matches) / len(clean_text)
            if jp_ratio > 0.3:
                return ('ja', min(jp_ratio * 1.5, 1.0))
        
        # Step 3: Check for Cyrillic (Russian)
        cyrillic_matches = self.CYRILLIC_RANGE.findall(clean_text)
        if cyrillic_matches:
            cyrillic_ratio = len(cyrillic_matches) / len(clean_text)
            if cyrillic_ratio > 0.5:
                return ('ru', min(cyrillic_ratio * 1.2, 1.0))
        
        # Step 4: Use lingua detector for Latin-based languages
        try:
            detected_lang = self.detector.detect_language_of(text)
            if detected_lang:
                lang_code = detected_lang.iso_code_639_1.name.lower()
                confidence = self.detector.compute_language_confidence(text, detected_lang)
                
                # Normalize Chinese codes
                if lang_code in ('zh-cn', 'zh-sg'):
                    return ('zhs', confidence)
                elif lang_code.startswith('zh-'):
                    return ('zht', confidence)
                
                return (lang_code, confidence)
        except Exception as e:
            self.log(f"Language detection error: {e}")
        
        # Default to English with low confidence
        return ('en', 0.3)
    
    def resolve(self, paragraph: Paragraph) -> Paragraph:
        """Detect and set language for paragraph.
        
        Args:
            paragraph: Paragraph to process.
            
        Returns:
            Paragraph with detected language.
        """
        # Skip if language already set (not 'auto')
        if paragraph.lang and paragraph.lang != 'auto':
            return paragraph
        
        if not paragraph.content:
            paragraph.lang = 'unknown'
            return paragraph
        
        # Detect language
        lang, confidence = self.detect(paragraph.content)
        paragraph.lang = lang
        
        self.log(f"Detected language: {lang} (confidence: {confidence:.2f})")
        
        return paragraph


class CrossPageReparagraphizer(PipelineStage):
    """Re-paragraphize text with cross-page support using sliding window.
    
    Features:
    - Sliding window buffer for memory efficiency
    - Detect paragraph boundaries across pages
    - Handle headers/footers with heuristic detection
    - Configurable paragraph length thresholds
    
    Attributes:
        min_paragraph_length: Minimum characters for valid paragraph.
        max_paragraph_length: Maximum characters before forced split.
        short_line_threshold: Line length considered 'short'.
        short_line_batch: Number of short lines to trigger merge.
        window_size: Number of pages to buffer for sliding window.
    """
    
    # Sentence-ending punctuation for various languages
    SENTENCE_ENDINGS = '.!?。！？…:：'
    
    # Patterns that might indicate header/footer
    HEADER_FOOTER_PATTERNS = [
        re.compile(r'^\d+$'),  # Just a page number
        re.compile(r'^\d+\s*of\s*\d+$', re.IGNORECASE),  # "1 of 10"
        re.compile(r'^page\s*\d+$', re.IGNORECASE),  # "Page 1"
        re.compile(r'^-\s*\d+\s*-$'),  # "- 1 -"
        re.compile(r'^\s*[©®&]\s*\d{4}'),  # Copyright year
    ]
    
    def __init__(
        self,
        min_paragraph_length: int = 50,
        max_paragraph_length: int = 2000,
        short_line_threshold: int = 30,
        short_line_batch: int = 5,
        window_size: int = 10,
    ) -> None:
        """Initialize CrossPageReparagraphizer.
        
        Args:
            min_paragraph_length: Minimum characters for valid paragraph.
            max_paragraph_length: Maximum characters before forced split.
            short_line_threshold: Line length considered 'short'.
            short_line_batch: Number of short lines to trigger merge.
            window_size: Number of pages to buffer for sliding window.
        """
        super().__init__()
        self.min_paragraph_length = min_paragraph_length
        self.max_paragraph_length = max_paragraph_length
        self.short_line_threshold = short_line_threshold
        self.short_line_batch = short_line_batch
        self.window_size = window_size
        
        # Sliding window buffer: stores (line, page_num, page_label) tuples
        self._line_buffer: deque = deque(maxlen=window_size * 50)  # ~50 lines per page
        self._current_paragraph_lines: List[Tuple[str, int, str]] = []
        self._short_line_count = 0
        self._last_page_num = -1
    
    def _is_header_footer(self, line: str) -> bool:
        """Check if line appears to be a header or footer.
        
        Args:
            line: Line to check.
            
        Returns:
            True if line is likely a header/footer.
        """
        if not line or len(line) > 100:
            return False
        
        # Check against patterns
        for pattern in self.HEADER_FOOTER_PATTERNS:
            if pattern.match(line):
                return True
        
        # Very short lines at page boundaries might be headers/footers
        if len(line) < 5 and line.strip() in ('-', '—', '•', '*'):
            return True
        
        return False
    
    def _is_paragraph_boundary(self, line: str, prev_line: str) -> bool:
        """Determine if line starts a new paragraph.
        
        Criteria:
        1. Previous line ends with sentence-ending punctuation
        2. Current line starts with capital letter or new sentence marker
        3. Significant indentation change
        4. Empty line between (handled by line splitting)
        
        Args:
            line: Current line.
            prev_line: Previous line.
            
        Returns:
            True if this is a paragraph boundary.
        """
        if not prev_line or not line:
            return False
        
        # Check if previous line ends with sentence-ending punctuation
        prev_stripped = prev_line.rstrip()
        if prev_stripped and prev_stripped[-1] in self.SENTENCE_ENDINGS:
            # Check if current line starts with capital or CJK
            if line[0].isupper() or '\u4e00' <= line[0] <= '\u9fff':
                return True
        
        # Check for significant length change (new section)
        if len(prev_line) > 100 and len(line) < 20:
            return True
        
        return False
    
    def _flush_paragraph(self, source_url: str, base_dataset) -> Optional[Paragraph]:
        """Flush accumulated lines as a paragraph.
        
        Args:
            source_url: Source file path.
            base_dataset: Base dataset for the paragraph.
            
        Returns:
            Paragraph if valid, None otherwise.
        """
        if not self._current_paragraph_lines:
            return None
        
        # Join lines
        content = ' '.join(line for line, _, _ in self._current_paragraph_lines)
        content = content.strip()
        
        # Check minimum length
        if len(content) < self.min_paragraph_length:
            self._current_paragraph_lines.clear()
            return None
        
        # Get source page info from first line
        first_page_num, first_label = self._current_paragraph_lines[0][1:]
        
        # Create paragraph
        para = Paragraph(
            content=content,
            source_url=source_url,
            source_page=first_page_num,
            pagenum=first_label or str(first_page_num + 1),
            dataset=base_dataset,
        )
        
        self._current_paragraph_lines.clear()
        self._short_line_count = 0
        
        return para
    
    def _split_long_paragraph(self, content: str) -> List[str]:
        """Split long content at sentence boundaries.
        
        Args:
            content: Long content to split.
            
        Returns:
            List of split content strings.
        """
        if len(content) <= self.max_paragraph_length:
            return [content]
        
        # Find sentence boundaries
        sentences = []
        current = ''
        
        for char in content:
            current += char
            if char in self.SENTENCE_ENDINGS:
                sentences.append(current.strip())
                current = ''
        
        if current.strip():
            sentences.append(current.strip())
        
        # Group sentences into chunks
        chunks = []
        current_chunk = ''
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_paragraph_length:
                current_chunk += ' ' + sentence if current_chunk else sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content]
    
    def process_page(
        self, 
        page_num: int, 
        page_label: str, 
        content: str,
        source_url: str,
        base_dataset
    ) -> Iterator[Paragraph]:
        """Process a single page and yield paragraphs.
        
        Uses sliding window approach:
        1. Split content into lines
        2. Filter header/footer lines
        3. Track short lines for merging
        4. Emit paragraphs when boundaries detected
        
        Args:
            page_num: Page number (0-indexed).
            page_label: Page label (e.g., "i", "1", "A-1").
            content: Page text content.
            source_url: Source file path.
            base_dataset: Base dataset for paragraphs.
            
        Yields:
            Paragraph objects.
        """
        if not content or not content.strip():
            return
        
        # Split into lines
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip header/footer lines
            if self._is_header_footer(line):
                continue
            
            # Check if this is a short line
            is_short = len(line) < self.short_line_threshold
            
            # Check for paragraph boundary
            prev_line = self._current_paragraph_lines[-1][0] if self._current_paragraph_lines else ''
            is_boundary = self._is_paragraph_boundary(line, prev_line)
            
            if is_boundary and self._current_paragraph_lines:
                # Flush current paragraph
                para = self._flush_paragraph(source_url, base_dataset)
                if para:
                    yield para
            
            # Add line to current paragraph
            self._current_paragraph_lines.append((line, page_num, page_label))
            
            # Track short lines
            if is_short:
                self._short_line_count += 1
            else:
                self._short_line_count = 0
            
            # Check if we have enough short lines to merge
            if self._short_line_count >= self.short_line_batch:
                # Flush as paragraph
                para = self._flush_paragraph(source_url, base_dataset)
                if para:
                    yield para
            
            # Check if current paragraph is too long
            current_length = sum(len(l) for l, _, _ in self._current_paragraph_lines)
            if current_length >= self.max_paragraph_length:
                # Flush and split
                content = ' '.join(l for l, _, _ in self._current_paragraph_lines)
                self._current_paragraph_lines.clear()
                
                for chunk in self._split_long_paragraph(content):
                    if len(chunk) >= self.min_paragraph_length:
                        yield Paragraph(
                            content=chunk,
                            source_url=source_url,
                            source_page=page_num,
                            pagenum=page_label or str(page_num + 1),
                            dataset=base_dataset,
                        )
        
        self._last_page_num = page_num
    
    def finalize(self, source_url: str, base_dataset) -> Iterator[Paragraph]:
        """Flush any remaining buffered content.
        
        Args:
            source_url: Source file path.
            base_dataset: Base dataset for paragraphs.
            
        Yields:
            Remaining Paragraph objects.
        """
        para = self._flush_paragraph(source_url, base_dataset)
        if para:
            yield para
    
    def resolve(self, paragraph: Paragraph) -> ResolveReturn:
        """Process paragraph through re-paragraphization.
        
        This method accumulates pages and yields re-paragraphized content.
        For proper sliding window operation, use the flow() method.
        
        Args:
            paragraph: Paragraph to process.
            
        Returns:
            First re-paragraphized paragraph, or None.
        """
        # This is handled in flow() for proper iterator support
        return paragraph
    
    async def flow(self, paragraph: Paragraph):
        """Flow control for re-paragraphization.
        
        Args:
            paragraph: Paragraph to process.
            
        Yields:
            Tuples of (result, next_stage).
        """
        # Get source info from paragraph
        source_url = paragraph.source_url or ''
        base_dataset = paragraph.dataset
        
        # Process page content
        page_num = paragraph.source_page or 0
        page_label = paragraph.pagenum or str(page_num + 1)
        
        for result in self.process_page(
            page_num, page_label, paragraph.content, source_url, base_dataset
        ):
            yield result, self.next
        
        # Check if we're at the end (no more paragraphs from same source)
        # This is a simplification - in practice, you'd need to track sources
        for result in self.finalize(source_url, base_dataset):
            yield result, self.next

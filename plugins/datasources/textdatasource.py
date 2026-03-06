"""Text File Data Sources

This module provides data source implementations for importing content from
plain text files, file patterns, and structured text formats like EndNote.
"""

import codecs
from typing import Iterator, List, Optional

from jindai.models import Paragraph
from jindai.pipeline import DataSourceStage, PipelineStage
from jindai.storage import storage


class FilePatternDataSource(DataSourceStage):
    """Create paragraphs from file path patterns.
    
    This data source matches file patterns (globs) and creates Paragraph
    objects with the file paths as content. It does not read file contents,
    only generates metadata about the files.
    
    Attributes:
        paths: List of matched file paths.
    """

    def apply_params(self, content: str = "") -> None:
        """Configure the file pattern data source.
        
        Args:
            content: File path patterns (one per line). Supports glob patterns.
        """
        self.paths = PipelineStage.parse_paths(content)

    async def fetch(self) -> Iterator[Paragraph]:
        """Yield paragraphs with file paths as content.
        
        Yields:
            Paragraph objects with content set to the file path.
        """
        for path in await self.paths:
            yield Paragraph(content=path)


class TextDataSource(DataSourceStage):
    """Import paragraphs from text files, one line per paragraph.
    
    This data source reads text files line by line and creates a Paragraph
    object for each line. It supports both local files and URLs.
    
    Attributes:
        name: Target dataset name.
        lang: Language code for imported paragraphs.
        files: List of text file paths to process.
    """

    def apply_params(self, dataset_name: str = '', lang: str = 'auto', content: str = '') -> None:
        """Configure the text file data source.
        
        Args:
            dataset_name: Name of the target dataset for imported paragraphs.
            lang: Language code for imported paragraphs ('auto' for automatic detection).
            content: Paths to text files (one per line). Supports URLs.
        """
        self.name = dataset_name
        self.lang = lang
        self.files = PipelineStage.parse_paths(content)

    async def fetch(self) -> Iterator[Paragraph]:
        """Process configured text files and yield paragraphs.
        
        Yields:
            Paragraph objects, one for each line in the files.
            Each paragraph contains:
                - content: The line text
                - source_url: Path to the source file
                - dataset: Target dataset ID
                - lang: Language code
                - outline: Line number (zero-padded)
        """
        for path in await self.files:
            for i, line in enumerate(storage.open(path)):
                yield Paragraph(
                    content=codecs.decode(line),
                    source_url=path if '://' in path else storage.relative_path(path),
                    dataset=self.name, 
                    lang=self.lang, 
                    outline=f'{i+1:06d}'
                )


class LinesDataSource(DataSourceStage):
    """Import paragraphs from directly specified text content.
    
    This data source splits a text string into paragraphs, one per line.
    It's useful for importing small text snippets or when content is
    provided directly rather than from a file.
    
    Attributes:
        name: Target dataset name.
        lang: Language code for imported paragraphs.
        lines: List of text lines to convert to paragraphs.
        params: Additional parameters for paragraph creation.
    """

    def apply_params(
        self, 
        dataset_name: str = '', 
        lang: str = "auto", 
        content: str = "", 
        params: Optional[dict] = None, 
        delimiter: str = '\n'
    ) -> None:
        """Configure the lines data source.
        
        Args:
            dataset_name: Name of the target dataset for imported paragraphs.
            lang: Language code for imported paragraphs ('auto' for automatic detection).
            content: Text content to split into paragraphs.
            params: Additional parameters to pass to Paragraph.from_dict().
            delimiter: String used to split content into lines.
        """
        self.name = dataset_name
        self.lang = lang
        self.lines = content.split(delimiter)
        self.params = params or {}

    async def fetch(self) -> Iterator[Paragraph]:
        """Yield paragraphs from configured text lines.
        
        Yields:
            Paragraph objects with content from each line.
        """
        self.params.pop('contnet', '')  # Fix typo in original code
        for line in self.lines:
            yield Paragraph(lang=self.lang, dataset=self.name, content=line)


class BiblioDataSource(DataSourceStage):
    """Import paragraphs from bibliography files (EndNote format).
    
    This data source parses bibliography files in EndNote format and creates
    Paragraph objects with structured metadata including authors, title,
    publication date, journal, and abstract.
    
    Supported formats:
    - EndNote: Standard EndNote export format
    
    EndNote Format:
        Each reference is separated by a blank line.
        Fields start with a percentage sign followed by a code:
        - %A: Authors
        - %T: Title
        - %J: Journal
        - %D: Publication date
        - %X: Abstract/Content
        - %K: Keywords
        - %P: Pages
        - %@: ISSN
        - %L: CN publishing number
        - %W: Catalog
        - %0: Item type
        - %+: Institutions
        
    Attributes:
        dataset: Target dataset name.
        lang: Language code for imported paragraphs.
        files: List of bibliography file paths to process.
        method: Parsing method for the input format.
    """

    def apply_params(
        self, 
        content: str = '', 
        dataset_name: str = '', 
        lang: str = 'zhs', 
        input_format: str = 'endnote'
    ) -> None:
        """Configure the bibliography data source.
        
        Args:
            content: Paths to bibliography files (one per line).
            dataset_name: Name of the target dataset for imported paragraphs.
            lang: Language code for imported paragraphs (default: 'zhs').
            input_format: Format of the input file (only 'endnote' supported).
        """
        if not hasattr(self, input_format):
            raise NotImplementedError(f"Format '{input_format}' is not supported")

        self.method = getattr(self, input_format)
        self.dataset = dataset_name
        self.lang = lang
        self.files = PipelineStage.parse_paths(content)

    def endnote(self, lines: Iterator[bytes]) -> Iterator[dict]:
        """Parse EndNote format bibliography entries.
        
        Args:
            lines: Iterator over raw bytes from the bibliography file.
            
        Yields:
            Dictionaries with parsed bibliography data including:
                - content: Abstract or main text
                - authors: List of author names
                - title: Publication title
                - journal: Journal name
                - pdate: Publication date
                - issue: Issue number
                - tags: List of keywords
                - pages: Page numbers
                - issn: ISSN
                - cn_publishing_number: CN publishing number
                - catalog: Catalog information
        """
        doc = {
            'content': '',
            'authors': []
        }
        field = ''
        for line in lines:
            if not line.strip():
                # End of entry - yield if we have content
                if doc:
                    yield Paragraph.from_dict(dict(dataset=self.dataset, lang=self.lang, **doc))
                doc = {
                    'content': '',
                    'authors': []
                }
            else:
                line = line.decode('utf-8').strip()
                if ' ' not in line:
                    # Line without field code - append to current field
                    value = line
                else:
                    # Parse field code and value
                    field, value = line.split(' ', 1)
                    # Map field codes to field names
                    field = {
                        '%0': 'item_type',
                        '%A': 'authors',
                        '%+': 'institutions',
                        '%J': 'journal',
                        '%D': 'pdate',
                        '%T': 'title',
                        '%N': 'issue',
                        '%K': 'tags',
                        '%X': 'content',
                        '%P': 'pages',
                        '%@': 'issn',
                        '%L': 'cn_publishing_number',
                        '%W': 'catalog'
                    }.get(field.upper(), 'content')

                # Handle multi-value fields (semicolon-separated)
                if ';' in value and field != 'content':
                    value = [_ for _ in value.split(';') if _]
                    
                # Accumulate values in doc dictionary
                if field in doc:
                    if field == 'content':
                        doc[field] += value
                    else:
                        if not isinstance(doc[field], list):
                            doc[field] = [doc[field]]
                        if isinstance(value, list):
                            doc[field] += value
                        else:
                            doc[field].append(value)
                else:
                    doc[field] = value
        
        # Yield final entry if any
        if doc:
            yield Paragraph.from_dict(dict(dataset=self.dataset, **doc))

    async def fetch(self) -> Iterator[Paragraph]:
        """Process configured bibliography files and yield paragraphs.
        
        Yields:
            Paragraph objects with parsed bibliography data.
        """
        for file in await self.files:
            for item in self.method(file):
                yield item

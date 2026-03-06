"""Office Document Data Sources

This module provides data source implementations for importing content from
Microsoft Word (.docx) and Excel (.xlsx) documents.
"""

import os
import subprocess
import tempfile
from typing import Iterable, List, Optional

import pandas as pd

from jindai.models import Dataset, Paragraph
from jindai.pipeline import DataSourceStage, PipelineStage
from jindai.storage import storage


class WordDataSource(DataSourceStage):
    """Import paragraphs from Microsoft Word documents.
    
    This data source uses the abiword command-line tool to convert
    Word documents to plain text, then creates Paragraph objects
    containing the extracted content.
    
    Requirements:
        - abiword must be installed and available in PATH
        
    Attributes:
        dataset_name: Target dataset name.
        lang: Language code for imported paragraphs.
        files: List of Word document paths to process.
    """

    def apply_params(self, dataset_name: str = '', lang: str = 'auto', content: str = '') -> None:
        """Configure the Word document data source.
        
        Args:
            dataset_name: Name of the target dataset for imported paragraphs.
            lang: Language code for imported paragraphs ('auto' for automatic detection).
            content: Paths to Word documents (one per line).
        """
        self.dataset_name = dataset_name
        self.lang = lang
        self.files = PipelineStage.parse_paths(content)

    def call_abiword(self, file: str) -> Optional[str]:
        """Extract text from a Word document using abiword.
        
        Args:
            file: Path to the Word document (.doc or .docx).
            
        Returns:
            Extracted text content, or None if extraction fails.
        """
        filename = tempfile.mktemp()
        subprocess.call(['abiword', '--to', 'txt', '-o', filename, file])
        if os.path.exists(filename):
            with open(filename, encoding='utf-8') as input_file:
                res = input_file.read()
            os.unlink(filename)
            return res
        return None

    async def fetch(self) -> Iterable[Paragraph]:
        """Process configured Word documents and yield paragraphs.
        
        Yields:
            Paragraph objects with extracted text content from each document.
        """
        dataset = await Dataset.get(self.dataset_name)
        for file in await self.files:
            doc = self.call_abiword(file)
            if doc:
                para = Paragraph(
                    lang=self.lang, 
                    content=doc,
                    source_url=storage.relative_path(file),
                    pagenum=1,
                    dataset=dataset.id,
                    outline=''
                )
                yield para


class ExcelDataSource(DataSourceStage):
    """Import paragraphs from Excel/CSV documents.
    
    This data source reads Excel files (XLSX, XLS) or CSV files and creates
    Paragraph objects from each row. Column names are mapped to Paragraph
    attributes (content, author, title, etc.).
    
    Attributes:
        dataset_name: Target dataset name.
        lang: Language code for imported paragraphs.
        files: List of Excel/CSV file paths to process.
    """

    def apply_params(self, content: str = '', dataset_name: str = '', lang: str = 'auto') -> None:
        """Configure the Excel data source.
        
        Args:
            content: Paths to Excel/CSV files (one per line).
            dataset_name: Name of the target dataset for imported paragraphs.
            lang: Language code for imported paragraphs ('auto' for automatic detection).
        """
        self.dataset_name = dataset_name
        self.lang = lang
        self.files = PipelineStage.parse_paths(content)

    async def fetch(self) -> Iterable[Paragraph]:
        """Process configured Excel files and yield paragraphs.
        
        Yields:
            Paragraph objects created from each row in the Excel files.
        """
        dataset = await Dataset.get(self.dataset_name)
        for file in await self.files:
            dataframe = pd.read_excel(file)
            for _, row in dataframe.iterrows():
                data = row.to_dict()
                # Set default values if not present in data
                if 'dataset' not in data:
                    data['dataset'] = dataset.id
                if 'lang' not in data:
                    data['lang'] = self.lang
                yield Paragraph.from_dict(data)

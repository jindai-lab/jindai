"""HTML and Web Data Sources

This module provides data source implementations for:
- Extracting paragraphs from HTML web pages
- Parsing JSON data from web APIs
- Extracting structured content from HTML documents
"""

import asyncio
import datetime
import json
import os
import tempfile
from hashlib import sha1
from typing import Dict, Iterable, List, Optional, Union

import regex as re
import httpx
from bs4 import BeautifulSoup as B

from jindai.config import config
from jindai.storage import storage
from jindai.helpers import safe_import, aeval
from jindai.models import Paragraph
from jindai.pipeline import DataSourceStage, PipelineStage

trafilatura = safe_import('trafilatura')
from trafilatura.settings import use_config

trafcfg = use_config()
trafcfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")


DEFAULT_IMG_PATTERNS = 'img[src]|[zoomfile]|[data-original]|[data-src]|[file]|[data-echo]'.replace(
    '|', '\n')


class CachedWebAccess:
    """Cached web access helper for downloading and storing web pages.
    
    This class provides asynchronous HTTP requests with caching support.
    It stores downloaded pages locally to avoid redundant network requests.
    Optionally supports browser-based rendering for JavaScript-heavy pages.
    
    Attributes:
        base: Directory path for storing cached files.
    """

    def __init__(self, base: str) -> None:
        """Initialize the cache directory.
        
        Args:
            base: Directory path for storing cached files. Created if it doesn't exist.
        """
        if not os.path.exists(base):
            os.makedirs(base)
        self.base = base
        
    def _digest(self, url: str) -> str:
        """Generate a file path for a URL's cached content.
        
        Args:
            url: The URL to generate a cache path for.
            
        Returns:
            Full file path for the cached content.
        """
        return os.path.join(self.base, sha1(url.encode('utf-8')).hexdigest())
    
    async def request(self, url: str) -> bytes:
        """Download a web page using httpx.
        
        Args:
            url: The URL to fetch.
            
        Returns:
            Raw bytes of the response content.
        """
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers={
                'User-Agent': 'Mozilla/5.0',
                'Referer': url
            })
        return resp.content    
    
    async def request_browserless(self, url: str, wait_for: str = '') -> bytes:
        """Download a web page using browserless (for JavaScript rendering).
        
        Args:
            url: The URL to fetch.
            wait_for: CSS selector to wait for before capturing content.
            
        Returns:
            Raw bytes of the rendered page HTML.
        """
        pyppeteer = safe_import('pyppeteer')

        browser = await pyppeteer.launcher.connect(
            browserWSEndpoint=config.browserless
        )
        page = await browser.newPage()
        await page.goto(url)
        if wait_for:
            await page.waitForSelector(wait_for)
        
        values = await page.evaluate('''() => document.documentElement.outerHTML''')
        
        await browser.close()
        return values.encode('utf-8')
        
    async def get(self, url: str, with_chrome: bool = False, wait_for: str = '') -> bytes:
        """Get page content, using cache if available.
        
        Args:
            url: The URL to fetch.
            with_chrome: If True, use browserless for rendering.
            wait_for: CSS selector to wait for before capturing (browserless only).
            
        Returns:
            Raw bytes of the page content.
        """
        hashed = self._digest(url)
        if os.path.exists(hashed):
            with open(hashed, 'rb') as fi:
                return fi.read()
        else:
            if url.split('://')[0] in ('http', 'https'):
                if with_chrome or wait_for:
                    data = await self.request_browserless(url, wait_for)
                else:
                    data = await self.request(url)
            else:
                data = storage.open(url, 'rb').read()
            if data:
                with open(hashed, 'wb') as fo:
                    fo.write(data)
            return data


class WebPageListingDataSource(DataSourceStage):
    """Extract paragraphs from web page listings and detail pages.
    
    This data source crawls web pages in two modes:
    1. Listing pages: Extract links to other pages (detail or listing pages)
    2. Detail pages: Extract paragraph content from the page
    
    It supports:
    - Multi-level crawling with configurable depth
    - URL scope filtering
    - Image pattern extraction
    - JavaScript rendering via browserless
    - Caching of downloaded pages
    
    Attributes:
        cache: Shared CachedWebAccess instance for all instances.
    """

    cache = CachedWebAccess(os.path.join(
        os.path.dirname(tempfile.mkdtemp()), 'wpdl'))

    @property
    def visited(self) -> set:
        """Set of visited URLs (stored in global context)."""
        if 'visited' not in self.gctx:
            self.gctx['visited'] = set()
        return self.gctx['visited']

    @property
    def queued(self) -> set:
        """Set of URLs queued for processing (stored in global context)."""
        if 'queued' not in self.gctx:
            self.gctx['queued'] = set()
        return self.gctx['queued']

    def apply_params(
        self, 
        dataset: str = '', 
        content: str = '', 
        scopes: str = '',
        lang: str = 'auto', 
        detail_link: str = '',
        list_link: str = '', 
        proxy: str = '', 
        list_depth: int = 1, 
        tags: str = '',
        img_pattern: str = '', 
        level: int = 1, 
        wait_for: str = '',
        with_chrome: bool = False,
        base_cls: Optional[type] = None
    ) -> None:
        """Configure the web page listing data source.
        
        Args:
            dataset: Name of the target dataset for imported paragraphs.
            content: Entry URLs (one per line) to start crawling from.
            scopes: URL scope patterns (one per line) to limit crawling.
            lang: Language code for imported paragraphs ('auto' for automatic).
            detail_link: Regex pattern for matching detail page URLs.
            list_link: Regex pattern for matching listing page URLs.
            proxy: HTTP proxy URL for requests.
            list_depth: Maximum crawling depth for listing pages.
            tags: Tags to apply to extracted paragraphs (one per line).
            img_pattern: CSS selectors for image elements (one per line).
            level: Starting depth level for crawling.
            wait_for: CSS selector to wait for before capturing (browserless).
            with_chrome: If True, use browserless for JavaScript rendering.
            base_cls: Base class for creating Paragraph objects (default: Paragraph).
        """
        self.base_cls = base_cls or Paragraph
        self.proxies = {} if not proxy else {
            'http': proxy,
            'https': proxy
        }
        self.paths = PipelineStage.parse_paths(content)
        self.scopes = PipelineStage.parse_paths(scopes) or self.paths
        self.list_depth = list_depth
        self.detail_link = re.compile(detail_link)
        self.list_link = re.compile(list_link)
        self.tags = PipelineStage.parse_lines(tags)
        self.dataset = dataset
        self.lang = lang
        self.image_patterns = PipelineStage.parse_lines(
            img_pattern) or DEFAULT_IMG_PATTERNS.split('\n')
        self.level = level
        self.wait_for = wait_for
        self.with_chrome = with_chrome

    async def get_url(self, url: str) -> Paragraph:
        """Download a web page and return it as a Paragraph.
        
        Args:
            url: The URL to fetch.
            
        Returns:
            Paragraph with source_url and extdata['html'] containing the raw HTML.
        """
        self.log('get url', url)
        try:
            data = await WebPageListingDataSource.cache.get(url, self.with_chrome, self.wait_for)
        except OSError as ose:
            self.log_exception(f'Error while reading from {url}', ose)
            data = b''
        return Paragraph(
            source_url=url, 
            extdata={'html': data.decode('utf-8', errors='ignore')}, 
            dataset=self.dataset, 
            lang=self.lang
        )

    def get_text(self, element) -> str:
        """Extract clean text from a BeautifulSoup element.
        
        Args:
            element: BeautifulSoup element to extract text from.
            
        Returns:
            Cleaned text with whitespace normalized.
        """
        if element and element.text:
            return re.sub(r'\s+', ' ', element.text)
        return ''

    def parse_detail(self, url: str, para: Paragraph, b: B) -> Paragraph:
        """Parse a detail page and extract paragraph content.
        
        Args:
            url: The URL of the detail page.
            para: Base Paragraph with HTML content.
            b: BeautifulSoup object of the parsed HTML.
            
        Returns:
            Paragraph with extracted content, title, date, and tags.
        """
        para.pdate = datetime.datetime.now()
        para.source_url = url
        para.dataset = self.dataset
        para.lang = self.lang
        para.content = self.get_text(b)
        para.keywords = self.tags
        title_elem = b.find('title')
        para.title = self.get_text(title_elem) if title_elem else ''
        
        return para

    async def parse_list(self, url: str, b: B) -> List[str]:
        """Parse a listing page and extract links to other pages.
        
        Args:
            url: The URL of the listing page.
            b: BeautifulSoup object of the parsed HTML.
            
        Returns:
            List of absolute URLs found on the page that match scope and pattern criteria.
        """
        if not b:
            self.log(f'Cannot read list from {url}')
            return []

        links: set = set()
        for a in b.select('a[href]'):
            link_url = a['href'] = urljoin(url, a['href'])
            link_url = link_url.split('#')[0]
            
            # Skip already visited URLs
            if link_url in self.visited:
                continue
                
            # Check if URL is within configured scopes
            for scope in await self.scopes:
                if link_url.startswith(scope):
                    break
            else:
                continue
                
            # Check if URL matches listing or detail patterns
            if self.list_link.search(link_url) or self.detail_link.search(link_url):
                links.add(link_url)

        self.log(len(links), 'links')
        return list(links)

    async def fetch(self):
        """Crawl web pages and yield paragraphs.
        
        Yields:
            - Paragraph objects for detail pages with extracted content
            - Paragraph objects for listing pages with next-level URLs
        """
        level = self.level or 1
        for url in await self.paths:
            if url in self.visited:
                continue
            self.visited.add(url)

            para = await self.get_url(url)
            b = B(para.extdata.get('html', ''), 'lxml')
            para.extdata['html'] = str(b)

            # Process listing pages
            if level <= self.list_depth and (self.list_link.search(url) or self.detail_link.search(url)):
                self.log('parse list', url, 'level', level)
                for upath in await self.parse_list(url, b):
                    if upath not in self.queued:
                        self.queued.add(upath)
                        yield Paragraph.from_dict(content=upath, level=level+1), self

            # Process detail pages
            if self.detail_link.search(url):
                self.log('parse detail', url)
                yield self.parse_detail(url, para, b)

    async def summarize(self, result: Dict) -> Dict:
        """Clean up visited and queued URL sets after processing.
        
        Args:
            result: The processing result dictionary.
            
        Returns:
            The result dictionary with visited and queued sets cleared.
        """
        self.log('clear visited & queued urls')
        self.visited.clear()
        self.queued.clear()


class JSONDataSource(DataSourceStage):
    """Parse JSON data into Paragraph objects.
    
    This data source converts JSON data (typically from web APIs) into
    Paragraph objects. It handles both single objects and arrays, and
    can extract results from nested structures (e.g., {'results': [...]})
    
    Attributes:
        content: Parsed JSON data as a list of dictionaries.
    """

    def apply_params(self, content: str = '', **kwargs) -> None:
        """Configure the JSON data source.
        
        Args:
            content: JSON string to parse. Can be:
                - A single JSON object
                - A JSON array of objects
                - A JSON object with a 'results' array
            **kwargs: Additional keyword arguments (ignored, for compatibility).
        """
        self.content = json.loads(content)
        if isinstance(self.content, dict) and 'results' in self.content:
            self.content = self.content['results']
        if not isinstance(self.content, list):
            self.content = [self.content]

    async def fetch(self) -> Iterable[Paragraph]:
        """Parse JSON content and yield Paragraph objects.
        
        Yields:
            Paragraph objects created from each JSON object using from_dict().
        """
        for paragraph in self.content:
            yield Paragraph.from_dict(paragraph)


class ExtractHTMLParagraphs(PipelineStage):
    """Extract paragraphs from HTML content using CSS selectors.
    
    This pipeline stage extracts structured content from HTML documents.
    It supports:
    - Custom CSS selectors for paragraph elements
    - Field assignments using CSS selectors
    - Automatic content extraction using trafilatura
    
    Field Assignment Syntax:
        field="css-selector//attribute"
        - css-selector: CSS selector for the element
        - attribute: Field to extract ('text', 'html', or any attribute name)
        
    Examples:
        content=".article-body//text"
        title="h1//text"
        author=".author//text"
        date=".pubdate//data-pubdate"
    """

    def __init__(
        self, 
        field: str = 'html', 
        autoextract: bool = False, 
        assignments: str = '', 
        paragraph_selector: str = ''
    ) -> None:
        """Initialize the HTML paragraph extractor.
        
        Args:
            field: Name of the field containing HTML content in input Paragraphs.
            autoextract: If True, use trafilatura to automatically extract main content.
            assignments: Field assignment string (parsed by aeval).
            paragraph_selector: CSS selector for paragraph elements.
                Empty string means the entire page is one paragraph.
        """
        super().__init__()
        self.field = field
        self.paragraph_selector = paragraph_selector
        if isinstance(assignments, str):
            assignments = aeval(assignments)
        self.assignments = assignments or {'content': '//text'}
        self.autoextract = autoextract

    def _get_text(self, bs_ele) -> str:
        """Extract and clean text from a BeautifulSoup element.
        
        Args:
            bs_ele: BeautifulSoup element to extract text from.
            
        Returns:
            Cleaned text with whitespace normalized.
        """
        if bs_ele and bs_ele.text:
            return re.sub(r'\s+', ' ', bs_ele.text)
        return ''

    def _resolve_assignments(self, bs_ele, para: Paragraph) -> None:
        """Extract field values from HTML element and set on Paragraph.
        
        Args:
            bs_ele: BeautifulSoup element to extract from.
            para: Paragraph object to set extracted values on.
        """
        for field_name, field_path in self.assignments.items():
            if '//' in field_path:
                field_path, field_attr = field_path.rsplit('//', 1)
            else:
                field_attr = 'text'
            elements = bs_ele.select(field_path) if field_path else [bs_ele]
            value = []
            for element in elements:
                if field_attr == 'text':
                    value.append(self._get_text(element))
                elif field_attr == 'html':
                    value.append(str(element))
                elif field_attr in element.attrs:
                    value.append(str(element.attrs[field_attr]))
            if field_name == 'content':
                value = '\n'.join(value)
            elif field_name != 'keywords':
                value = ' '.join(value)
            setattr(para, field_name, value)
        if self.autoextract:
            para.content = self.extract(str(bs_ele))
    
    def extract(self, html: str) -> Optional[str]:
        """Extract main content from HTML using trafilatura.
        
        Args:
            html: Raw HTML string to extract from.
            
        Returns:
            Extracted text content, or None if extraction fails.
        """
        return trafilatura.extract(html, config=trafcfg)

    def resolve(self, paragraph: Paragraph) -> Iterable[Paragraph]:
        """Extract paragraphs from HTML content.
        
        Args:
            paragraph: Input Paragraph with HTML in extdata[field].
            
        Yields:
            Paragraph objects with extracted content from HTML elements.
        """
        html = paragraph.extdata.get(self.field, '') or ''
        b = B(html, 'lxml')
        self.log('load html data of length', len(html))

        # Select elements to extract
        elements = b.select(self.paragraph_selector) if self.paragraph_selector else [b]
        
        for html_para in elements:
            para = Paragraph(
                lang=paragraph.lang,
                content='',
                source_url=paragraph.source_url,
                pagenum=1,
                dataset=paragraph.dataset,
                outline=paragraph.outline,
                keywords=[],
                extdata={'html': str(html_para)},
            )
            self._resolve_assignments(html_para, para)
            self.log('Extract para at', para.source_url)
            yield para

import asyncio
import httpx
import re
import time
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional, Dict, Any, List

from jindai.models import Dataset, Paragraph
from jindai.pipeline import DataSourceStage, PipelineStage
from jindai.storage import storage
from jindai.config import config


class MinerULocalClient:
    """MinerU 本地 API 客户端（异步版本，基于 httpx）"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000", 
        timeout: int = 300,
        max_connections: int = 10
    ):
        """
        初始化客户端
        
        Args:
            base_url: 本地 API 服务地址，默认 http://localhost:8000
            timeout: 请求超时时间（秒）
            max_connections: 最大并发连接数
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = httpx.Timeout(timeout)
        
        # 创建客户端连接池
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=max_connections),
            http2=True  # 启用 HTTP/2
        )
    
    async def convert_to_markdown(
        self, 
        file_path: str, 
        output_path: Optional[str] = None,
        lang_list: list = None,
        formula_enable: bool = True,
        table_enable: bool = True,
        parse_method: str = "auto",
        return_md: bool = True
    ) -> str:
        """
        将文档转换为 Markdown 格式
        
        Args:
            file_path: 文档路径（支持 PDF、DOCX、PPT、图片等）
            output_path: 输出 Markdown 文件路径（可选）
            lang_list: 语言列表，如 ["ch"] 中文、["en"] 英文
            formula_enable: 是否启用公式识别
            table_enable: 是否启用表格识别
            parse_method: 解析方法，auto/txt/ocr
            return_md: 是否返回 Markdown 内容
            
        Returns:
            Markdown 格式的文本内容
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 检查文件大小（限制 200MB）
        file_size = file_path.stat().st_size
        if file_size > 200 * 1024 * 1024:
            raise ValueError(f"文件大小超过 200MB 限制: {file_size / 1024 / 1024:.1f}MB")
        
        print(f"📄 正在处理文件: {file_path.name} ({file_size / 1024:.1f}KB)")
        
        # 准备请求参数
        url = f"{self.base_url}/file_parse"
        
        # 准备文件和数据
        files = {
            'files': (file_path.name, open(file_path, 'rb'), self._get_mime_type(file_path))
        }
        
        data = {
            'lang_list': lang_list or ['ch'],
            'formula_enable': str(formula_enable).lower(),
            'table_enable': str(table_enable).lower(),
            'parse_method': parse_method,
            'return_md': str(return_md).lower(),
        }
        
        print("⏳ 正在解析文档，请稍候...")
        start_time = time.time()
        
        try:
            # 发送请求（httpx 会自动处理 multipart/form-data）
            response = await self.client.post(
                url, 
                files=files,  # httpx 支持 files 参数
                data=data
            )
            response.raise_for_status()
            result = response.json()
            
        finally:
            # 确保文件被关闭
            files['files'][1].close()
        
        elapsed = time.time() - start_time
        print(f"✅ 解析完成，耗时 {elapsed:.1f} 秒")
        
        # 提取 Markdown 内容
        md_content = self._extract_markdown_from_response(result, file_path.name)
        
        # 保存到文件
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(md_content, encoding='utf-8')
            print(f"💾 Markdown 已保存至: {output_file}")
        
        return md_content
    
    def _extract_markdown_from_response(self, response: Dict[str, Any], filename: str) -> str:
        """从 API 响应中提取 Markdown 内容"""
        # 根据 MinerU API 的响应格式提取
        if 'results' in response:
            # 多文件场景
            for key, value in response['results'].items():
                if key == filename or key == Path(filename).stem:
                    if isinstance(value, dict):
                        return value.get('md_content', value.get('markdown', str(value)))
                    return str(value)
            # 取第一个结果
            first_result = list(response['results'].values())[0]
            if isinstance(first_result, dict):
                return first_result.get('md_content', first_result.get('markdown', str(first_result)))
            return str(first_result)
        
        if 'md_content' in response:
            return response['md_content']
        
        if 'markdown' in response:
            return response['markdown']
        
        if 'data' in response and isinstance(response['data'], dict):
            return response['data'].get('markdown', str(response))
        
        # 如果找不到特定字段，返回整个响应的 JSON
        import json
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    def _get_mime_type(self, file_path: Path) -> str:
        """根据文件扩展名获取 MIME 类型"""
        ext = file_path.suffix.lower()
        mime_types = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.txt': 'text/plain',
            '.html': 'text/html',
        }
        return mime_types.get(ext, 'application/octet-stream')
    
    async def convert_batch(
        self, 
        file_paths: List[str], 
        output_dir: str = "./output", 
        max_concurrent: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        批量转换文档（支持并发控制）
        
        Args:
            file_paths: 文件路径列表
            output_dir: 输出目录
            max_concurrent: 最大并发数（避免资源耗尽）
            **kwargs: 传递给 convert_to_markdown 的其他参数
            
        Returns:
            转换结果统计
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'success': [],
            'failed': [],
            'total': len(file_paths)
        }
        
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def convert_one(file_path: str, index: int):
            async with semaphore:
                try:
                    file_name = Path(file_path).stem
                    output_path = output_dir / f"{file_name}.md"
                    
                    print(f"\n[{index}/{len(file_paths)}] ", end="")
                    md_content = await self.convert_to_markdown(
                        file_path, str(output_path), **kwargs
                    )
                    results['success'].append({
                        'file': file_path, 
                        'output': str(output_path)
                    })
                    
                except Exception as e:
                    print(f"❌ 转换失败: {file_path}, 错误: {e}")
                    results['failed'].append({
                        'file': file_path, 
                        'error': str(e)
                    })
        
        # 创建所有转换任务
        tasks = [
            convert_one(file_path, i + 1) 
            for i, file_path in enumerate(file_paths)
        ]
        
        # 等待所有任务完成
        await asyncio.gather(*tasks)
        
        print(f"\n📊 批量转换完成: 成功 {len(results['success'])} 个, 失败 {len(results['failed'])} 个")
        return results
    
    async def convert_streaming(self, file_path: str) -> AsyncIterator[bytes]:
        """
        流式获取转换结果（适用于大文件）
        
        Args:
            file_path: 文件路径
            
        Yields:
            流式数据块
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        url = f"{self.base_url}/file_parse"
        
        files = {
            'files': (file_path.name, open(file_path, 'rb'), self._get_mime_type(file_path))
        }
        
        data = {
            'lang_list': ['ch'],
            'formula_enable': 'true',
            'table_enable': 'true',
            'return_md': 'true',
            'stream': 'true'  # 如果 API 支持流式响应
        }
        
        try:
            async with self.client.stream(
                'POST', 
                url, 
                files=files, 
                data=data
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        finally:
            files['files'][1].close()
    
    async def close(self):
        """关闭客户端连接池"""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class MinerUDataSource(DataSourceStage):
    """Import paragraphs from documents using MinerU for OCR and layout analysis.
    
    This data source uses the MinerU local API service to convert documents
    (PDF, DOCX, PPT, images, etc.) into structured Markdown, then parses
    the Markdown into Paragraph objects based on document structure.
    
    Attributes:
        base_url: Base URL for MinerU service.
        content: File paths to process.
        max_connections: Max parallel connections to the service.
        dataset_name: Target dataset name.
        lang: Language code for imported paragraphs.
    """
    
    # Regex to match Markdown headings (e.g., # Heading 1, ## Heading 2, etc.)
    HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def __init__(self, **params) -> None:
        self.base_url = config.mineru
        super().__init__(**params)
    
    def apply_params(
        self,
        content: str,
        max_connections: int = 10,
        dataset_name: str = "",
        lang: str = "auto"
    ) -> None:
        """Configure the MinerU data source.
        
        Args:
            content: Paths to documents (one per line).
            max_connections: Max parallel connections to the service.
            dataset_name: Name of the target dataset for imported paragraphs.
            lang: Language code for imported paragraphs ('auto' for automatic detection).
        """
        self.max_connections = max_connections
        self.files = PipelineStage.parse_paths(content)
        self.dataset_name = dataset_name
        self.lang = lang

    def _parse_markdown_to_sections(self, md_content: str) -> Iterator[Paragraph]:
        """Parse Markdown content into Paragraph objects by sections.
        
        Splits the Markdown content by headings and creates a Paragraph
        for each section. The heading text is used as the outline/summary.
        
        Args:
            md_content: The Markdown content to parse.
            source_url: Path to the source document.
            
        Yields:
            Paragraph objects for each section.
        """
        # Find all headings with their positions
        headings = list(self.HEADING_RE.finditer(md_content))
        
        if not headings:
            # No headings found - yield entire content as single paragraph
            content = md_content.strip()
            if content:
                yield Paragraph(
                    lang=self.lang,
                    content=content,
                    source_page=0,
                    pagenum="1",
                    dataset=self.dataset_name,
                    outline="Full Document",
                )
            return
        
        # Process each section
        for i, match in enumerate(headings):
            heading_level = len(match.group(1))  # Number of # characters
            heading_text = match.group(2).strip()
            start_pos = match.end()
            
            # End position is the start of the next heading (or end of content)
            if i + 1 < len(headings):
                end_pos = headings[i + 1].start()
            else:
                end_pos = len(md_content)
            
            # Extract section content (excluding the heading itself)
            section_content = md_content[start_pos:end_pos].strip()
            
            # Skip empty sections
            if not section_content:
                continue
            
            yield Paragraph(
                lang=self.lang,
                content=section_content,
                source_page=0,
                pagenum=str(i + 1),
                dataset=self.dataset_name,
                outline=heading_text,
                extdata={"heading_level": heading_level},
            )

    async def fetch(self) -> Iterator[Paragraph]:
        """Process configured documents and yield paragraphs.
        
        Yields:
            Paragraph objects created from the parsed Markdown content.
        """
        dataset = await Dataset.get(self.dataset_name)
        
        async with MinerULocalClient(
            base_url=self.base_url,
            timeout=300,
            max_connections=self.max_connections
        ) as client:
            for path in await self.files:
                md_content = await client.convert_to_markdown(
                    file_path=path,
                    lang_list=['ch', 'en', 'fr', 'de', 'ru', 'ja'],
                    formula_enable=True,   # 启用公式识别
                    table_enable=True      # 启用表格识别
                )
                
                # Parse markdown content into paragraphs
                source_url = storage.relative_path(path) if hasattr(storage, 'relative_path') else str(path)
                for para in self._parse_markdown_to_sections(md_content):
                    # Ensure dataset is set correctly
                    para.dataset = dataset.id
                    para.source_url = source_url
                    yield para

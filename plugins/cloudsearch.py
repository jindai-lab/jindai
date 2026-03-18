"""Cloud Search Integration for Multiple Providers
@zhs 多云搜索集成
"""
import hmac
import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import regex as re

from jindai.pipeline import PipelineStage
from jindai.plugin import Plugin
from jindai.models import Paragraph


class CloudSearchBase(PipelineStage):
    """Base class for cloud search integration.
    
    Provides common functionality for calling cloud search APIs including:
    - HTTP client management
    - Request/response logging
    - Error handling
    """
    
    provider_name: str = "CloudSearch"
    
    def __init__(
        self,
        api_endpoint: str = "",
        api_key: str = "",
        index_name: str = "",
        search_fields: List[str] = None,
        result_field: str = "cloud_search_results",
        timeout: float = 30.0,
        verbose: bool = False,
    ) -> None:
        """Initialize cloud search stage.
        
        Args:
            api_endpoint: Cloud search API endpoint
                @zhs 搜索API地址
            api_key: API key for authentication
                @zhs API密钥
            index_name: Name of the search index to query
                @zhs 搜索索引名称
            search_fields: Fields to search in (default: ['content'])
                @zhs 搜索字段列表
            result_field: Field name to store search results in paragraph.extdata
                @zhs 结果存储字段名
            timeout: Request timeout in seconds (default: 30.0)
                @zhs 请求超时时间（秒）
            verbose: Enable verbose logging (default: False)
                @zhs 启用详细日志
        """
        super().__init__()
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.index_name = index_name
        self.search_fields = search_fields or ['content']
        self.result_field = result_field
        self.timeout = timeout
        self.verbose = verbose
        self.client: Optional[httpx.AsyncClient] = None
        self._initialized = False
    
    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized.
        
        Returns:
            Initialized httpx.AsyncClient instance.
        """
        if not self._initialized:
            self.client = httpx.AsyncClient(timeout=self.timeout)
            self._initialized = True
        return self.client
    
    async def _close_client(self) -> None:
        """Close HTTP client."""
        if self.client and self._initialized:
            await self.client.aclose()
            self._initialized = False
    
    def _log(self, *args) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logging.info(f"[{self.provider_name}] {' '.join(map(str, args))}")
    
    async def _build_search_query(self, content: str) -> Dict[str, Any]:
        """Build search query from content.
        
        Args:
            content: Content to search for
            
        Returns:
            Search query dictionary
        """
        # Clean content for search
        clean_content = re.sub(r'\s+', ' ', content).strip()
        return clean_content[:1000]  # Limit query length
    
    async def _call_search_api(self, query: str) -> Dict[str, Any]:
        """Call cloud search API.
        
        Args:
            query: Search query string
            
        Returns:
            API response dictionary
        """
        raise NotImplementedError
    
    async def resolve(self, paragraph: Paragraph) -> Paragraph:
        """Search for paragraph content using cloud search.
        
        Args:
            paragraph: Paragraph to search
            
        Returns:
            Paragraph with search results in extdata
        """
        if not paragraph.content:
            self._log("No content to search")
            return paragraph
        
        if not self.api_key:
            self._log("API key not configured")
            return paragraph
        
        if not self.index_name:
            self._log("Index name not configured")
            return paragraph
        
        try:
            # Build search query
            query = await self._build_search_query(paragraph.content)
            
            # Call API
            result = await self._call_search_api(query)
            
            # Store results in paragraph.extdata
            if paragraph.extdata is None:
                paragraph.extdata = {}
            
            paragraph.extdata[self.result_field] = result
            
            self._log(f"Found {len(result.get('hits', []))} results")
            
        except httpx.HTTPStatusError as e:
            self._log(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            self._log(f"Request error: {e}")
        except Exception as e:
            self._log(f"Error during search: {e}")
        
        return paragraph
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_client()


class TencentCloudSearch(CloudSearchBase):
    """Tencent Cloud Search Integration
    @zhs 腾讯云搜索集成
    
    This pipeline stage calls Tencent Cloud Search API to search for content
    extracted from paragraph.content.
    """
    
    provider_name: str = "TencentCloudSearch"
    
    def __init__(
        self,
        api_endpoint: str = "https://search.tencentcloudapi.com",
        api_key: str = "",
        index_name: str = "",
        search_fields: List[str] = None,
        result_field: str = "tencent_search_results",
        timeout: float = 30.0,
        verbose: bool = False,
    ) -> None:
        """Initialize Tencent Cloud Search stage.
        
        Args:
            api_endpoint: Tencent Cloud Search API endpoint
                @zhs 腾讯云搜索API地址
            api_key: API key for authentication
                @zhs API密钥
            index_name: Name of the search index to query
                @zhs 搜索索引名称
            search_fields: Fields to search in (default: ['content'])
                @zhs 搜索字段列表
            result_field: Field name to store search results in paragraph.extdata
                @zhs 结果存储字段名
            timeout: Request timeout in seconds (default: 30.0)
                @zhs 请求超时时间（秒）
            verbose: Enable verbose logging (default: False)
                @zhs 启用详细日志
        """
        super().__init__(
            api_endpoint=api_endpoint,
            api_key=api_key,
            index_name=index_name,
            search_fields=search_fields,
            result_field=result_field,
            timeout=timeout,
            verbose=verbose,
        )
    
    async def _build_search_query(self, content: str) -> Dict[str, Any]:
        """Build search query from content.
        
        Args:
            content: Content to search for
            
        Returns:
            Search query dictionary
        """
        clean_content = re.sub(r'\s+', ' ', content).strip()
        
        query = {
            "query": {
                "simple_query": clean_content
            }
        }
        
        if self.search_fields:
            query["query"]["fields"] = self.search_fields
        
        return query
    
    async def _call_search_api(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Call Tencent Cloud Search API.
        
        Args:
            query: Search query dictionary
            
        Returns:
            API response dictionary
        """
        client = await self._ensure_client()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Request-Id": str(uuid.uuid4()),
        }
        
        body = {
            "index": self.index_name,
            "query": query["query"],
        }
        
        if "fields" in query.get("query", {}):
            body["query"]["fields"] = query["query"]["fields"]
        
        self._log(f"Calling API with query: {query}")
        
        response = await client.post(
            self.api_endpoint,
            json=body,
            headers=headers,
        )
        
        response.raise_for_status()
        return response.json()


class AzureCognitiveSearch(CloudSearchBase):
    """Azure Cognitive Search Integration
    @zhs 微软Azure认知搜索集成
    
    This pipeline stage calls Azure Cognitive Search API to search for content
    extracted from paragraph.content.
    """
    
    provider_name: str = "AzureCognitiveSearch"
    
    def __init__(
        self,
        api_endpoint: str = "",
        api_key: str = "",
        index_name: str = "",
        search_fields: List[str] = None,
        result_field: str = "azure_search_results",
        timeout: float = 30.0,
        verbose: bool = False,
    ) -> None:
        """Initialize Azure Cognitive Search stage.
        
        Args:
            api_endpoint: Azure Cognitive Search API endpoint
                Format: https://{account}.search.azure.net
                @zhs Azure搜索API地址
            api_key: API key for authentication
                @zhs API密钥
            index_name: Name of the search index to query
                @zhs 搜索索引名称
            search_fields: Fields to search in (default: ['content'])
                @zhs 搜索字段列表
            result_field: Field name to store search results in paragraph.extdata
                @zhs 结果存储字段名
            timeout: Request timeout in seconds (default: 30.0)
                @zhs 请求超时时间（秒）
            verbose: Enable verbose logging (default: False)
                @zhs 启用详细日志
        """
        super().__init__(
            api_endpoint=api_endpoint,
            api_key=api_key,
            index_name=index_name,
            search_fields=search_fields,
            result_field=result_field,
            timeout=timeout,
            verbose=verbose,
        )
    
    async def _build_search_query(self, content: str) -> Dict[str, Any]:
        """Build search query from content.
        
        Args:
            content: Content to search for
            
        Returns:
            Search query dictionary
        """
        clean_content = re.sub(r'\s+', ' ', content).strip()
        
        query = {
            "search": clean_content,
            "queryType": "simple",
            "top": 10,
            "count": True,
        }
        
        if self.search_fields:
            query["select"] = ",".join(self.search_fields)
        
        return query
    
    async def _call_search_api(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Call Azure Cognitive Search API.
        
        Args:
            query: Search query dictionary
            
        Returns:
            API response dictionary
        """
        client = await self._ensure_client()
        
        # Build full URL with index name
        url = f"{self.api_endpoint}/indexes/{self.index_name}/docs/search"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        
        self._log(f"Calling API at {url}")
        
        response = await client.post(
            url,
            json=query,
            headers=headers,
        )
        
        response.raise_for_status()
        return response.json()


class AWSCloudSearch(CloudSearchBase):
    """AWS CloudSearch Integration
    @zhs 亚马逊AWS云搜索集成
    
    This pipeline stage calls AWS CloudSearch API to search for content
    extracted from paragraph.content.
    """
    
    provider_name: str = "AWSCloudSearch"
    
    def __init__(
        self,
        api_endpoint: str = "",
        api_key: str = "",
        index_name: str = "",
        search_fields: List[str] = None,
        result_field: str = "aws_search_results",
        timeout: float = 30.0,
        verbose: bool = False,
    ) -> None:
        """Initialize AWS CloudSearch stage.
        
        Args:
            api_endpoint: AWS CloudSearch API endpoint
                Format: https://search-{domain}.{region}.cloudsearch.amazonaws.com
                @zhs AWS搜索API地址
            api_key: AWS Access Key ID for authentication
                @zhs AWS访问密钥ID
            index_name: Name of the search index to query (not used in AWS)
                @zhs 搜索索引名称
            search_fields: Fields to search in (default: ['content'])
                @zhs 搜索字段列表
            result_field: Field name to store search results in paragraph.extdata
                @zhs 结果存储字段名
            timeout: Request timeout in seconds (default: 30.0)
                @zhs 请求超时时间（秒）
            verbose: Enable verbose logging (default: False)
                @zhs 启用详细日志
        """
        super().__init__(
            api_endpoint=api_endpoint,
            api_key=api_key,
            index_name=index_name,
            search_fields=search_fields,
            result_field=result_field,
            timeout=timeout,
            verbose=verbose,
        )
        # AWS uses secret key for signing, stored in api_key as "ACCESS_KEY_ID:SECRET_ACCESS_KEY"
        self._parse_aws_credentials()
    
    def _parse_aws_credentials(self) -> None:
        """Parse AWS credentials from api_key field."""
        if self.api_key and ':' in self.api_key:
            parts = self.api_key.split(':', 1)
            self.aws_access_key = parts[0]
            self.aws_secret_key = parts[1]
        else:
            self.aws_access_key = self.api_key
            self.aws_secret_key = ""
    
    def _create_aws_signature(
        self,
        method: str,
        host: str,
        path: str,
        query_string: str,
        date: str,
    ) -> str:
        """Create AWS Signature Version 4.
        
        Args:
            method: HTTP method
            host: Host name
            path: Path
            query_string: Query string
            date: Date string
            
        Returns:
            Authorization header value
        """
        # AWS Signature Version 4 signing key
        def sign(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()
        
        # Service and region from host
        parts = host.split('.')
        service = parts[0].replace('search-', '')
        region = parts[1] if len(parts) > 1 else 'us-east-1'
        
        # Create canonical request
        canonical_uri = path
        canonical_querystring = query_string
        canonical_headers = f"host:{host}\n"
        signed_headers = "host"
        payload_hash = hashlib.sha256(''.encode('utf-8')).hexdigest()
        
        canonical_request = f"{method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        
        # Create string to sign
        algorithm = "AWS4-HMAC-SHA256"
        credential_scope = f"{date}/{region}/{service}/aws4_request"
        string_to_sign = f"{algorithm}\n{date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        
        # Calculate signature
        signing_key = sign(
            sign(
                sign(
                    sign(f"AWS4{self.aws_secret_key}".encode('utf-8'), date),
                    region,
                ),
                service,
            ),
            "aws4_request",
        )
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        return f"{algorithm} Credential={self.aws_access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
    
    async def _build_search_query(self, content: str) -> Dict[str, Any]:
        """Build search query from content.
        
        Args:
            content: Content to search for
            
        Returns:
            Query string for AWS CloudSearch
        """
        clean_content = re.sub(r'\s+', ' ', content).strip()
        
        # Build query string
        query_parts = [f"q={clean_content}"]
        query_parts.append("q.parser=simple")
        query_parts.append("size=10")
        query_parts.append("return=true")
        
        if self.search_fields:
            query_parts.append(f"return={','.join(self.search_fields)}")
        
        return "&".join(query_parts)
    
    async def _call_search_api(self, query: str) -> Dict[str, Any]:
        """Call AWS CloudSearch API.
        
        Args:
            query: Search query string
            
        Returns:
            API response dictionary
        """
        client = await self._ensure_client()
        
        # Parse endpoint to get host and path
        from urllib.parse import urlparse
        parsed = urlparse(self.api_endpoint)
        host = parsed.netloc
        path = parsed.path or "/2013-01-01/search"
        
        # Create date for signing
        now = datetime.now(timezone.utc)
        date = now.strftime("%Y%m%dT%H%M%SZ")
        short_date = now.strftime("%Y%m%d")
        
        # Create signature
        authorization = self._create_aws_signature(
            method="GET",
            host=host,
            path=path,
            query_string=query,
            date=date,
        )
        
        url = f"{self.api_endpoint}?{query}"
        
        headers = {
            "Host": host,
            "X-Amz-Date": date,
            "Authorization": authorization,
        }
        
        self._log(f"Calling API at {url}")
        
        response = await client.get(
            url,
            headers=headers,
        )
        
        response.raise_for_status()
        return response.json()


class CloudSearchPlugin(Plugin):
    """Plugin for Cloud Search integration.
    @zhs 云搜索插件
    """
    
    def __init__(self, pmanager, **conf) -> None:
        """Initialize cloud search plugin.
        
        Args:
            pmanager: Plugin manager instance
            **conf: Plugin configuration
        """
        super().__init__(pmanager, **conf)
        
        # Register pipeline stages
        self.register_pipelines([
            TencentCloudSearch,
            AzureCognitiveSearch,
            AWSCloudSearch,
        ])

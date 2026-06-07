"""MCP Server"""

from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastmcp import FastMCP
from sqlalchemy import select

from .models import QueryFilters, get_db_session, Dataset, Paragraph


class MCPServer:
    """MCP Server SSE"""

    def __init__(self):
        self.mcp = FastMCP("Jindai Tools for scholarly literature retrieval")
        
        @self.mcp.tool()
        async def query(
            query: str, datasets: List[str], embeddings: Optional[bool] = None, limit: int = 10
        ) -> str:
            """
            Query database with given keywords/text snippet
            When `embeddings` is true, query will be converted into embeddings.
            `dataset` should be given, which can be fetched by using `datasets` tool.
            """
            stmt = await Paragraph.build_query(QueryFilters(q=query, embeddings=embeddings, datasets=datasets, limit=limit))
            async with get_db_session() as session:
                res = await session.execute(stmt)
                return res.mappings().all()

        @self.mcp.tool()
        async def datasets() -> str:
            """Get all available datasets"""
            async with get_db_session() as session:
                res = await session.execute(select(Dataset))
                return res.mappings().all()

    def mount(self, app: FastAPI):
        """Register MCP routes"""
        mcp_app = self.mcp.http_app(path='/api/v2/mcp')
        app.include_router(mcp_app.router)
        return mcp_app.router.lifespan_context

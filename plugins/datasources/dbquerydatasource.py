"""Database Query Data Source

This module provides a data source implementation for querying existing paragraphs
from the database using various filter criteria.
"""

from typing import Iterable, Optional, Union

from jindai.models import Paragraph, QueryFilters, get_db_session
from jindai.pipeline import DataSourceStage


class DBQueryDataSource(DataSourceStage):
    """Query existing paragraphs from the database.
    
    This data source allows filtering paragraphs by various criteria including
    text content, language, date ranges, and grouping options. It supports
    pagination, sorting, and can return either Paragraph objects or raw dictionaries.
    
    Query Expression Format:
        - Simple keywords: Space-separated terms for text matching
        - Query expression: Starts with '?' for structured queries
        - Aggregation expression: Starts with '??' for aggregate queries
    
    Group Options:
        - none: No grouping (default)
        - group: Group by paragraph group
        - source: Group by source URL
        - both: Group by both group and source
    """

    def apply_params(
        self, 
        query: str = "", 
        limit: int = 0, 
        skip: int = 0, 
        sort: str = "",
        groups: str = ""
    ) -> None:
        """Configure the database query parameters.
        
        Args:
            query: Query expression or keywords. Can be:
                - Simple text keywords for matching
                - Expression starting with '?' for structured queries
                - Expression starting with '??' for aggregation queries
            limit: Maximum number of results to return. 0 means no limit.
            skip: Number of results to skip (for pagination).
            sort: Sorting expression (e.g., 'pdate DESC' or 'author ASC').
            groups: Grouping option. Choose from:
                - none: No grouping (default)
                - group: Group by paragraph group
                - source: Group by source URL
                - both: Group by both group and source
        """
        self.query = Paragraph.build_query(QueryFilters(
            q=query, 
            offset=skip, 
            limit=limit, 
            groupBy=groups, 
            sort=sort
        ))

    async def fetch(self) -> Iterable[Paragraph]:
        """Execute the configured database query.
        
        Yields:
            Paragraph objects matching the query criteria.
        """
        async with get_db_session() as session:
            result = await session.execute(self.query)
            for item in result:
                yield item

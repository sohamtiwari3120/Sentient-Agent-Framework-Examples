from tavily import AsyncTavilyClient
import logging

logger = logging.getLogger(__name__)

class SearchProvider:
    def __init__(
            self,
            api_key: str
    ):
        self.client = AsyncTavilyClient(api_key=api_key)


    async def search(
            self,
            query: str
    ) -> dict:
        results = await self.client.search(query)
        return results

    async def search_linkedin(self, query: str) -> dict:
        """Search LinkedIn for information."""
        try:
            # Use the Tavily API to search LinkedIn
            response = await self.client.get(
                "https://api.tavily.com/v1/search",
                params={
                    "query": query,
                    "engine": "linkedin",
                    "num_results": 5
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error searching LinkedIn: {str(e)}")
            return {} 
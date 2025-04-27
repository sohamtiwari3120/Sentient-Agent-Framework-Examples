import logging
import os
from dotenv import load_dotenv
from src.search_agent.providers.model_provider import ModelProvider
from src.search_agent.providers.search_provider import SearchProvider
from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler)
from typing import AsyncIterator


load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SearchAgent(AbstractAgent):
    def __init__(
            self,
            name: str
    ):
        super().__init__(name)

        model_api_key = os.getenv("MODEL_API_KEY")
        if not model_api_key:
            raise ValueError("MODEL_API_KEY is not set")
        self._model_provider = ModelProvider(api_key=model_api_key)

        search_api_key = os.getenv("TAVILY_API_KEY")
        if not search_api_key:
            raise ValueError("TAVILY_API_KEY is not set") 
        self._search_provider = SearchProvider(api_key=search_api_key)


    # Implement the assist method as required by the AbstractAgent class
    async def assist(
            self,
            session: Session,
            query: Query,
            response_handler: ResponseHandler
    ):
        """Search the internet for information."""
        # Search for information
        await response_handler.emit_text_block(
            "SEARCH", "Searching internet for results..."
        )
        # search_results = await self._search_provider.search(query.prompt)
        search_results: dict = {
            "results": [],
            "images": []
        }
        if len(search_results["results"]) > 0:
            # Use response handler to emit JSON to the client
            await response_handler.emit_json(
                "SOURCES", {"results": search_results["results"]}
            )
        if len(search_results["images"]) > 0:
            # Use response handler to emit JSON to the client
            await response_handler.emit_json(
                "IMAGES", {"images": search_results["images"]}
            )

        # Process search results
        # Use response handler to create a text stream to stream the final 
        # response to the client
        final_response_stream = response_handler.create_text_stream(
            "FINAL_RESPONSE"
            )
        final_response_str = []
        async for chunk in self.__process_search_results(query.prompt, search_results["results"]):
            # Use the text stream to emit chunks of the final response to the client
            await final_response_stream.emit_chunk(chunk)
            final_response_str.append(chunk)
        
        with open('final_response.txt', 'w') as f:
            f.write(''.join(final_response_str))
        # Mark the text stream as complete
        await final_response_stream.complete()
        # Mark the response as complete
        await response_handler.complete()
    

    async def __process_search_results(
            self,
            prompt: str,
            search_results: dict
    ) -> AsyncIterator[str]:
        """Process the search results."""
        # process_search_results_query = f"Summarise the provided search results and use them to answer the provided prompt. Prompt: {prompt}. Search results: {search_results}"
        # process_search_results_query = f"Given the following resume, write a roast as if you're reviewing it in a brutal way. Don't hold back. The tone should be funny, savage, colorful, emotional, and fast-paced. Mention and joke about specific elements from the resume. Ensure it is highly personalized to the resumes unique components. Use vivid metaphors, cultural references, and exaggerations to make it more entertaining. Then finish with a 🍗 Roast-o-Meter: X/5 Sizzler: Compare the resume to a cultural moment, viral trend, or funny disaster using vivid language, playful exaggeration, and strong imagery. Limit this to one vivid, punchy sentence. Return output in a paragraph structure. Resume:{prompt}"
        with open('resume_update_prompt.txt', 'r') as f:
            process_search_results_query = f.read() + f"\n\nResume: {prompt}"
        async for chunk in self._model_provider.query_stream(process_search_results_query):
            yield chunk


if __name__ == "__main__":
    # Create an instance of a SearchAgent
    agent = SearchAgent(name="Search Agent")
    # Create a server to handle requests to the agent
    server = DefaultServer(agent)
    # Run the server
    server.run()
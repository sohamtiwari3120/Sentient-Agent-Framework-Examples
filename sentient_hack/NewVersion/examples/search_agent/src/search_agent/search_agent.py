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
from composio_openai import ComposioToolSet, Action


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

        # Initialize Composio ToolSet
        composio_api_key = os.getenv("COMPOSIO_API_KEY")
        if not composio_api_key:
            raise ValueError("COMPOSIO_API_KEY is not set")
        self._toolset = ComposioToolSet()

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
        try:
            # 1. Get the base prompt from resume_update_prompt.txt
            prompt_file = os.path.join(os.path.dirname(__file__), 'resume_update_prompt.txt')
            with open(prompt_file, 'r') as f:
                base_prompt = f.read()

            # 2. Get the resume from Google Docs using Composio
            try:
                # (Optional) Check connected tools (good habit)
                tools = self._toolset.get_tools(
                    actions=["GOOGLEDOCS_GET_DOCUMENT_BY_ID"],
                    check_connected_accounts=True
                )

                if not tools:
                    raise Exception("No Google Docs tool found. Check your account connection.")

                # Execute the action by name, passing parameters as a positional dict
                result = await self._toolset.execute_action(
                    "GOOGLEDOCS_GET_DOCUMENT_BY_ID",
                    {"id": "1QrcpKg2CkBfpmJt4l-xoUzLgejpaECSE"}
                )

                print("#########", result)

                # Parse result safely
                if isinstance(result, dict):
                    if result.get("successful") and "data" in result and "content" in result["data"]:
                        resume_content = result["data"]["content"]
                    else:
                        raise Exception(f"Unexpected response format: {result}")
                else:
                    raise Exception(f"Result is not a dictionary: {result}")

                if not resume_content:
                    raise Exception("No content found in document.")

            except Exception as e:
                logger.error(f"Failed to get resume from Google Docs: {str(e)}")
                resume_content = "Failed to retrieve resume content"
            # 3. Combine everything: base prompt + resume + job description
            process_search_results_query = f"{base_prompt}\n\nResume: {resume_content}\n\nJob Description: {prompt}"

        except Exception as e:
            logger.error(f"Error processing search results: {str(e)}")
            # Fallback to just using the prompt if something goes wrong
            process_search_results_query = f"{base_prompt}\n\nResume: {prompt}"

        async for chunk in self._model_provider.query_stream(process_search_results_query):
            yield chunk


if __name__ == "__main__":
    # Create an instance of a SearchAgent
    agent = SearchAgent(name="Search Agent")
    # Create a server to handle requests to the agent
    server = DefaultServer(agent)
    # Run the server
    server.run()
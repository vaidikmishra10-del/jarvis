import logging

from langchain_community.tools import DuckDuckGoSearchRun
from livekit.agents import RunContext, function_tool

logger = logging.getLogger("agent")


@function_tool
async def search_web(context: RunContext, query: str):
    """Use this tool to search the web for current information, news, facts, or anything not known.

    Args:
        query: The search query to look up on the web.
    """
    try:
        result = DuckDuckGoSearchRun().run(tool_input=query)
        logging.info(f"Web search result for query '{query}': {result}")
        return result
    except Exception as e:
        logging.error(
            f"Error occurred while searching the web for query '{query}': {e}"
        )
        raise

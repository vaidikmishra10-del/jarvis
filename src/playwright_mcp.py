import os

from livekit.agents import mcp
from livekit.agents.llm import ToolError
from typing_extensions import Self

_PLAYWRIGHT_ALLOWLIST = {
    # Excludes tools whose names collide with BrowserToolset (browser_click,
    # browser_fill, browser_back, browser_forward, browser_reload,
    # browser_get_text, browser_get_links).  Tool names MUST be unique across
    # all toolsets in a session — duplicates cause a ValueError at ToolContext
    # flatten time and silently prevent ALL tools from reaching the LLM.
    "browser_navigate",
    "browser_click_press",
    "browser_type",
    "browser_type_paste",
    "browser_hover",
    "browser_press_key",
    "browser_select_option",
    "browser_get_url",
    "browser_get_snapshot",
    "browser_execute",
    "browser_wait_for",
    "browser_goto_search",
}

MAX_RESULT_CHARS = 4000


def _compact_result(ctx: mcp.MCPToolResultContext) -> str:
    parts = []
    for item in ctx.result.content:
        if item.type == "text":
            parts.append(item.text)
        elif getattr(item, "type", None) == "image":
            data = getattr(item, "data", b"") or b""
            parts.append(f"[image of {len(data)} bytes omitted]")
    text = "\n".join(parts).strip()
    if not text:
        raise ToolError(
            f"Tool '{ctx.tool_name}' produced no readable output. "
            "State that clearly and try a different action."
        )
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "\n...[truncated]"
    return text


class PlaywrightMCPToolset(mcp.MCPToolset):
    """Full Playwright browser surface exposed through the official MCP server.

    Runs ``@playwright/mcp`` over stdio via npx and exposes a voice-safe subset
    of its tools. Results are compacted to text so snapshots never flood the
    conversation. Requires Node.js (and Chromium) in the deployment image.
    """

    def __init__(self) -> None:
        env = dict(os.environ)
        if browsers_path := os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
            env["PLAYWRIGHT_BROWSERS_PATH"] = browsers_path
        super().__init__(
            id="playwright-mcp",
            mcp_server=mcp.MCPServerStdio(
                command="npx",
                args=["-y", "@playwright/mcp", "--headless"],
                env=env,
                client_session_timeout_seconds=120,
                tool_result_resolver=_compact_result,
            ),
        )

    async def setup(self, *, reload: bool = False) -> Self:
        await super().setup(reload=reload)
        self.filter_tools(lambda tool: tool.id in _PLAYWRIGHT_ALLOWLIST)
        return self

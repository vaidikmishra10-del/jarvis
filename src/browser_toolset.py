import asyncio
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import ToolError, Toolset
from playwright.async_api import (
    Browser,
    BrowserContext,
    Locator,
    Page,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from typing_extensions import Self

logger = logging.getLogger("agent.browser")

DEFAULT_TIMEOUT_MS = 20_000
NAV_TIMEOUT_MS = 45_000
TEXT_CHAR_LIMIT = 3000
LINK_LIMIT = 25
HEADING_LIMIT = 15


def _clean(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, limit: int = TEXT_CHAR_LIMIT) -> str:
    text = _clean(text)
    if len(text) > limit:
        return f"{text[:limit]}..."
    return text


def _require_http_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ToolError(
            f"'{url}' is not a valid web address. "
            "Provide a full URL starting with http:// or https://."
        )
    return url


class BrowserToolset(Toolset):
    """One shared headless browser for the whole voice session.

    The browser is launched on ``setup()`` and closed on ``aclose()``. Every tool
    returns compact, spoken-friendly text (never raw HTML). Concurrency is
    serialized per tab so simultaneous tool calls cannot fight over the page.
    """

    def __init__(self) -> None:
        super().__init__(id="browser_toolset")
        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._lock = asyncio.Lock()
        self._headless = os.environ.get("JARVIS_BROWSER_HEADLESS", "1") != "0"
        self._user_agent = os.environ.get("JARVIS_BROWSER_UA") or None
        self._shots_dir = Path(
            os.environ.get("JARVIS_BROWSER_SHOTS_DIR", "browser_shots")
        )

    async def setup(self) -> Self:
        await super().setup()
        from playwright.async_api import async_playwright

        try:
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(
                headless=self._headless,
                args=["--no-sandbox"],
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 800},
                locale="en-US",
                user_agent=self._user_agent,
                ignore_https_errors=True,
            )
            self._page = await self._context.new_page()
            self._page.set_default_timeout(DEFAULT_TIMEOUT_MS)
        except BaseException as e:
            logger.exception("Failed to launch browser")
            await self.aclose()
            raise ToolError(f"The browser could not be started: {e}") from e
        return self

    async def aclose(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                logger.exception("Error closing browser")
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                logger.exception("Error stopping playwright")
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        await super().aclose()

    def _require_page(self) -> Page:
        if self._page is None:
            raise ToolError(
                "The browser is not ready yet. Please try again in a moment."
            )
        return self._page

    async def _input_field(self, page: Page, label: str) -> Locator:
        by_label = page.get_by_label(label)
        if await by_label.count() > 0:
            return by_label.first
        by_placeholder = page.get_by_placeholder(label)
        if await by_placeholder.count() > 0:
            return by_placeholder.first
        raise ToolError(
            f"Could not find an input field labelled '{label}' on this page. "
            "Inspect the page with browser_get_text or ask the user for the visible label."
        )

    async def _navigate(self, page: Page, url: str) -> float:
        start = time.monotonic()
        await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        return time.monotonic() - start

    async def _tab_summary(self, page: Page) -> str:
        return f"Page: {page.url}\nTitle: {_clean(await page.title())}"

    @function_tool()
    async def browser_open(self, context: RunContext, url: str) -> str:
        """Open a web address in the browser and load its page.

        Use this to fetch a specific site, article, or any URL the user mentions.
        Returns the page title and a short summary of the main text.

        Args:
            url: The full web address to open, for example https://news.ycombinator.com.
        """
        url = _require_http_url(url)
        page = self._require_page()
        async with self._lock:
            host = urlparse(url).netloc
            await context.update(f"Opening {host} now.")
            async with context.with_filler(
                "The page is loading, one moment please.", delay=5
            ):
                await self._navigate(page, url)
            title = _clean(await page.title())
            body = await page.locator("body").inner_text()
            return f"Opened {url}\nTitle: {title}\n\nPage content:\n{_truncate(body)}"

    @function_tool()
    async def browser_status(self, context: RunContext) -> str:
        """Report which page the browser is currently on and its title.

        Use this before interacting with a page to confirm the current location.
        """
        async with self._lock:
            page = self._require_page()
            return f"Currently on {page.url}\nTitle: {_clean(await page.title())}"

    @function_tool()
    async def browser_get_text(self, context: RunContext, max_chars: int = 1500) -> str:
        """Read the visible text of the current page and return it.

        Use this to understand what is on a page. Larger pages are truncated.

        Args:
            max_chars: Maximum number of characters to return, between 200 and 8000.
        """
        limit = max(200, min(int(max_chars), 8000))
        async with self._lock:
            page = self._require_page()
            body = await page.locator("body").inner_text()
            return _truncate(body, limit)

    @function_tool()
    async def browser_get_links(self, context: RunContext, limit: int = 20) -> str:
        """List the links on the current page with their destinations.

        Use this to find clickable destinations before using browser_click.

        Args:
            limit: Maximum number of links to return, between 1 and 50.
        """
        limit = max(1, min(int(limit), 50))
        async with self._lock:
            page = self._require_page()
            links = await page.locator("a").evaluate_all(
                "els => els.map(e => ({"
                "t: (e.innerText || e.getAttribute('aria-label') || '').trim(),"
                "h: e.href"
                "})).filter(l => l.t || l.h)"
            )
        seen = set()
        rows = []
        for link in links:
            text = _clean(link.get("t") or "")
            href = (link.get("h") or "").strip()
            if not href or (text, href) in seen:
                continue
            seen.add((text, href))
            rows.append(f"- {text or href}: {href}")
            if len(rows) >= limit:
                break
        if not rows:
            return "No links found on this page."
        return "Links on this page:\n" + "\n".join(rows)

    @function_tool()
    async def browser_get_headings(self, context: RunContext, limit: int = 15) -> str:
        """Return the page's heading outline (h1, h2, h3) to grasp its structure.

        Args:
            limit: Maximum number of headings to return, between 1 and 30.
        """
        limit = max(1, min(int(limit), 30))
        async with self._lock:
            page = self._require_page()
            headings = await page.locator("h1, h2, h3").evaluate_all(
                "els => els.map(e => e.tagName + ': ' + e.innerText.trim()).filter(s => s.length > 2)"
            )
        rows = [_clean(h) for h in headings[:limit]]
        if not rows:
            return "No headings found on this page."
        return "Page headings:\n" + "\n".join(rows)

    @function_tool()
    async def browser_click(self, context: RunContext, label: str) -> str:
        """Click an element on the current page by its visible text label.

        Use after reading the page with browser_get_text or browser_get_links.
        Examples: a button labelled 'Sign in', or a link labelled 'Read more'.

        Args:
            label: The visible text of the element to click.
        """
        async with self._lock:
            page = self._require_page()
            target = page.get_by_text(label, exact=False).first
            try:
                await target.click(timeout=10_000)
            except PlaywrightTimeoutError:
                raise ToolError(
                    f"Could not find anything to click labelled '{label}' on this page. "
                    "Check the visible text with browser_get_text, then try again."
                ) from None
            return f"Clicked '{label}'.\n{await self._tab_summary(page)}"

    @function_tool()
    async def browser_fill(self, context: RunContext, label: str, value: str) -> str:
        """Type text into an input field on the current page.

        Use for search boxes, login forms, and any text input. The label is the
        field's visible label or placeholder text.

        Args:
            label: The field's visible label or placeholder, for example 'Email' or 'Enter your name'.
            value: The text to type into the field.
        """
        async with self._lock:
            page = self._require_page()
            field = await self._input_field(page, label)
            await field.fill(value)
            return f"Filled '{label}' with '{value}'."

    @function_tool()
    async def browser_select(self, context: RunContext, label: str, value: str) -> str:
        """Choose an option in a dropdown/select menu on the current page.

        Args:
            label: The dropdown's visible label, for example 'Country'.
            value: The option value or visible text to select.
        """
        async with self._lock:
            page = self._require_page()
            field = await self._input_field(page, label)
            await field.select_option(value)
            return f"Selected '{value}' in '{label}'."

    @function_tool()
    async def browser_press(
        self, context: RunContext, key: str, label: str | None = None
    ) -> str:
        """Press a keyboard key, optionally inside a specific field.

        Useful keys: Enter, Escape, Tab, ArrowDown, ArrowUp, Backspace.
        Example: press Enter after typing in a search box to submit it.

        Args:
            key: The keyboard key to press, for example 'Enter' or 'Escape'.
            label: Optional field label or placeholder to focus before pressing the key.
        """
        async with self._lock:
            page = self._require_page()
            if label:
                field = await self._input_field(page, label)
                await field.press(key)
            else:
                await page.keyboard.press(key)
            return f"Pressed {key}.\n{await self._tab_summary(page)}"

    @function_tool()
    async def browser_back(self, context: RunContext) -> str:
        """Go back one page in the browser history."""
        async with self._lock:
            page = self._require_page()
            went = await page.go_back()
            if went is None:
                raise ToolError("There is no previous page in the history.")
            return f"Went back.\n{await self._tab_summary(page)}"

    @function_tool()
    async def browser_forward(self, context: RunContext) -> str:
        """Go forward one page in the browser history."""
        async with self._lock:
            page = self._require_page()
            went = await page.go_forward()
            if went is None:
                raise ToolError("There is no next page in the history.")
            return f"Went forward.\n{await self._tab_summary(page)}"

    @function_tool()
    async def browser_reload(self, context: RunContext) -> str:
        """Reload the current page to fetch the latest version."""
        async with self._lock:
            page = self._require_page()
            await page.reload(timeout=NAV_TIMEOUT_MS)
            return f"Reloaded.\n{await self._tab_summary(page)}"

    @function_tool()
    async def browser_scroll(
        self, context: RunContext, direction: str = "down", steps: int = 1
    ) -> str:
        """Scroll the current page up or down a few screen-lengths.

        Args:
            direction: 'down' or 'up'.
            steps: How many screen-lengths to scroll, between 1 and 10.
        """
        steps = max(1, min(int(steps), 10))
        async with self._lock:
            page = self._require_page()
            delta = 700 * steps if direction.strip().lower() != "up" else -700 * steps
            await page.mouse.wheel(0, delta)
            return f"Scrolled {direction} {steps} step(s) on {page.url}."

    @function_tool()
    async def browser_screenshot(self, context: RunContext, name: str = "shot") -> str:
        """Save a screenshot of the current page to the shots folder.

        Returns the file path so it can be shared later. Use sparingly.

        Args:
            name: A short name for the screenshot, for example 'homepage'.
        """
        async with self._lock:
            page = self._require_page()
            self._shots_dir.mkdir(parents=True, exist_ok=True)
            path = self._shots_dir / f"{name.strip() or 'shot'}_{int(time.time())}.png"
            data = await page.screenshot(type="png")
            path.write_bytes(data)
            return f"Saved screenshot to {path.resolve()} ({len(data)} bytes)."

    @function_tool()
    async def browser_new_tab(self, context: RunContext, url: str) -> str:
        """Open a new browser tab and switch to it.

        Args:
            url: The full web address to open in the new tab.
        """
        url = _require_http_url(url)
        async with self._lock:
            if self._context is None:
                raise ToolError(
                    "The browser is not ready yet. Please try again in a moment."
                )
            page = await self._context.new_page()
            page.set_default_timeout(DEFAULT_TIMEOUT_MS)
            await self._navigate(page, url)
            self._page = page
            return f"Opened a new tab.\n{await self._tab_summary(page)}"

    @function_tool()
    async def browser_switch_tab(self, context: RunContext, index: int) -> str:
        """Switch to another open tab by its number (first tab is 1).

        Args:
            index: Tab number to switch to, starting at 1.
        """
        async with self._lock:
            if self._context is None:
                raise ToolError(
                    "The browser is not ready yet. Please try again in a moment."
                )
            pages = self._context.pages
            if not pages:
                raise ToolError("The browser has no open tabs.")
            try:
                page = pages[int(index) - 1]
            except (IndexError, ValueError):
                raise ToolError(
                    f"Tab {index} does not exist. There are {len(pages)} open tabs."
                ) from None
            self._page = page
            return f"Switched to tab {index}.\n{await self._tab_summary(page)}"

    @function_tool()
    async def browser_close_tab(self, context: RunContext, index: int = 0) -> str:
        """Close a browser tab (first tab is 1; 0 closes the current tab).

        Keeps at least one tab open.

        Args:
            index: Tab number to close, starting at 1. Omit to close the current tab.
        """
        async with self._lock:
            if self._context is None:
                raise ToolError(
                    "The browser is not ready yet. Please try again in a moment."
                )
            pages = self._context.pages
            if not pages:
                raise ToolError("The browser has no open tabs.")
            idx = int(index)
            if idx < 0:
                raise ToolError("Tab index must be 0 or a positive number.")
            if idx > len(pages):
                raise ToolError(
                    f"Tab {idx} does not exist. There are {len(pages)} open tabs."
                )
            closing = pages[idx - 1] if idx > 0 else self._page
            if closing is None:
                raise ToolError("There is no current tab to close.")
            if len(pages) <= 1:
                raise ToolError("Cannot close the last remaining tab.")
            await closing.close()
            if self._page is closing:
                self._page = pages[0] if self._page not in pages else self._page
            return f"Closed a tab. {len(self._context.pages)} tab(s) remain."

    @function_tool()
    async def browser_search(self, context: RunContext, query: str) -> str:
        """Search the web for information and return the top results.

        Use this when the user asks to look something up, compare options, or find
        recent news. Returns up to 8 results with title, link, and snippet.

        Args:
            query: The search query, for example 'best espresso machines 2026'.
        """
        query = query.strip()
        if not query:
            raise ToolError("The search query cannot be empty.")
        page = self._require_page()
        async with self._lock:
            url = "https://html.duckduckgo.com/html/?q=" + urlencode({"q": query})
            await context.update(f"Searching the web for '{query}'.")
            async with context.with_filler(
                "Still searching, one moment please.", delay=5
            ):
                await self._navigate(page, url)
            results = await page.locator(".result").evaluate_all(
                "els => els.slice(0, 8).map(e => {"
                "const t = e.querySelector('.result__a');"
                "const s = e.querySelector('.result__snippet');"
                "const u = e.querySelector('.result__url');"
                "return {t: t ? t.innerText.trim() : '', s: s ? s.innerText.trim() : '',"
                "u: u ? u.innerText.trim() : ''};"
                "}).filter(r => r.t)"
            )
        if not results:
            return (
                f"No results found for '{query}'. The page may be showing a captcha; "
                "suggest using openai or try a more specific query."
            )
        rows = [
            f"{i + 1}. {_clean(r['t'])} — {r['u']}\n   {_clean(r['s'])[:300]}"
            for i, r in enumerate(results)
        ]
        return f"Search results for '{query}':\n" + "\n".join(rows)

    @function_tool()
    async def browser_evaluate(self, context: RunContext, script: str) -> str:
        """Run JavaScript in the current page and return the result.

        Use for advanced tasks the other tools cannot do, such as reading an
        element's attribute or extracting a computed value.

        Args:
            script: A self-contained JavaScript expression that evaluates to a value.
        """
        async with self._lock:
            page = self._require_page()
            try:
                result = await page.evaluate(script)
            except PlaywrightTimeoutError:
                raise ToolError("The script timed out on the page.") from None
            return _truncate(str(result), 2000)

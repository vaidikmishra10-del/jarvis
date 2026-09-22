import asyncio
import http.server
import threading

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from browser_toolset import BrowserToolset

TOOL_IDS = {
    "browser_open",
    "browser_status",
    "browser_get_text",
    "browser_get_links",
    "browser_get_headings",
    "browser_click",
    "browser_fill",
    "browser_select",
    "browser_press",
    "browser_back",
    "browser_forward",
    "browser_reload",
    "browser_scroll",
    "browser_screenshot",
    "browser_new_tab",
    "browser_switch_tab",
    "browser_close_tab",
    "browser_search",
    "browser_evaluate",
}


class _NullAsyncCM:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeRunContext:
    def __init__(self) -> None:
        self.updates = []

    async def update(self, message: str) -> None:
        self.updates.append(message)

    def with_filler(self, *args, **kwargs):
        return _NullAsyncCM()


class FakeKeyboard:
    def __init__(self, page) -> None:
        self.page = page

    async def press(self, key: str) -> None:
        self.page.pressed_keys.append(key)


class FakeMouse:
    def __init__(self, page) -> None:
        self.page = page

    async def wheel(self, dx: int, dy: int) -> None:
        self.page.scrolls += 1
        self.page.last_scroll_delta = dy


class FakeLocator:
    def __init__(self, page, key: str) -> None:
        self.page = page
        self.key = key
        self.first = self

    async def inner_text(self) -> str:
        if self.key == "body":
            return self.page.body_text
        return ""

    async def evaluate_all(self, script: str):
        if self.key == "a":
            return [{"t": text, "h": href} for text, href in self.page.links]
        if self.key == "h1, h2, h3":
            return list(self.page.headings)
        return []

    async def count(self) -> int:
        if self.key == "label":
            return 1 if self.page.has_label else 0
        if self.key == "placeholder":
            return 1 if self.page.has_placeholder else 0
        return 0

    async def click(self, timeout: int = 0) -> None:
        if self.page.missing_click:
            raise PlaywrightTimeoutError("click timed out")
        self.page.clicks.append(self.page.latest_label)

    async def fill(self, value: str, **kwargs) -> None:
        self.page.filled.append((self.page.latest_label, value))

    async def select_option(self, value: str, **kwargs) -> None:
        self.page.selected.append((self.page.latest_label, value))

    async def press(self, key: str) -> None:
        self.page.pressed_keys.append(key)

    async def hover(self) -> None:
        pass


class FakeContext:
    def __init__(self, pages) -> None:
        self.pages = pages

    async def new_page(self):
        page = FakePage()
        self.pages.append(page)
        return page


class FakePage:
    def __init__(self) -> None:
        self.url = "about:blank"
        self.title_text = "Test Title"
        self.body_text = "Hello  world,   this is a test page."
        self.links = [("Example", "https://example.com/")]
        self.headings = ["H1: Title here", "H2: Section one"]
        self.clicks = []
        self.filled = []
        self.selected = []
        self.pressed_keys = []
        self.scrolls = 0
        self.last_scroll_delta = 0
        self.nav_log = []
        self.has_label = True
        self.has_placeholder = True
        self.missing_click = False
        self.goto_sleep = 0.0
        self.active = 0
        self.max_active = 0
        self.screenshot_bytes = b"PNG-fake"
        self.back_value = None
        self.forward_value = None
        self.timeout = 0
        self.latest_label = ""

    def set_default_timeout(self, ms: int) -> None:
        self.timeout = ms

    async def title(self) -> str:
        return self.title_text

    async def goto(self, url: str, **kwargs) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.goto_sleep:
            await asyncio.sleep(self.goto_sleep)
        self.url = url
        self.nav_log.append(url)
        self.active -= 1

    async def go_back(self):
        if self.back_value is None:
            return None
        self.url = "back"
        return self.back_value

    async def go_forward(self):
        if self.forward_value is None:
            return None
        self.url = "forward"
        return self.forward_value

    async def reload(self, **kwargs) -> None:
        self.nav_log.append("RELOAD")

    async def screenshot(self, **kwargs) -> bytes:
        return self.screenshot_bytes

    async def evaluate(self, script: str):
        return "evaluated result"

    def locator(self, selector: str):
        return FakeLocator(self, selector)

    def get_by_text(self, text: str, exact: bool = False):
        self.latest_label = text
        return FakeLocator(self, "text")

    def get_by_label(self, text: str):
        self.latest_label = text
        return FakeLocator(self, "label")

    def get_by_placeholder(self, text: str):
        self.latest_label = text
        return FakeLocator(self, "placeholder")

    @property
    def keyboard(self):
        return FakeKeyboard(self)

    @property
    def mouse(self):
        return FakeMouse(self)


def _toolset(page: FakePage | None = None) -> BrowserToolset:
    ts = BrowserToolset()
    if page is not None:
        ts._page = page
    return ts


def test_all_tools_registered_with_descriptions() -> None:
    ts = _toolset()
    ids = {tool.info.name for tool in ts.tools}
    assert ids == TOOL_IDS
    for tool in ts.tools:
        assert tool.info.name in TOOL_IDS
        assert tool.info.description, f"{tool.info.name} has no description"


@pytest.mark.asyncio
async def test_browser_status_reports_url_and_title() -> None:
    ts = _toolset(FakePage())
    result = await ts.browser_status(FakeRunContext())
    assert "about:blank" in result
    assert "Test Title" in result


@pytest.mark.asyncio
async def test_browser_not_ready_raises_tool_error() -> None:
    ts = _toolset()
    with pytest.raises(Exception) as excinfo:
        await ts.browser_status(FakeRunContext())
    assert "not ready" in str(excinfo.value)


@pytest.mark.asyncio
async def test_browser_open_navigates_and_returns_summary(monkeypatch) -> None:
    page = FakePage()
    page.body_text = "  Some\nlong   body   content  "
    ts = _toolset(page)
    result = await ts.browser_open(FakeRunContext(), "https://example.com")
    assert page.nav_log == ["https://example.com"]
    assert "Test Title" in result
    assert "Some long body content" in result


@pytest.mark.asyncio
async def test_browser_open_rejects_non_http_url() -> None:
    ts = _toolset(FakePage())
    with pytest.raises(Exception) as excinfo:
        await ts.browser_open(FakeRunContext(), "javascript:alert(1)")
    assert "valid web address" in str(excinfo.value)


@pytest.mark.asyncio
async def test_browser_get_text_cleans_and_truncates(monkeypatch) -> None:
    page = FakePage()
    page.body_text = "  a   b  c d    e f  " + "word " * 500
    ts = _toolset(page)
    result = await ts.browser_get_text(FakeRunContext(), max_chars=200)
    assert len(result) <= 210
    assert "..." in result
    assert "  " not in result.replace("...", "")


@pytest.mark.asyncio
async def test_browser_get_links_lists_unique_links() -> None:
    page = FakePage()
    page.links = [
        ("Example", "https://example.com/"),
        ("Example", "https://example.com/"),
        ("", "https://orphan.com/"),
    ]
    ts = _toolset(page)
    result = await ts.browser_get_links(FakeRunContext(), limit=10)
    assert result.count("Example") == 1
    assert "https://orphan.com/" in result


@pytest.mark.asyncio
async def test_browser_click_succeeds_and_missing_click_raises() -> None:
    page = FakePage()
    ts = _toolset(page)
    result = await ts.browser_click(FakeRunContext(), "Sign in")
    assert page.clicks == ["Sign in"]
    assert "Clicked" in result

    page.missing_click = True
    with pytest.raises(Exception) as excinfo:
        await ts.browser_click(FakeRunContext(), "Sign in")
    assert "Could not find" in str(excinfo.value)


@pytest.mark.asyncio
async def test_browser_fill_prefers_label_then_placeholder() -> None:
    page = FakePage()
    ts = _toolset(page)
    await ts.browser_fill(FakeRunContext(), "Email", "a@b.com")
    assert ("Email", "a@b.com") in page.filled

    page.has_label = False
    page.has_placeholder = True
    await ts.browser_select(FakeRunContext(), "Country", "India")
    assert ("Country", "India") in page.selected

    page.has_label = False
    page.has_placeholder = False
    with pytest.raises(Exception) as excinfo:
        await ts.browser_fill(FakeRunContext(), "Email", "a@b.com")
    assert "Could not find an input field" in str(excinfo.value)


@pytest.mark.asyncio
async def test_browser_screenshot_writes_png(tmp_path) -> None:
    ts = _toolset(FakePage())
    ts._shots_dir = tmp_path
    result = await ts.browser_screenshot(FakeRunContext(), "home")
    assert "Saved screenshot" in result
    files = list(tmp_path.glob("home_*.png"))
    assert len(files) == 1
    assert files[0].read_bytes() == b"PNG-fake"


@pytest.mark.asyncio
async def test_browser_press_key_and_scroll() -> None:
    page = FakePage()
    ts = _toolset(page)
    await ts.browser_press(FakeRunContext(), "Enter")
    await ts.browser_scroll(FakeRunContext(), "down", steps=2)
    assert page.pressed_keys == ["Enter"]
    assert page.scrolls == 1
    assert page.last_scroll_delta == 1400


@pytest.mark.asyncio
async def test_browser_back_forward_without_history_raise() -> None:
    page = FakePage()
    ts = _toolset(page)
    with pytest.raises(Exception) as excinfo:
        await ts.browser_back(FakeRunContext())
    assert "no previous page" in str(excinfo.value)
    with pytest.raises(Exception) as excinfo:
        await ts.browser_forward(FakeRunContext())
    assert "no next page" in str(excinfo.value)


@pytest.mark.asyncio
async def test_browser_history_works() -> None:
    page = FakePage()
    page.back_value = object()
    page.forward_value = object()
    ts = _toolset(page)
    result = await ts.browser_back(FakeRunContext())
    assert "Went back" in result
    result = await ts.browser_forward(FakeRunContext())
    assert "Went forward" in result


@pytest.mark.asyncio
async def test_browser_tabs_switch_and_close() -> None:
    first = FakePage()
    second = FakePage()
    ts = _toolset(first)
    ts._context = FakeContext([first, second])
    await ts.browser_switch_tab(FakeRunContext(), 2)
    assert ts._page is second
    with pytest.raises(Exception) as excinfo:
        await ts.browser_switch_tab(FakeRunContext(), 9)
    assert "does not exist" in str(excinfo.value)
    with pytest.raises(Exception) as excinfo:
        await ts.browser_close_tab(FakeRunContext(), len(ts._context.pages) + 1)
    assert "does not exist" in str(excinfo.value)


@pytest.mark.asyncio
async def test_browser_close_last_tab_refused() -> None:
    page = FakePage()
    ts = _toolset(page)
    ts._context = FakeContext([page])
    with pytest.raises(Exception) as excinfo:
        await ts.browser_close_tab(FakeRunContext())
    assert "last remaining tab" in str(excinfo.value)


@pytest.mark.asyncio
async def test_browser_search_requires_query() -> None:
    ts = _toolset(FakePage())
    with pytest.raises(Exception) as excinfo:
        await ts.browser_search(FakeRunContext(), "   ")
    assert "cannot be empty" in str(excinfo.value)


@pytest.mark.asyncio
async def test_browser_open_serializes_concurrent_calls() -> None:
    page = FakePage()
    page.goto_sleep = 0.05
    ts = _toolset(page)
    ctx = FakeRunContext()

    async def run():
        return await ts.browser_open(ctx, "https://a.com")

    await asyncio.gather(run(), run())
    assert page.max_active == 1
    assert len(page.nav_log) == 2


_HTML = b"""<!doctype html><html><head><title>Test Site</title></head><body>
<h1>Hello world</h1>
<input id="name" placeholder="Your name" />
<button onclick="window.location='/clicked?name='+document.getElementById('name').value">Submit</button>
</body></html>"""


class _FixtureHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/clicked"):
            body = b"<html><title>Clicked page</title><body><h2>Done</h2></body></html>"
        else:
            body = _HTML
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@pytest.mark.browser
@pytest.mark.asyncio
async def test_real_browser_interaction(tmp_path) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        ts = BrowserToolset()
        ts._shots_dir = tmp_path
        try:
            await ts.setup()
        except BaseException as e:
            pytest.skip(f"Chromium not available, skipped: {e}")

        ctx = FakeRunContext()
        base_url = f"http://127.0.0.1:{server.server_port}/"
        opened = await ts.browser_open(ctx, base_url)
        assert "Test Site" in opened
        assert "Hello world" in await ts.browser_get_text(ctx)

        await ts.browser_fill(ctx, "Your name", "Jarvis")
        await ts.browser_click(ctx, "Submit")
        assert "/clicked" in await ts.browser_status(ctx)

        shot = await ts.browser_screenshot(ctx, "check")
        assert "Saved screenshot" in shot
        assert list(tmp_path.glob("check_*.png"))

        await ts.aclose()
    finally:
        server.shutdown()

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "agent-starter-python"
version = "1.0.0"
description = "Simple voice AI assistant built with LiveKit Agents for Python"
requires-python = ">=3.10, <3.15"

dependencies = [
    "ddgs>=9.16.0",
    "langchain-community>=0.4.2",
    "livekit-agents[google,images,mcp]~=1.5",
    "livekit-plugins-ai-coustics~=0.2",
    "playwright>=1.48",
    "python-dotenv",
    "soundfile>=0.14.0",
    "speechbrain",
    "torch",
    "torchaudio",
]

[dependency-groups]
dev = [
    "pytest",
    "pytest-asyncio",
    "ruff",
]

[tool.uv]
constraint-dependencies = ["yarl<1.24"]

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchaudio = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
markers = [
    "browser: requires installed Chromium (run `uv run playwright install chromium`)",
]

[tool.ruff]
line-length = 88
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "B", "A", "C4", "UP", "SIM", "RUF"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
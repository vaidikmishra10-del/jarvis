# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.14
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim

# Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

# Ensure local models and torch caches are accessible
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# Install system dependencies: build tools, Node.js/npm for Playwright, and libsndfile1 for soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    nodejs \
    npm \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./
RUN mkdir -p src

# Install all Python dependencies using uv sync (manages .venv automatically)
RUN uv sync --locked

# Pre-download any ML models or files the agent needs
RUN uv run --module livekit.agents download-files

# Install Playwright and Chromium browser
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.cache/ms-playwright
RUN uv pip install --system playwright
RUN uv run python -m playwright install --with-deps chromium

# Copy all remaining application files into the container
COPY . .

# Create a non-privileged user and setup specific writable directories securely
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser \
  && mkdir -p /app/browser_shots /app/.cache \
  && chown -R appuser:appuser /app/browser_shots /app/.cache

ENV HOME=/app
USER appuser

# Run the AgentServer using uv run so it uses the virtual environment correctly
CMD ["uv", "run", "src/agent.py", "start"]
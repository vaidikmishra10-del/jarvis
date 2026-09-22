# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    nodejs \
    npm \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency configuration files
COPY pyproject.toml uv.lock ./

# Install dependencies globally using uv pip to skip editable build errors
RUN uv pip install --system --no-cache .

# Install Playwright and Chromium browser
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.cache/ms-playwright
RUN uv run python -m playwright install --with-deps chromium

# Copy remaining source code
COPY . .

# Run the agent server
CMD ["python", "src/server.py"]
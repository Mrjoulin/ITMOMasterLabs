# Single-stage production image
FROM python:3.11-slim

# Keep Python from writing .pyc files and buffer disabled
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system deps and the `uv` runner
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies (production only, no dev deps)
RUN uv sync --frozen --no-dev && \
    groupadd -r app && useradd -r -g app -d /home/app -s /bin/sh app && \
    mkdir -p /app && chown -R app:app /app

# Copy application files (honors .dockerignore to avoid sensitive files)
COPY main.py /app/
COPY utils  /app/utils
COPY listeners /app/listeners

# Ensure files are owned by non-root user
RUN chown -R app:app /app
RUN mkdir -p /home/app/.cache && chown -R app:app /home/app/.cache

# Switch to non-root user for runtime
USER app
WORKDIR /app

# Run database migrations then start the app (runs as non-root)
CMD ["sh", "-c", "uv run main.py polymarket binance"]
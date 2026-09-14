# Single-stage production image
FROM python:3.11-slim

# Keep Python from writing .pyc files and buffer disabled
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system deps and the `uv` runner
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    pip install --no-cache-dir uv && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock .python-version ./

# Install Python dependencies (production only, no dev deps)
RUN uv sync --frozen --no-dev && \
    groupadd -r app && useradd -r -g app -d /home/app -s /bin/sh app && \
    mkdir -p /app && chown -R app:app /app

# Copy application files (honors .dockerignore to avoid sensitive files)
COPY . /app/

# Ensure files are owned by non-root user
RUN chown -R app:app /app

# Switch to non-root user for runtime
USER app

WORKDIR /app

EXPOSE 5555

# Run database migrations then start the app (runs as non-root)
CMD ["sh", "-c", "uv run alembic upgrade head && uv run main.py"]

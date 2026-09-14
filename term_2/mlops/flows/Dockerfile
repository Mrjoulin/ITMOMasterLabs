FROM prefecthq/prefect:3-latest

# Keep Python from writing .pyc files and buffer disabled
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install the `uv` runner
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies (production only, no dev deps)
RUN uv pip install -r pyproject.toml

RUN groupadd -r app && useradd -r -g app -d /home/app -s /bin/sh app && \
    mkdir -p /app && chown -R app:app /app

# Copy all flow files
COPY src /app/src
COPY model /app/model
COPY deployments/*.yaml /app/deployments/
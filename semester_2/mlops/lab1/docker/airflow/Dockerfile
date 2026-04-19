FROM apache/airflow:latest-python3.14

USER root

# Install system dependencies for performance
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    postgresql-client \
    redis-tools \
    curl \
    htop \
    iotop \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

USER airflow

# Install uv for ultra-fast package management
RUN pip install uv

# Copy project files
COPY --chown=airflow:0 pyproject.toml /opt/airflow/pyproject.toml
COPY --chown=airflow:0 uv.lock /opt/airflow/uv.lock
COPY --chown=airflow:0 src /opt/airflow/src
COPY --chown=airflow:0 config /opt/airflow/config
COPY --chown=airflow:0 dags /opt/airflow/dags

# Install Python packages using uv
WORKDIR /opt/airflow
RUN uv pip install .

# Install additional airflow providers
RUN uv pip install \
    apache-airflow-providers-postgres \
    apache-airflow-providers-redis \
    apache-airflow-providers-http

# Create necessary directories
RUN mkdir -p /opt/airflow/{logs,plugins,config} && \
    chown -R airflow:0 /opt/airflow/{logs,plugins,config,src}

# Performance tuning: Reduce Airflow latency
ENV AIRFLOW__CORE__SQL_ALCHEMY_POOL_SIZE=20
ENV AIRFLOW__CORE__SQL_ALCHEMY_MAX_OVERFLOW=30
ENV AIRFLOW__CORE__SQL_ALCHEMY_POOL_RECYCLE=1800
ENV AIRFLOW__CORE__SQL_ALCHEMY_POOL_PRE_PING=true
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=60
ENV AIRFLOW__CORE__DAG_FILE_PROCESSOR_TIMEOUT=120
ENV AIRFLOW__CORE__MIN_SERIALIZED_DAG_UPDATE_INTERVAL=60
ENV AIRFLOW__CORE__STORE_SERIALIZED_DAGS=true

# Scheduler optimization
ENV AIRFLOW__SCHEDULER__MIN_FILE_PROCESS_INTERVAL=5
ENV AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL=60
ENV AIRFLOW__SCHEDULER__MAX_DAGRUNS_PER_LOOP_TO_SCHEDULE=32
ENV AIRFLOW__SCHEDULER__SCHEDULER_IDLE_SLEEP_TIME=0
ENV AIRFLOW__SCHEDULER__PARSING_PROCESSES=4

# Worker optimization
ENV AIRFLOW__CELERY__WORKER_CONCURRENCY=${WORKER_CONCURRENCY:-8}
ENV AIRFLOW__CELERY__WORKER_PREFETCH_MULTIPLIER=${WORKER_PREFETCH_MULTIPLIER:-1}
ENV AIRFLOW__CELERY__WORKER_ENABLE_REMOTE_CONTROL=true
ENV AIRFLOW__CELERY__SYNC_PARALLELISM=8

# Webserver optimization
ENV AIRFLOW__WEBSERVER__WORKERS=${WEBSERVER_WORKERS:-4}
ENV AIRFLOW__WEBSERVER__WORKER_TIMEOUT=120
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=true
ENV AIRFLOW__WEBSERVER__EXPOSE_GC_ENDPOINTS=true

# Reduce logging overhead for speed
ENV AIRFLOW__LOGGING__LOGGING_LEVEL=INFO
ENV AIRFLOW__LOGGING__FAB_LOGGING_LEVEL=WARNING

# Set Python path
ENV PYTHONPATH=/opt/airflow/src

USER airflow

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)"

WORKDIR /opt/airflow
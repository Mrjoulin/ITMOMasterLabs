import os
from typing import Dict, Any
from datetime import timedelta

# Rate limiting configuration
MAX_REQUESTS_PER_SECOND = float(os.getenv("MAX_REQUESTS_PER_SECOND", "1.0"))
MIN_REQUEST_INTERVAL_MS = float(os.getenv("MIN_REQUEST_INTERVAL_MS", "100.0"))
REQUEST_TIMEOUT_MS = int(1000.0 / MAX_REQUESTS_PER_SECOND)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "polymarket"),
    "user": os.getenv("POSTGRES_USER", "polymarket"),
    "password": os.getenv("POSTGRES_PASSWORD", "polymarket_pass"),
    "min_size": 5,
    "max_size": 20,
    "command_timeout": 30,
    "server_settings": {
        "jit": "off",
        "application_name": "polymarket_pipeline",
    }
}

# Redis configuration
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "redis"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "db": 0,
    "decode_responses": True,
    "socket_connect_timeout": 0.1,
    "socket_timeout": 0.1,
    "max_connections": 50,
}

# Polymarket API configuration
POLYMARKET_CONFIG = {
    "base_url": os.getenv("POLYMARKET_BASE_URL", "https://gamma-api.polymarket.com"),
    "api_key": os.getenv("POLYMARKET_API_KEY", ""),
    "timeout": 5.0,
    "retries": 3,
    "backoff_factor": 0.3,
    "max_backoff": 2.0,
}

# DAG configuration
DAG_CONFIG = {
    "schedule_interval": "*/5 * * * *",  # Every 5 minutes
    "max_active_runs": 1,
    "catchup": False,
    "default_args": {
        "owner": "polymarket",
        "depends_on_past": False,
        "email": ["admin@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 3,
        "retry_delay": timedelta(minutes=1),
        "retry_exponential_backoff": True,
        "max_retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(minutes=10),
        "sla": timedelta(minutes=15),
    }
}

# Performance tuning
PERFORMANCE_CONFIG = {
    "batch_size": 1000,
    "insert_workers": 4,
    "query_timeout": 30,
    "connection_pool_size": 20,
    "connection_max_overflow": 30,
}

# Monitoring
MONITORING_CONFIG = {
    "metrics_port": 9100,
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "enable_metrics": True,
    "prometheus_port": 9090,
}

# Cache configuration
CACHE_CONFIG = {
    "ttl": 300,  # 5 minutes
    "max_size": 1000,
    "serializer": "json",
}


def get_connection_string() -> str:
    """Get PostgreSQL connection string for SQLAlchemy."""
    return (
        f"postgresql+asyncpg://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


async def get_asyncpg_pool_config() -> Dict[str, Any]:
    """Get asyncpg pool configuration for ultra-fast connections."""
    return {
        "min_size": DB_CONFIG["min_size"],
        "max_size": DB_CONFIG["max_size"],
        "command_timeout": DB_CONFIG["command_timeout"],
        "server_settings": DB_CONFIG["server_settings"],
    }
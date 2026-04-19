#!/usr/bin/env python3
"""
Setup validation script for Polymarket pipeline.
Checks environment, configuration, and dependencies before deployment.
"""

import sys
import os
from pathlib import Path
import asyncio
from typing import List, Dict, Any
import httpx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import structlog
from dotenv import load_dotenv

logger = structlog.get_logger(__name__)

class SetupValidator:
    """Validates Polymarket pipeline setup."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []

    def log_result(self, name: str, passed: bool, message: str = ""):
        """Log validation result."""
        if passed:
            self.passed.append(name)
            logger.info(f"✅ {name}: {message or 'OK'}")
        else:
            self.errors.append(name)
            logger.error(f"❌ {name}: {message}")

    def log_warning(self, name: str, message: str):
        """Log warning."""
        self.warnings.append(name)
        logger.warning(f"⚠️  {name}: {message}")

    def check_env_file(self):
        """Check if .env file exists and has required variables."""
        env_file = Path(".env")
        if not env_file.exists():
            self.log_result("env_file", False, ".env file not found. Copy from .env.example")
            return

        # Check required variables
        required_vars = [
            ("POLYMARKET_API_KEY", "Polymarket API key"),
            ("POSTGRES_PASSWORD", "PostgreSQL password"),
            ("AIRFLOW_PASSWORD", "Airflow password"),
        ]

        load_dotenv(".env")
        for var, desc in required_vars:
            value = os.getenv(var)
            if not value or value == f"your_{var.lower()}" or value == "your_secure_password":
                self.log_result(f"env_{var}", False, f"{desc} not set or using default")
            else:
                self.log_result(f"env_{var}", True, f"{desc} set")

        self.log_result("env_file", True, f"Found at {env_file.absolute()}")

    def check_docker_compose(self):
        """Check docker-compose.yml structure."""
        compose_file = Path("docker-compose.yml")
        if not compose_file.exists():
            self.log_result("docker_compose", False, "docker-compose.yml not found")
            return

        try:
            import yaml
            with open(compose_file) as f:
                config = yaml.safe_load(f)

            # Check required services
            required_services = ["postgres", "redis", "airflow-webserver", "airflow-scheduler", "airflow-worker"]
            for service in required_services:
                if service not in config.get("services", {}):
                    self.log_result(f"docker_service_{service}", False, f"Service {service} not defined")
                else:
                    self.log_result(f"docker_service_{service}", True, f"Service {service} configured")

            self.log_result("docker_compose", True, f"Valid structure, {len(config.get('services', {}))} services defined")

        except ImportError:
            self.log_warning("yaml_check", "PyYAML not installed, skipping detailed validation")
            self.log_result("docker_compose", True, "File exists")
        except Exception as e:
            self.log_result("docker_compose", False, f"Invalid YAML: {e}")

    def check_src_structure(self):
        """Check source code structure."""
        required_dirs = ["src", "src/api", "src/db", "src/models", "src/processors"]
        required_files = [
            "src/api/polymarket_client.py",
            "src/db/postgresql_client.py",
            "src/models/polymarket_data.py",
            "src/processors/feature_calculator.py",
        ]

        for directory in required_dirs:
            dir_path = Path(directory)
            if not dir_path.is_dir():
                self.log_result(f"dir_{directory}", False, f"Directory {directory}/ not found")
            else:
                self.log_result(f"dir_{directory}", True, f"{directory}/ exists")

        for file in required_files:
            file_path = Path(file)
            if not file_path.exists():
                self.log_result(f"file_{file}", False, f"{file} not found")
            else:
                self.log_result(f"file_{file}", True, f"{file} found ({file_path.stat().st_size} bytes)")

    def check_dag_file(self):
        """Check DAG file and syntax."""
        dag_file = Path("dags/polymarket_btc_pipeline.py")
        if not dag_file.exists():
            self.log_result("dag_file", False, "DAG file not found")
            return

        try:
            # Try Python syntax check
            import py_compile
            py_compile.compile(str(dag_file), doraise=True)
            self.log_result("dag_syntax", True, "Python syntax valid")
        except Exception as e:
            self.log_result("dag_syntax", False, f"Syntax error: {e}")

        # Check for required DAG
        content = dag_file.read_text()
        if "polymarket_btc_5m_pipeline" in content:
            self.log_result("dag_name", True, "DAG ID found in file")
        else:
            self.log_result("dag_name", False, "DAG ID not found")

        self.log_result("dag_file", True, f"DAG file exists ({dag_file.stat().st_size} bytes)")

    async def check_polymarket_api(self):
        """Test Polymarket API connectivity."""
        api_key = os.getenv("POLYMARKET_API_KEY")
        if not api_key or api_key.startswith("your_"):
            self.log_result("api_key", False, "Valid API key required")
            return

        base_url = os.getenv("POLYMARKET_BASE_URL", "https://gamma-api.polymarket.com")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                }
                response = await client.get(f"{base_url}/markets", headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    self.log_result("api_connectivity", True, f"API OK, found {len(data)} markets")
                elif response.status_code == 401:
                    self.log_result("api_auth", False, f"Invalid API key: {response.status_code}")
                else:
                    self.log_result("api_response", False, f"API returned {response.status_code}")

        except Exception as e:
            self.log_result("api_test", False, f"API connection failed: {e}")

    def check_config_values(self):
        """Validate configuration values."""
        # Rate limit check
        max_req = float(os.getenv("MAX_REQUESTS_PER_SECOND", "1.0"))
        min_interval = float(os.getenv("MIN_REQUEST_INTERVAL_MS", "100.0"))

        if max_req > 1.0:
            self.log_warning("rate_limit", f"MAX_REQUESTS_PER_SECOND ({max_req}) exceeds Polymarket limit of 1.0")
        else:
            self.log_result("rate_limit_max", True, f"Max {max_req} req/sec correct")

        if min_interval < 100.0:
            self.log_warning("min_interval", f"MIN_REQUEST_INTERVAL_MS ({min_interval}) below 100ms limit")
        else:
            self.log_result("rate_limit_min", True, f"Min interval {min_interval}ms correct")

        # Database config
        required_db_vars = ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"]
        for var in required_db_vars:
            if os.getenv(var):
                self.log_result(f"db_config_{var}", True, f"{var} set")
            else:
                self.log_result(f"db_config_{var}", False, f"{var} not set")

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("SETUP VALIDATION SUMMARY")
        print("="*60)
        print(f"Passed:   {len(self.passed)} checks")
        print(f"Warnings: {len(self.warnings)} checks")
        print(f"Errors:   {len(self.errors)} checks")

        if self.warnings:
            print("\n⚠️  Warnings (non-critical):")
            for warning in self.warnings:
                print(f"   - {warning}")

        if self.errors:
            print("\n❌ Errors (must fix before deployment):")
            for error in self.errors:
                print(f"   - {error}")
            print("\n🛑 Setup validation FAILED\n")
            return False
        else:
            print("\n✅ Setup validation PASSED\n")
            return True

    def generate_checklist(self):
        """Generate deployment checklist."""
        checklist = """
Deployment Checklist:

[ ] Copy .env.example to .env and configure:
    - POLYMARKET_API_KEY (required)
    - POSTGRES_PASSWORD (change from default)
    - AIRFLOW_PASSWORD (change from default)

[ ] Verify Docker and Docker Compose are installed
    $ docker --version
    $ docker compose version

[ ] Build the Docker images
    $ docker compose build

[ ] Start the stack
    $ docker compose up -d

[ ] Monitor initialization (wait 2-3 minutes)
    $ docker compose logs -f airflow-init

[ ] Verify Airflow is accessible
    http://localhost:8080 (login with credentials from .env)

[ ] Check PostgreSQL hypertable was created
    $ docker compose exec postgres psql -U polymarket -d polymarket -c "\dt polymarket.*"

[ ] Trigger a manual DAG run
    - Open Airflow UI
    - Find "polymarket_btc_5m_pipeline"
    - Click the ▶️ button
    - Monitor logs

[ ] Verify data is flowing to TimescaleDB
    $ docker compose exec postgres psql -U polymarket -d polymarket -c "SELECT COUNT(*) FROM polymarket.features;"

[ ] Set up monitoring (optional)
    - Open Prometheus: http://localhost:9090
    - Check metrics: polymarket_pipeline_*

[ ] Enable scheduling (default: off to prevent accidental runs)
    - Unpause the DAG in Airflow UI

[ ] Verify rate limiting in logs
    - Check airflow-worker logs for timing messages
    - Should see "acquired rate limit" delays of 100-1000ms
"""
        print(checklist)

async def main():
    """Run validation."""
    validator = SetupValidator()

    print("🔍 Validating Polymarket Pipeline Setup\n")
    print("="*60)

    # Run checks
    print("\n📋 Checking configuration...")
    validator.check_env_file()
    validator.check_config_values()

    print("\n🏗️  Checking project structure...")
    validator.check_docker_compose()
    validator.check_src_structure()
    validator.check_dag_file()

    print("\n🌐 Testing external services...")
    await validator.check_polymarket_api()

    # Print summary
    print("\n📝 Generating deployment checklist...")
    validator.generate_checklist()

    success = validator.print_summary()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # Simple async setup
    asyncio.run(main())

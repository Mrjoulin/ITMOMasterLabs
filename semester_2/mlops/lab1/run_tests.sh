#!/bin/bash
# Quick test runner for Polymarket pipeline

set -e

echo "🧪 Running Polymarket Pipeline Tests"
echo "====================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please run:"
    echo "   cp .env.example .env"
    echo "   then edit .env with your values"
    exit 1
fi

# Load .env for testing
export $(grep -v '^#' .env | xargs)

echo ""
echo "1️⃣  Checking imports..."
python tests/test_imports.py
echo "✅ Import tests passed"
echo ""

echo "2️⃣  Running validation..."
python validate_setup.py
echo ""

echo "3️⃣  Checking file structure..."
required_files=(
    "dags/polymarket_btc_pipeline.py"
    "src/api/polymarket_client.py"
    "src/db/postgresql_client.py"
    "src/models/polymarket_data.py"
    "src/processors/feature_calculator.py"
    "config/config.py"
    "docker-compose.yml"
    ".env.example"
)

echo "Checking required files:"
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file MISSING"
        exit 1
    fi
done
echo ""

echo "4️⃣  Checking directories..."
required_dirs=(
    "src"
    "dags"
    "config"
    "monitoring"
    "docker"
    "tests"
)

echo "Checking required directories:"
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/"
    else
        echo "  ❌ $dir/ MISSING"
        exit 1
    fi
done
echo ""

echo "🎉 All tests passed! Pipeline is ready for deployment."
echo ""
echo "Next steps:"
echo "  1. uv sync (install dependencies)"
echo "  2. docker compose up -d (start services)"
echo "  3. Check Airflow UI at http://localhost:8080"
echo "  4. See DEPLOYMENT.md for detailed deployment guide"
echo ""
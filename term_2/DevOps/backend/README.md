# Backend on FastAPI + SQLModel + PostgreSQL

## How to set up:
1. Clone repo:
```bash
git clone https://github.com/Miryz21/DevOps-2026.git
cd DevOps-2026/backend
```
2. Make sure [uv](https://docs.astral.sh/uv/) installed: `uv --version`
3. Set up venv with uv: `uv sync`
4. Make sure PostgreSQL running and set `DATABASE_URL` in `.env` file
5. Run alembic migration:
```bash
uv run alembic revision --autogenerate -m "Init db"
uv run alembic upgrade head
```
6. Generate openssl RSA keys for secure JWT creation:
```bash
openssl genrsa -out keys/private.pem 2048
openssl rsa -in keys/private.pem -pubout > keys/public.pem
```
7. Run server: `uv run main.py`

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes import user_router, area_router, task_router, note_router, search_router

app = FastAPI()

API_PREFIX = "/api/v1"

# Include routers
app.include_router(user_router, prefix=API_PREFIX)
app.include_router(area_router, prefix=API_PREFIX)
app.include_router(task_router, prefix=API_PREFIX)
app.include_router(note_router, prefix=API_PREFIX)
app.include_router(search_router, prefix=API_PREFIX)

# Configure CORS from environment variable `ALLOWED_ORIGINS` (comma-separated).
# Do NOT default to wide-open origins in production.
allowed = os.getenv("ALLOWED_ORIGINS")
if allowed:
    origins = [o.strip() for o in allowed.split(",") if o.strip()]
else:
    origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    # Import uvicorn here so importing this module in tests doesn't require uvicorn
    import uvicorn

    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5555"))
    uvicorn.run(app, host=host, port=port)

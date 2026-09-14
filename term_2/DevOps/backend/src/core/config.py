from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Safe defaults for local development and tests. In production, override via env.
    DATABASE_URL: str = "sqlite:///:memory:"

    # SECRET_KEY / PUBLIC_KEY may contain raw PEM content or a path to a PEM file.
    # Provide these via env vars in production.
    SECRET_KEY: str = ""
    PUBLIC_KEY: str = ""

    class Config:
        env_file = ".env"


settings = Settings()

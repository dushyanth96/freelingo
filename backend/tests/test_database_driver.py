from app.core.config import Settings


def test_database_url_uses_asyncpg_for_postgresql_urls() -> None:
    settings = Settings(
        DATABASE_URL="postgresql://user:pass@host:5432/db",
        SECRET_KEY="test-secret-key-for-pytest-ci-32b",
    )
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@host:5432/db"


def test_database_url_preserves_asyncpg_urls() -> None:
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/db?sslmode=require",
        SECRET_KEY="test-secret-key-for-pytest-ci-32b",
    )
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@host:5432/db?sslmode=require"

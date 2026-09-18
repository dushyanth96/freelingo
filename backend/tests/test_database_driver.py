from app.core.config import Settings
from app.core.database import _database_connect_args


def test_database_url_uses_asyncpg_for_postgresql_urls() -> None:
    settings = Settings(
        DATABASE_URL="postgresql://user:pass@host:5432/db",
        SECRET_KEY="test-secret-key-for-pytest-ci-32b",
    )
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@host:5432/db"


def test_database_url_removes_libpq_ssl_params_for_asyncpg() -> None:
    url = "postgresql://user:pass@host:5432/db?sslmode=require&channel_binding=require"
    settings = Settings(DATABASE_URL=url, SECRET_KEY="test-secret-key-for-pytest-ci-32b")
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@host:5432/db"
    assert _database_connect_args(url) == {"ssl": "require"}


def test_database_url_preserves_asyncpg_urls() -> None:
    url = "postgresql+asyncpg://user:pass@host:5432/db?sslmode=require"
    settings = Settings(DATABASE_URL=url, SECRET_KEY="test-secret-key-for-pytest-ci-32b")
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@host:5432/db"
    assert _database_connect_args(url) == {"ssl": "require"}

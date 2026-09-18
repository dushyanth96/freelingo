from sqlalchemy.engine import make_url
from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg

from app.core.config import Settings


def _asyncpg_connect_kwargs(database_url: str) -> dict[str, object]:
    _, kwargs = PGDialect_asyncpg().create_connect_args(make_url(database_url))
    return kwargs


def test_database_url_uses_asyncpg_for_postgresql_urls() -> None:
    settings = Settings(
        DATABASE_URL="postgresql://user:pass@host:5432/db",
        SECRET_KEY="test-secret-key-for-pytest-ci-32b",
    )
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@host:5432/db"


def test_database_url_removes_libpq_ssl_params_for_asyncpg() -> None:
    url = "postgresql://user:pass@host:5432/db?sslmode=require&channel_binding=require"
    settings = Settings(DATABASE_URL=url, SECRET_KEY="test-secret-key-for-pytest-ci-32b")
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@host:5432/db?ssl=require"
    kwargs = _asyncpg_connect_kwargs(settings.DATABASE_URL)
    assert kwargs["ssl"] == "require"
    assert "sslmode" not in kwargs
    assert "channel_binding" not in kwargs


def test_database_url_preserves_asyncpg_urls() -> None:
    url = "postgresql+asyncpg://user:pass@host:5432/db?sslmode=require"
    settings = Settings(DATABASE_URL=url, SECRET_KEY="test-secret-key-for-pytest-ci-32b")
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@host:5432/db?ssl=require"
    kwargs = _asyncpg_connect_kwargs(settings.DATABASE_URL)
    assert kwargs["ssl"] == "require"
    assert "sslmode" not in kwargs

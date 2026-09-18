from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import _normalize_postgres_url, settings


def _database_connect_args(url: str | None = None) -> dict[str, str]:
    target_url = url or settings.DATABASE_URL
    _, connect_args = _normalize_postgres_url(target_url)
    return connect_args


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args=_database_connect_args(),
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

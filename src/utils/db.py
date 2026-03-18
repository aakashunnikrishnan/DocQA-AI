"""
Async database connection pool for DocQA AI system.
Provides connection pooling, retry logic, health checks, and monitoring for PostgreSQL.
"""

import os
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncpg
from asyncpg import Pool, Connection
from asyncpg.exceptions import PostgresError, InterfaceError, ConnectionDoesNotExistError
import json

from src.utils.logger import get_logger
from src.utils.monitoring import get_performance_monitor, measure

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class PoolConfig:
    """Configuration for database connection pool."""
    min_size: int = 5
    max_size: int = 20
    max_queries: int = 50000
    max_inactive_connection_lifetime: float = 300.0  # 5 minutes
    setup_timeout: float = 60.0
    timeout: float = 30.0
    command_timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 30  # seconds
    pool_recycle: int = 3600  # 1 hour
    connection_lifetime: int = 3600  # 1 hour

    @classmethod
    def from_env(cls) -> 'PoolConfig':
        """Create config from environment variables."""
        return cls(
            min_size=int(os.getenv("DB_POOL_MIN_SIZE", "5")),
            max_size=int(os.getenv("DB_POOL_MAX_SIZE", "20")),
            max_queries=int(os.getenv("DB_MAX_QUERIES", "50000")),
            max_inactive_connection_lifetime=float(os.getenv("DB_MAX_INACTIVE_LIFETIME", "300")),
            setup_timeout=float(os.getenv("DB_SETUP_TIMEOUT", "60")),
            timeout=float(os.getenv("DB_TIMEOUT", "30")),
            command_timeout=float(os.getenv("DB_COMMAND_TIMEOUT", "60")),
            retry_attempts=int(os.getenv("DB_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("DB_RETRY_DELAY", "1")),
            health_check_interval=int(os.getenv("DB_HEALTH_CHECK_INTERVAL", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
            connection_lifetime=int(os.getenv("DB_CONNECTION_LIFETIME", "3600"))
        )


@dataclass
class PoolStats:
    """Statistics for database connection pool."""
    total_connections: int = 0
    available_connections: int = 0
    active_connections: int = 0
    connection_requests: int = 0
    connection_timeouts: int = 0
    connection_errors: int = 0
    query_count: int = 0
    query_errors: int = 0
    avg_query_time_ms: float = 0.0
    pool_created_at: float = field(default_factory=time.time)
    last_health_check: float = 0.0
    is_healthy: bool = False


class DatabasePool:
    """
    Async database connection pool with health checks, retry logic, and monitoring.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        config: Optional[PoolConfig] = None,
        **kwargs
    ):
        """
        Initialize database pool.

        Args:
            dsn: PostgreSQL connection string
            config: Pool configuration
            **kwargs: Additional connection parameters
        """
        self.dsn = dsn or os.getenv("DATABASE_URL", "postgresql://localhost/docqa")
        self.config = config or PoolConfig.from_env()
        self.kwargs = kwargs

        self._pool: Optional[Pool] = None
        self._pool_stats = PoolStats()
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_closing = False
        self._lock = asyncio.Lock()

        # Parse DSN components
        self._parse_dsn()

        # Setup connection parameters
        self._connection_kwargs = {
            "min_size": self.config.min_size,
            "max_size": self.config.max_size,
            "max_queries": self.config.max_queries,
            "max_inactive_connection_lifetime": self.config.max_inactive_connection_lifetime,
            "setup_timeout": self.config.setup_timeout,
            "timeout": self.config.timeout,
            "command_timeout": self.config.command_timeout,
            "server_settings": {
                "application_name": "docqa_ai",
                "statement_timeout": f"{self.config.command_timeout * 1000}ms",
                "idle_in_transaction_session_timeout": "60000"
            },
            **kwargs
        }

        logger.info(f"DatabasePool initialized: dsn={self.dsn[:50]}...")

    def _parse_dsn(self):
        """Parse DSN to extract components."""
        # Simple parsing - asyncpg handles full DSN parsing
        # We just extract some info for logging
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(self.dsn)
            self.db_name = parsed.path.lstrip('/') if parsed.path else 'docqa'
            self.db_host = parsed.hostname or 'localhost'
            self.db_port = parsed.port or 5432
            self.db_user = parsed.username or 'postgres'
        except Exception as e:
            logger.warning(f"Failed to parse DSN: {e}")
            self.db_name = 'docqa'
            self.db_host = 'localhost'
            self.db_port = 5432
            self.db_user = 'postgres'

    async def initialize(self) -> bool:
        """
        Initialize the database connection pool.

        Returns:
            True if successful, False otherwise
        """
        if self._pool is not None:
            return True

        try:
            logger.info("Initializing database connection pool...")

            # Create pool with retry
            for attempt in range(self.config.retry_attempts):
                try:
                    self._pool = await asyncpg.create_pool(
                        self.dsn,
                        **self._connection_kwargs
                    )

                    # Test connection
                    async with self._pool.acquire() as conn:
                        await conn.execute("SELECT 1")
                        await self._initialize_schema(conn)

                    # Update stats
                    self._pool_stats.is_healthy = True
                    self._pool_stats.pool_created_at = time.time()

                    # Start health check task
                    self._start_health_check()

                    logger.info(f"Database pool initialized successfully "
                               f"(min={self.config.min_size}, max={self.config.max_size})")
                    return True

                except Exception as e:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        raise

            return False

        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            self._pool_stats.is_healthy = False
            return False

    async def _initialize_schema(self, conn: Connection):
        """Initialize database schema if needed."""
        # Create extensions
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

        # Create metadata table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS docqa_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # Create version table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS docqa_version (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # Create documents table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT,
                file_type TEXT,
                size_bytes BIGINT,
                status TEXT,
                ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'::jsonb,
                version INTEGER DEFAULT 1
            )
        """)

        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_ingested_at ON documents(ingested_at)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_name ON documents USING gin(name gin_trgm_ops)
        """)

        logger.info("Database schema initialized")

    async def close(self):
        """
        Close the database connection pool.
        """
        if self._is_closing:
            return

        self._is_closing = True
        logger.info("Closing database connection pool...")

        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close pool
        if self._pool:
            await self._pool.close()
            self._pool = None

        self._pool_stats.is_healthy = False
        logger.info("Database connection pool closed")

    def _start_health_check(self):
        """Start the health check background task."""
        if self._health_check_task and not self._health_check_task.done():
            return

        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Database health check started")

    async def _health_check_loop(self):
        """Background health check loop."""
        while not self._is_closing:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Skip if pool is closing or not initialized
                if self._is_closing or not self._pool:
                    continue

                # Check pool health
                try:
                    async with self._pool.acquire() as conn:
                        await conn.execute("SELECT 1")
                    self._pool_stats.is_healthy = True
                    self._pool_stats.last_health_check = time.time()
                except Exception as e:
                    logger.warning(f"Health check failed: {e}")
                    self._pool_stats.is_healthy = False

                    # Attempt to reconnect
                    await self._reconnect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)

    async def _reconnect(self):
        """Attempt to reconnect the pool."""
        if self._is_closing:
            return

        try:
            logger.warning("Attempting to reconnect database pool...")

            # Close existing pool
            if self._pool:
                await self._pool.close()
                self._pool = None

            # Recreate pool
            self._pool = await asyncpg.create_pool(
                self.dsn,
                **self._connection_kwargs
            )

            # Test connection
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")

            self._pool_stats.is_healthy = True
            logger.info("Database pool reconnected successfully")

        except Exception as e:
            logger.error(f"Failed to reconnect database pool: {e}")
            self._pool_stats.is_healthy = False

    @asynccontextmanager
    async def acquire(self, retry: bool = True):
        """
        Acquire a connection from the pool with retry logic.

        Args:
            retry: Whether to retry on failure

        Yields:
            Database connection
        """
        if not self._pool:
            logger.warning("Pool not initialized, attempting to initialize...")
            await self.initialize()

            if not self._pool:
                raise RuntimeError("Database pool is not available")

        attempt = 0
        last_error = None

        while attempt < (self.config.retry_attempts if retry else 1):
            try:
                # Get connection
                conn = await self._pool.acquire()

                # Update stats
                self._pool_stats.connection_requests += 1
                self._pool_stats.total_connections = self._pool.get_size()
                self._pool_stats.available_connections = self._pool.get_available_size()
                self._pool_stats.active_connections = self._pool.get_size() - self._pool.get_available_size()

                try:
                    yield conn
                except Exception as e:
                    # Log error and let it propagate
                    logger.error(f"Database operation failed: {e}")
                    raise
                finally:
                    # Release connection back to pool
                    await self._pool.release(conn)
                    return

            except (ConnectionDoesNotExistError, InterfaceError, asyncpg.exceptions.ConnectionDoesNotExistError) as e:
                last_error = e
                attempt += 1
                self._pool_stats.connection_errors += 1
                self._pool_stats.is_healthy = False

                if attempt < self.config.retry_attempts and retry:
                    logger.warning(f"Connection attempt {attempt} failed: {e}, retrying...")
                    await asyncio.sleep(self.config.retry_delay * attempt)

                    # Attempt to reconnect
                    await self._reconnect()
                else:
                    break

            except Exception as e:
                last_error = e
                self._pool_stats.connection_errors += 1
                self._pool_stats.is_healthy = False
                raise

        if last_error:
            raise RuntimeError(f"Failed to acquire connection after {attempt} attempts: {last_error}")
        else:
            raise RuntimeError("Failed to acquire connection")

    async def execute(self, query: str, *args, retry: bool = True) -> str:
        """
        Execute a query and return the result.

        Args:
            query: SQL query
            *args: Query parameters
            retry: Whether to retry on failure

        Returns:
            Query result
        """
        start_time = time.time()

        async with self.acquire(retry=retry) as conn:
            try:
                result = await conn.execute(query, *args)

                # Update stats
                self._pool_stats.query_count += 1
                query_time = (time.time() - start_time) * 1000
                self._pool_stats.avg_query_time_ms = (
                    (self._pool_stats.avg_query_time_ms * (self._pool_stats.query_count - 1) + query_time) /
                    self._pool_stats.query_count
                )

                return result

            except Exception as e:
                self._pool_stats.query_errors += 1
                logger.error(f"Query execution failed: {e}\nQuery: {query[:200]}")
                raise

    async def fetch(self, query: str, *args, retry: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch multiple rows from the database.

        Args:
            query: SQL query
            *args: Query parameters
            retry: Whether to retry on failure

        Returns:
            List of rows as dictionaries
        """
        start_time = time.time()

        async with self.acquire(retry=retry) as conn:
            try:
                rows = await conn.fetch(query, *args)

                # Update stats
                self._pool_stats.query_count += 1
                query_time = (time.time() - start_time) * 1000
                self._pool_stats.avg_query_time_ms = (
                    (self._pool_stats.avg_query_time_ms * (self._pool_stats.query_count - 1) + query_time) /
                    self._pool_stats.query_count
                )

                return [dict(row) for row in rows]

            except Exception as e:
                self._pool_stats.query_errors += 1
                logger.error(f"Fetch failed: {e}\nQuery: {query[:200]}")
                raise

    async def fetchrow(self, query: str, *args, retry: bool = True) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row from the database.

        Args:
            query: SQL query
            *args: Query parameters
            retry: Whether to retry on failure

        Returns:
            Row as dictionary or None
        """
        start_time = time.time()

        async with self.acquire(retry=retry) as conn:
            try:
                row = await conn.fetchrow(query, *args)

                # Update stats
                self._pool_stats.query_count += 1
                query_time = (time.time() - start_time) * 1000
                self._pool_stats.avg_query_time_ms = (
                    (self._pool_stats.avg_query_time_ms * (self._pool_stats.query_count - 1) + query_time) /
                    self._pool_stats.query_count
                )

                return dict(row) if row else None

            except Exception as e:
                self._pool_stats.query_errors += 1
                logger.error(f"Fetchrow failed: {e}\nQuery: {query[:200]}")
                raise

    async def fetchval(self, query: str, *args, retry: bool = True) -> Any:
        """
        Fetch a single value from the database.

        Args:
            query: SQL query
            *args: Query parameters
            retry: Whether to retry on failure

        Returns:
            Single value or None
        """
        start_time = time.time()

        async with self.acquire(retry=retry) as conn:
            try:
                result = await conn.fetchval(query, *args)

                # Update stats
                self._pool_stats.query_count += 1
                query_time = (time.time() - start_time) * 1000
                self._pool_stats.avg_query_time_ms = (
                    (self._pool_stats.avg_query_time_ms * (self._pool_stats.query_count - 1) + query_time) /
                    self._pool_stats.query_count
                )

                return result

            except Exception as e:
                self._pool_stats.query_errors += 1
                logger.error(f"Fetchval failed: {e}\nQuery: {query[:200]}")
                raise

    async def execute_many(self, query: str, params_list: List[tuple]) -> List[Any]:
        """
        Execute multiple queries in batch.

        Args:
            query: SQL query with placeholders
            params_list: List of parameter tuples

        Returns:
            List of results
        """
        results = []

        async with self.acquire() as conn:
            async with conn.transaction():
                for params in params_list:
                    try:
                        result = await conn.execute(query, *params)
                        results.append(result)
                        self._pool_stats.query_count += 1
                    except Exception as e:
                        self._pool_stats.query_errors += 1
                        logger.error(f"Batch execution failed: {e}")
                        raise

        return results

    async def vacuum(self):
        """Run VACUUM ANALYZE on the database."""
        try:
            async with self.acquire(retry=False) as conn:
                await conn.execute("VACUUM ANALYZE")
            logger.info("Database vacuum completed")
        except Exception as e:
            logger.warning(f"Vacuum failed: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        stats = {
            "is_healthy": self._pool_stats.is_healthy,
            "total_connections": self._pool_stats.total_connections,
            "available_connections": self._pool_stats.available_connections,
            "active_connections": self._pool_stats.active_connections,
            "connection_requests": self._pool_stats.connection_requests,
            "connection_timeouts": self._pool_stats.connection_timeouts,
            "connection_errors": self._pool_stats.connection_errors,
            "query_count": self._pool_stats.query_count,
            "query_errors": self._pool_stats.query_errors,
            "avg_query_time_ms": self._pool_stats.avg_query_time_ms,
            "pool_created_at": datetime.fromtimestamp(self._pool_stats.pool_created_at).isoformat(),
            "last_health_check": datetime.fromtimestamp(self._pool_stats.last_health_check).isoformat() if self._pool_stats.last_health_check else None,
            "config": {
                "min_size": self.config.min_size,
                "max_size": self.config.max_size,
                "max_queries": self.config.max_queries,
                "timeout": self.config.timeout,
                "command_timeout": self.config.command_timeout,
                "pool_recycle": self.config.pool_recycle
            },
            "database": {
                "name": self.db_name,
                "host": self.db_host,
                "port": self.db_port,
                "user": self.db_user
            }
        }

        # Add PostgreSQL stats if available
        try:
            async with self.acquire(retry=False) as conn:
                # Get connection count
                conn_count = await conn.fetchval("SELECT count(*) FROM pg_stat_activity")
                stats["active_queries"] = conn_count - 1

                # Get database size
                db_size = await conn.fetchval("SELECT pg_database_size($1)", self.db_name)
                stats["database_size_bytes"] = db_size
                stats["database_size_mb"] = db_size / (1024 * 1024) if db_size else 0

        except Exception as e:
            logger.debug(f"Failed to get PostgreSQL stats: {e}")

        return stats

    def is_healthy(self) -> bool:
        """Check if the pool is healthy."""
        return self._pool is not None and self._pool_stats.is_healthy

    def get_pool_size(self) -> int:
        """Get the current pool size."""
        return self._pool.get_size() if self._pool else 0


# ============================================================
# Global Database Pool
# ============================================================

_db_pool: Optional[DatabasePool] = None
_pool_lock = asyncio.Lock()


async def get_database_pool(
    dsn: Optional[str] = None,
    config: Optional[PoolConfig] = None,
    **kwargs
) -> DatabasePool:
    """
    Get or create the global database pool.

    Args:
        dsn: PostgreSQL connection string
        config: Pool configuration
        **kwargs: Additional connection parameters

    Returns:
        DatabasePool instance
    """
    global _db_pool

    if _db_pool is None:
        async with _pool_lock:
            if _db_pool is None:
                _db_pool = DatabasePool(dsn, config, **kwargs)
                await _db_pool.initialize()

    return _db_pool


async def close_database_pool():
    """
    Close the global database pool.
    """
    global _db_pool

    if _db_pool:
        await _db_pool.close()
        _db_pool = None


# ============================================================
# Convenience Functions
# ============================================================

async def execute_query(query: str, *args, retry: bool = True) -> str:
    """
    Execute a query using the global pool.

    Args:
        query: SQL query
        *args: Query parameters
        retry: Whether to retry on failure

    Returns:
        Query result
    """
    pool = await get_database_pool()
    return await pool.execute(query, *args, retry=retry)


async def fetch_query(query: str, *args, retry: bool = True) -> List[Dict[str, Any]]:
    """
    Fetch multiple rows using the global pool.

    Args:
        query: SQL query
        *args: Query parameters
        retry: Whether to retry on failure

    Returns:
        List of rows as dictionaries
    """
    pool = await get_database_pool()
    return await pool.fetch(query, *args, retry=retry)


async def fetch_one(query: str, *args, retry: bool = True) -> Optional[Dict[str, Any]]:
    """
    Fetch a single row using the global pool.

    Args:
        query: SQL query
        *args: Query parameters
        retry: Whether to retry on failure

    Returns:
        Row as dictionary or None
    """
    pool = await get_database_pool()
    return await pool.fetchrow(query, *args, retry=retry)


async def fetch_value(query: str, *args, retry: bool = True) -> Any:
    """
    Fetch a single value using the global pool.

    Args:
        query: SQL query
        *args: Query parameters
        retry: Whether to retry on failure

    Returns:
        Single value or None
    """
    pool = await get_database_pool()
    return await pool.fetchval(query, *args, retry=retry)


# ============================================================
# Decorator for Database Operations
# ============================================================

def db_transaction(max_retries: int = 3):
    """
    Decorator for database transaction operations with retry logic.

    Args:
        max_retries: Maximum number of retry attempts
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            pool = await get_database_pool()
            last_error = None

            for attempt in range(max_retries):
                try:
                    async with pool.acquire() as conn:
                        async with conn.transaction():
                            return await func(conn, *args, **kwargs)

                except (InterfaceError, ConnectionDoesNotExistError, PostgresError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Transaction failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        raise

            if last_error:
                raise last_error

        return wrapper
    return decorator


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    import asyncio

    async def test_db():
        """Test database connection pool."""
        logging.basicConfig(level=logging.INFO)

        print("Testing Database Connection Pool...")
        print("=" * 60)

        # Get pool
        pool = await get_database_pool()

        # Test connection
        try:
            result = await pool.execute("SELECT 1")
            print("✅ Connection successful")

            # Test fetch
            rows = await pool.fetch("SELECT version()")
            print(f"✅ PostgreSQL version: {rows[0]['version']}")

            # Test transaction
            @db_transaction()
            async def test_transaction(conn):
                await conn.execute("CREATE TABLE IF NOT EXISTS test (id SERIAL PRIMARY KEY, name TEXT)")
                await conn.execute("INSERT INTO test (name) VALUES ($1)", "test_value")
                result = await conn.fetch("SELECT * FROM test")
                return result

            result = await test_transaction()
            print(f"✅ Transaction successful: {len(result)} rows")

            # Get stats
            stats = await pool.get_stats()
            print(f"\n📊 Pool Stats:")
            for key, value in stats.items():
                if not isinstance(value, dict):
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"❌ Test failed: {e}")
        finally:
            await close_database_pool()
            print("✅ Pool closed")

    asyncio.run(test_db())

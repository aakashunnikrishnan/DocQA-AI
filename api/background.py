"""
Background task management for async operations.
Handles long-running tasks like document ingestion, batch processing, etc.
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from fastapi import UploadFile

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """Background task representation."""
    id: str
    name: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False


class BackgroundTaskManager:
    """
    Manages background tasks for async processing.
    """

    def __init__(self, max_concurrent_tasks: int = 5):
        """
        Initialize task manager.

        Args:
            max_concurrent_tasks: Maximum number of concurrent tasks
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks: Dict[str, BackgroundTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self._lock = asyncio.Lock()

        logger.info(f"BackgroundTaskManager initialized with max_concurrent={max_concurrent_tasks}")

    async def start(self):
        """Start the task manager."""
        if self.is_running:
            return

        self.is_running = True

        # Start worker tasks
        for i in range(self.max_concurrent_tasks):
            worker = asyncio.create_task(self._worker_loop(i))
            self.workers.append(worker)

        logger.info("BackgroundTaskManager started")

    async def shutdown(self):
        """Shutdown the task manager."""
        if not self.is_running:
            return

        self.is_running = False

        # Wait for workers to finish
        for worker in self.workers:
            worker.cancel()

        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        logger.info("BackgroundTaskManager shutdown")

    async def create_task(
        self,
        func: Callable,
        *args,
        name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Create a new background task.

        Args:
            func: Async function to run
            *args: Function arguments
            name: Task name (optional)
            **kwargs: Function keyword arguments

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        task_name = name or func.__name__

        task = BackgroundTask(
            id=task_id,
            name=task_name,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            metadata={"func": func.__name__}
        )

        async with self._lock:
            self.tasks[task_id] = task

        # Queue the task
        await self.task_queue.put({
            "task_id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs
        })

        logger.info(f"Created background task: {task_id} ({task_name})")
        return task_id

    async def _worker_loop(self, worker_id: int):
        """Worker loop for processing tasks."""
        logger.debug(f"Worker {worker_id} started")

        while self.is_running:
            try:
                # Get task from queue with timeout
                try:
                    item = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                task_id = item["task_id"]
                func = item["func"]
                args = item["args"]
                kwargs = item["kwargs"]

                # Update task status
                async with self._lock:
                    if task_id not in self.tasks:
                        continue
                    task = self.tasks[task_id]
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()

                logger.info(f"Worker {worker_id} processing task: {task_id}")

                try:
                    # Check for cancellation
                    if task.cancel_requested:
                        raise asyncio.CancelledError()

                    # Execute task
                    result = await func(*args, **kwargs)

                    # Update task status
                    async with self._lock:
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.now()
                        task.progress = 100.0
                        task.result = result

                    logger.info(f"Worker {worker_id} completed task: {task_id}")

                except asyncio.CancelledError:
                    # Task was cancelled
                    async with self._lock:
                        task.status = TaskStatus.CANCELLED
                        task.completed_at = datetime.now()

                    logger.info(f"Worker {worker_id} cancelled task: {task_id}")

                except Exception as e:
                    # Task failed
                    async with self._lock:
                        task.status = TaskStatus.FAILED
                        task.completed_at = datetime.now()
                        task.error = str(e)

                    logger.error(f"Worker {worker_id} failed task: {task_id} - {e}")

                # Mark task as done
                self.task_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)

        logger.debug(f"Worker {worker_id} stopped")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return {
            "id": task.id,
            "name": task.name,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "progress": task.progress,
            "error": task.error,
            "metadata": task.metadata
        }

    def list_tasks(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List tasks."""
        tasks = list(self.tasks.values())

        if status:
            tasks = [t for t in tasks if t.status.value == status]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return [self.get_task_status(t.id) for t in tasks[:limit]]

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False

        # Request cancellation
        task.cancel_requested = True

        # If task is pending, just mark as cancelled
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get task statistics."""
        total = len(self.tasks)
        pending = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
        running = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        cancelled = sum(1 for t in self.tasks.values() if t.status == TaskStatus.CANCELLED)

        return {
            "total": total,
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
            "cancelled": cancelled,
            "queue_size": self.task_queue.qsize(),
            "workers": len(self.workers)
        }


# Global task manager instance
task_manager = BackgroundTaskManager()


async def process_ingestion_task(
    files: List[UploadFile],
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str
) -> Dict[str, Any]:
    """
    Background task for document ingestion.
    """
    from api.routes import process_ingestion_async

    logger.info(f"Processing ingestion task: {len(files)} files")

    try:
        result = await process_ingestion_async(
            files=files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy
        )

        return result.dict()

    except Exception as e:
        logger.error(f"Ingestion task failed: {e}")
        raise


async def cleanup_expired_tasks():
    """Clean up expired tasks (older than 24 hours)."""
    cutoff = datetime.now() - timedelta(hours=24)

    async with task_manager._lock:
        expired = [
            task_id for task_id, task in task_manager.tasks.items()
            if task.completed_at and task.completed_at < cutoff
        ]

        for task_id in expired:
            del task_manager.tasks[task_id]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired tasks")

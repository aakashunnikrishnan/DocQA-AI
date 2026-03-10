"""
Feedback collection module for DocQA AI system.
Provides endpoints and utilities for collecting user feedback on responses,
tracking satisfaction metrics, and improving system performance.
"""

import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict

from fastapi import APIRouter, HTTPException, Request, Depends, status
from pydantic import BaseModel, Field, validator

from src.utils.logger import get_logger
from src.utils.session import get_session, SessionData
from src.utils.cache import get_cache_manager

logger = get_logger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


# ============================================================
# Enums and Models
# ============================================================

class FeedbackType(str, Enum):
    """Types of feedback."""
    RATING = "rating"
    COMMENT = "comment"
    CORRECTION = "correction"
    THUMBS = "thumbs"
    USEFULNESS = "usefulness"
    ACCURACY = "accuracy"
    HALLUCINATION = "hallucination"
    OTHER = "other"


class FeedbackRating(str, Enum):
    """Feedback rating values."""
    VERY_BAD = "very_bad"
    BAD = "bad"
    NEUTRAL = "neutral"
    GOOD = "good"
    VERY_GOOD = "very_good"


class FeedbackSource(str, Enum):
    """Where feedback originated."""
    WEB_UI = "web_ui"
    API = "api"
    SDK = "sdk"
    CLI = "cli"
    WEBSOCKET = "websocket"
    OTHER = "other"


@dataclass
class Feedback:
    """Feedback data structure."""
    id: str
    session_id: str
    user_id: Optional[str] = None
    query_id: str = ""
    response_id: str = ""
    type: FeedbackType = FeedbackType.RATING
    rating: Optional[FeedbackRating] = None
    score: Optional[float] = None  # 0-10 score
    comment: Optional[str] = None
    correction: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: FeedbackSource = FeedbackSource.API
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "query_id": self.query_id,
            "response_id": self.response_id,
            "type": self.type.value,
            "rating": self.rating.value if self.rating else None,
            "score": self.score,
            "comment": self.comment,
            "correction": self.correction,
            "metadata": self.metadata,
            "source": self.source.value,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feedback':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            query_id=data.get("query_id", ""),
            response_id=data.get("response_id", ""),
            type=FeedbackType(data.get("type", "rating")),
            rating=FeedbackRating(data["rating"]) if data.get("rating") else None,
            score=data.get("score"),
            comment=data.get("comment"),
            correction=data.get("correction"),
            metadata=data.get("metadata", {}),
            source=FeedbackSource(data.get("source", "api")),
            created_at=data.get("created_at", time.time())
        )


@dataclass
class FeedbackStats:
    """Feedback statistics."""
    total_feedback: int = 0
    average_rating: float = 0.0
    average_score: float = 0.0
    rating_distribution: Dict[str, int] = field(default_factory=dict)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    source_distribution: Dict[str, int] = field(default_factory=dict)
    comments: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    time_range: Dict[str, float] = field(default_factory=dict)


# ============================================================
# Pydantic Schemas
# ============================================================

class FeedbackRequest(BaseModel):
    """Feedback request model."""
    query_id: Optional[str] = Field(None, description="Query ID")
    response_id: Optional[str] = Field(None, description="Response ID")
    type: FeedbackType = Field(FeedbackType.RATING, description="Feedback type")
    rating: Optional[FeedbackRating] = Field(None, description="Rating")
    score: Optional[float] = Field(None, description="Score (0-10)", ge=0, le=10)
    comment: Optional[str] = Field(None, description="Comment", max_length=1000)
    correction: Optional[str] = Field(None, description="Correction")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    source: FeedbackSource = Field(FeedbackSource.API, description="Feedback source")

    @validator('score')
    def validate_score(cls, v, values):
        """Validate score based on type."""
        if v is not None:
            if v < 0 or v > 10:
                raise ValueError('Score must be between 0 and 10')
        return v


class FeedbackResponse(BaseModel):
    """Feedback response model."""
    success: bool
    feedback_id: str
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class FeedbackStatsResponse(BaseModel):
    """Feedback statistics response."""
    stats: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class FeedbackListResponse(BaseModel):
    """Feedback list response."""
    feedback: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    total_pages: int


# ============================================================
# Feedback Storage
# ============================================================

class FeedbackStorage:
    """
    Storage for feedback data.
    Supports in-memory, file, and Redis backends.
    """

    def __init__(self, storage_type: str = "memory", config: Optional[Dict[str, Any]] = None):
        """
        Initialize feedback storage.

        Args:
            storage_type: 'memory', 'file', 'redis'
            config: Storage configuration
        """
        self.storage_type = storage_type
        self.config = config or {}

        if storage_type == "memory":
            self._memory_storage: Dict[str, Dict] = {}
        elif storage_type == "file":
            self._file_path = self.config.get("file_path", "./data/feedback.jsonl")
            self._init_file_storage()
        elif storage_type == "redis":
            self._init_redis_storage()
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        logger.info(f"FeedbackStorage initialized: type={storage_type}")

    def _init_file_storage(self):
        """Initialize file storage."""
        from pathlib import Path
        path = Path(self._file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()

    def _init_redis_storage(self):
        """Initialize Redis storage."""
        try:
            import redis
            self._redis_client = redis.from_url(
                self.config.get("redis_url", "redis://localhost:6379/0"),
                decode_responses=True
            )
        except ImportError:
            logger.warning("Redis not available, falling back to memory")
            self._memory_storage: Dict[str, Dict] = {}
            self.storage_type = "memory"

    def store(self, feedback: Feedback) -> bool:
        """
        Store feedback.

        Args:
            feedback: Feedback object

        Returns:
            Success status
        """
        feedback_dict = feedback.to_dict()
        feedback_id = feedback.id

        if self.storage_type == "memory":
            self._memory_storage[feedback_id] = feedback_dict
            return True

        elif self.storage_type == "file":
            try:
                with open(self._file_path, 'a') as f:
                    f.write(json.dumps(feedback_dict) + '\n')
                return True
            except Exception as e:
                logger.error(f"Failed to store feedback: {e}")
                return False

        elif self.storage_type == "redis":
            try:
                key = f"feedback:{feedback_id}"
                self._redis_client.setex(key, 86400 * 30, json.dumps(feedback_dict))
                self._redis_client.sadd("feedback:ids", feedback_id)
                return True
            except Exception as e:
                logger.error(f"Failed to store feedback in Redis: {e}")
                return False

        return False

    def get(self, feedback_id: str) -> Optional[Feedback]:
        """Get feedback by ID."""
        if self.storage_type == "memory":
            data = self._memory_storage.get(feedback_id)
            if data:
                return Feedback.from_dict(data)

        elif self.storage_type == "file":
            try:
                with open(self._file_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("id") == feedback_id:
                            return Feedback.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to get feedback: {e}")

        elif self.storage_type == "redis":
            try:
                key = f"feedback:{feedback_id}"
                data = self._redis_client.get(key)
                if data:
                    return Feedback.from_dict(json.loads(data))
            except Exception as e:
                logger.error(f"Failed to get feedback from Redis: {e}")

        return None

    def list_feedback(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Feedback]:
        """List feedback with pagination and filters."""
        feedbacks = []

        if self.storage_type == "memory":
            items = list(self._memory_storage.values())

        elif self.storage_type == "file":
            items = []
            try:
                with open(self._file_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            items.append(json.loads(line))
                        except Exception:
                            continue
            except Exception as e:
                logger.error(f"Failed to list feedback: {e}")
                return []

        elif self.storage_type == "redis":
            try:
                ids = self._redis_client.smembers("feedback:ids")
                items = []
                for fid in ids:
                    data = self._redis_client.get(f"feedback:{fid}")
                    if data:
                        items.append(json.loads(data))
            except Exception as e:
                logger.error(f"Failed to list feedback from Redis: {e}")
                return []
        else:
            return []

        # Apply filters
        if filters:
            filtered_items = []
            for item in items:
                match = True
                for key, value in filters.items():
                    if key not in item or item[key] != value:
                        match = False
                        break
                if match:
                    filtered_items.append(item)
            items = filtered_items

        # Sort by created_at descending
        items.sort(key=lambda x: x.get("created_at", 0), reverse=True)

        # Apply pagination
        paginated = items[offset:offset + limit]

        for data in paginated:
            try:
                feedbacks.append(Feedback.from_dict(data))
            except Exception as e:
                logger.error(f"Failed to parse feedback: {e}")

        return feedbacks

    def get_stats(
        self,
        time_range: Optional[Tuple[float, float]] = None
    ) -> FeedbackStats:
        """Get feedback statistics."""
        stats = FeedbackStats()

        if self.storage_type == "memory":
            items = list(self._memory_storage.values())

        elif self.storage_type == "file":
            items = []
            try:
                with open(self._file_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            items.append(json.loads(line))
                        except Exception:
                            continue
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return stats

        elif self.storage_type == "redis":
            try:
                ids = self._redis_client.smembers("feedback:ids")
                items = []
                for fid in ids:
                    data = self._redis_client.get(f"feedback:{fid}")
                    if data:
                        items.append(json.loads(data))
            except Exception as e:
                logger.error(f"Failed to get stats from Redis: {e}")
                return stats
        else:
            return stats

        # Filter by time range
        if time_range:
            start, end = time_range
            items = [i for i in items if start <= i.get("created_at", 0) <= end]

        stats.total_feedback = len(items)

        # Calculate ratings and scores
        ratings = [i.get("rating") for i in items if i.get("rating")]
        scores = [i.get("score") for i in items if i.get("score") is not None]

        if ratings:
            stats.average_rating = sum(r.value_to_score() for r in ratings) / len(ratings)

        if scores:
            stats.average_score = sum(scores) / len(scores)

        # Distribution
        for item in items:
            rating = item.get("rating")
            if rating:
                stats.rating_distribution[rating] = stats.rating_distribution.get(rating, 0) + 1

            fb_type = item.get("type")
            if fb_type:
                stats.type_distribution[fb_type] = stats.type_distribution.get(fb_type, 0) + 1

            source = item.get("source")
            if source:
                stats.source_distribution[source] = stats.source_distribution.get(source, 0) + 1

            comment = item.get("comment")
            if comment:
                stats.comments.append(comment[:500])

            correction = item.get("correction")
            if correction:
                stats.corrections.append(correction[:500])

        # Time range
        if items:
            created_at = [i.get("created_at", 0) for i in items]
            stats.time_range = {
                "min": min(created_at),
                "max": max(created_at)
            }

        return stats


# ============================================================
# Feedback Service
# ============================================================

class FeedbackService:
    """
    Service for managing feedback collection and analysis.
    """

    def __init__(self, storage: Optional[FeedbackStorage] = None):
        """
        Initialize feedback service.

        Args:
            storage: Feedback storage instance
        """
        self.storage = storage or FeedbackStorage()

        # Cache for aggregate stats
        self._stats_cache = {
            "data": None,
            "timestamp": 0,
            "ttl": 300  # 5 minutes
        }

        logger.info("FeedbackService initialized")

    def submit_feedback(self, request: FeedbackRequest, session_id: str) -> Feedback:
        """
        Submit feedback.

        Args:
            request: Feedback request
            session_id: Session ID

        Returns:
            Feedback object
        """
        feedback = Feedback(
            id=str(uuid.uuid4()),
            session_id=session_id,
            query_id=request.query_id or "",
            response_id=request.response_id or "",
            type=request.type,
            rating=request.rating,
            score=request.score,
            comment=request.comment,
            correction=request.correction,
            metadata=request.metadata or {},
            source=request.source
        )

        # Store feedback
        success = self.storage.store(feedback)
        if not success:
            raise RuntimeError("Failed to store feedback")

        # Invalidate stats cache
        self._stats_cache["data"] = None
        self._stats_cache["timestamp"] = 0

        logger.info(f"Feedback submitted: {feedback.id}")
        return feedback

    def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """Get feedback by ID."""
        return self.storage.get(feedback_id)

    def list_feedback(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Feedback]:
        """List feedback."""
        return self.storage.list_feedback(limit, offset, filters)

    def get_stats(
        self,
        force_refresh: bool = False,
        time_range: Optional[Tuple[float, float]] = None
    ) -> FeedbackStats:
        """Get feedback statistics."""
        # Use cache if available
        if not force_refresh and self._stats_cache["data"] is not None:
            cache_age = time.time() - self._stats_cache["timestamp"]
            if cache_age < self._stats_cache["ttl"]:
                return self._stats_cache["data"]

        # Get fresh stats
        stats = self.storage.get_stats(time_range)

        # Update cache
        self._stats_cache["data"] = stats
        self._stats_cache["timestamp"] = time.time()

        return stats

    def get_rating_distribution(self) -> Dict[str, int]:
        """Get rating distribution."""
        stats = self.get_stats()
        return stats.rating_distribution

    def get_average_rating(self) -> float:
        """Get average rating."""
        stats = self.get_stats()
        return stats.average_rating

    def get_comments(self, limit: int = 10) -> List[str]:
        """Get recent comments."""
        stats = self.get_stats()
        return stats.comments[:limit]


# ============================================================
# API Endpoints
# ============================================================

# Global feedback service
_feedback_service: Optional[FeedbackService] = None


def get_feedback_service() -> FeedbackService:
    """Get or create feedback service."""
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService()
    return _feedback_service


@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    req: Request,
    service: FeedbackService = Depends(get_feedback_service)
):
    """
    Submit feedback for a response.
    """
    # Get session ID
    session_id = req.headers.get("X-Session-ID", "unknown")

    try:
        feedback = service.submit_feedback(request, session_id)

        return FeedbackResponse(
            success=True,
            feedback_id=feedback.id,
            message="Feedback submitted successfully"
        )

    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@router.get("/{feedback_id}")
async def get_feedback(
    feedback_id: str,
    service: FeedbackService = Depends(get_feedback_service)
):
    """
    Get feedback by ID.
    """
    feedback = service.get_feedback(feedback_id)
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found"
        )

    return feedback.to_dict()


@router.get("/list")
async def list_feedback(
    limit: int = 100,
    page: int = 1,
    type_filter: Optional[str] = None,
    rating_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
    service: FeedbackService = Depends(get_feedback_service)
):
    """
    List feedback with pagination and filters.
    """
    offset = (page - 1) * limit

    filters = {}
    if type_filter:
        filters["type"] = type_filter
    if rating_filter:
        filters["rating"] = rating_filter
    if source_filter:
        filters["source"] = source_filter

    feedbacks = service.list_feedback(limit, offset, filters)
    total = len(feedbacks)

    return FeedbackListResponse(
        feedback=[f.to_dict() for f in feedbacks],
        total=total,
        page=page,
        page_size=limit,
        total_pages=(total + limit - 1) // limit if total > 0 else 1
    )


@router.get("/stats")
async def get_feedback_stats(
    force_refresh: bool = False,
    service: FeedbackService = Depends(get_feedback_service)
):
    """
    Get feedback statistics.
    """
    stats = service.get_stats(force_refresh)

    return FeedbackStatsResponse(
        stats={
            "total_feedback": stats.total_feedback,
            "average_rating": stats.average_rating,
            "average_score": stats.average_score,
            "rating_distribution": stats.rating_distribution,
            "type_distribution": stats.type_distribution,
            "source_distribution": stats.source_distribution,
            "comment_count": len(stats.comments),
            "correction_count": len(stats.corrections),
            "time_range": stats.time_range
        }
    )


@router.post("/rating")
async def submit_rating(
    rating: FeedbackRating,
    query_id: Optional[str] = None,
    response_id: Optional[str] = None,
    req: Request = None,
    service: FeedbackService = Depends(get_feedback_service)
):
    """
    Quick rating submission.
    """
    request = FeedbackRequest(
        query_id=query_id,
        response_id=response_id,
        type=FeedbackType.RATING,
        rating=rating,
        source=FeedbackSource.WEB_UI
    )

    session_id = req.headers.get("X-Session-ID", "unknown")
    feedback = service.submit_feedback(request, session_id)

    return FeedbackResponse(
        success=True,
        feedback_id=feedback.id,
        message="Rating submitted successfully"
    )


@router.post("/thumbs")
async def submit_thumbs(
    thumbs_up: bool,
    query_id: Optional[str] = None,
    response_id: Optional[str] = None,
    req: Request = None,
    service: FeedbackService = Depends(get_feedback_service)
):
    """
    Submit thumbs up/down feedback.
    """
    rating = FeedbackRating.GOOD if thumbs_up else FeedbackRating.BAD

    request = FeedbackRequest(
        query_id=query_id,
        response_id=response_id,
        type=FeedbackType.THUMBS,
        rating=rating,
        source=FeedbackSource.WEB_UI
    )

    session_id = req.headers.get("X-Session-ID", "unknown")
    feedback = service.submit_feedback(request, session_id)

    return FeedbackResponse(
        success=True,
        feedback_id=feedback.id,
        message=f"Thumbs {'up' if thumbs_up else 'down'} submitted"
    )


@router.post("/correction")
async def submit_correction(
    query_id: str,
    response_id: str,
    correction: str,
    req: Request = None,
    service: FeedbackService = Depends(get_feedback_service)
):
    """
    Submit a correction for a response.
    """
    request = FeedbackRequest(
        query_id=query_id,
        response_id=response_id,
        type=FeedbackType.CORRECTION,
        correction=correction,
        source=FeedbackSource.WEB_UI
    )

    session_id = req.headers.get("X-Session-ID", "unknown")
    feedback = service.submit_feedback(request, session_id)

    return FeedbackResponse(
        success=True,
        feedback_id=feedback.id,
        message="Correction submitted successfully"
    )


# ============================================================
# Feedback Collection Utilities
# ============================================================

class FeedbackCollector:
    """
    Utility class for collecting feedback in code.
    """

    def __init__(self, service: Optional[FeedbackService] = None):
        self.service = service or FeedbackService()

    def collect_rating(
        self,
        rating: FeedbackRating,
        session_id: str,
        query_id: Optional[str] = None,
        response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """Collect a rating."""
        request = FeedbackRequest(
            query_id=query_id,
            response_id=response_id,
            type=FeedbackType.RATING,
            rating=rating,
            metadata=metadata or {}
        )
        return self.service.submit_feedback(request, session_id)

    def collect_score(
        self,
        score: float,
        session_id: str,
        query_id: Optional[str] = None,
        response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """Collect a score (0-10)."""
        request = FeedbackRequest(
            query_id=query_id,
            response_id=response_id,
            type=FeedbackType.USEFULNESS,
            score=score,
            metadata=metadata or {}
        )
        return self.service.submit_feedback(request, session_id)

    def collect_comment(
        self,
        comment: str,
        session_id: str,
        query_id: Optional[str] = None,
        response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """Collect a comment."""
        request = FeedbackRequest(
            query_id=query_id,
            response_id=response_id,
            type=FeedbackType.COMMENT,
            comment=comment,
            metadata=metadata or {}
        )
        return self.service.submit_feedback(request, session_id)

    def collect_correction(
        self,
        correction: str,
        session_id: str,
        query_id: Optional[str] = None,
        response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """Collect a correction."""
        request = FeedbackRequest(
            query_id=query_id,
            response_id=response_id,
            type=FeedbackType.CORRECTION,
            correction=correction,
            metadata=metadata or {}
        )
        return self.service.submit_feedback(request, session_id)


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Example usage
    import asyncio
    import random

    async def test_feedback():
        """Test feedback collection."""
        print("Testing Feedback Collection...")

        # Create service
        service = FeedbackService()
        collector = FeedbackCollector(service)

        # Submit some feedback
        session_id = "test_session_123"

        # Ratings
        for _ in range(5):
            rating = random.choice(list(FeedbackRating))
            collector.collect_rating(
                rating,
                session_id,
                query_id=f"query_{uuid.uuid4().hex[:8]}"
            )

        # Scores
        for _ in range(3):
            score = random.uniform(0, 10)
            collector.collect_score(
                score,
                session_id
            )

        # Comments
        comments = [
            "Great response! Very helpful.",
            "This was not accurate. Please improve.",
            "Good answer but needs more detail.",
            "Excellent explanation of the concepts.",
            "The response was confusing."
        ]
        for comment in random.sample(comments, 3):
            collector.collect_comment(
                comment,
                session_id
            )

        # Get stats
        stats = service.get_stats()
        print(f"\nFeedback Statistics:")
        print(f"  Total: {stats.total_feedback}")
        print(f"  Avg Rating: {stats.average_rating:.2f}")
        print(f"  Avg Score: {stats.average_score:.2f}")
        print(f"  Rating Distribution: {stats.rating_distribution}")
        print(f"  Comments: {len(stats.comments)}")

        # List feedback
        feedbacks = service.list_feedback(limit=10)
        print(f"\nRecent Feedback:")
        for f in feedbacks[:5]:
            print(f"  {f.type.value}: {f.rating.value if f.rating else 'N/A'} - {f.comment[:50] if f.comment else ''}")

    asyncio.run(test_feedback())

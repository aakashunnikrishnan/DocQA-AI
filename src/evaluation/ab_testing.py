"""
A/B testing framework for DocQA AI system.
Provides tools for running experiments, collecting results, and analyzing performance differences.
"""

import os
import json
import time
import hashlib
import random
import statistics
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import uuid
import math

from src.utils.logger import get_logger
from src.utils.cache import CacheManager
from src.utils.monitoring import get_performance_monitor

logger = get_logger(__name__)


class ExperimentStatus(Enum):
    """Experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class VariantStatus(Enum):
    """Variant status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"


class ExperimentType(Enum):
    """Experiment types."""
    MODEL_COMPARISON = "model_comparison"
    PROMPT_COMPARISON = "prompt_comparison"
    RETRIEVAL_COMPARISON = "retrieval_comparison"
    CHUNKING_COMPARISON = "chunking_comparison"
    EMBEDDING_COMPARISON = "embedding_comparison"
    RERANKING_COMPARISON = "reranking_comparison"
    TEMPERATURE_COMPARISON = "temperature_comparison"
    TOP_K_COMPARISON = "top_k_comparison"
    HYBRID_SEARCH_COMPARISON = "hybrid_search_comparison"
    CUSTOM = "custom"


@dataclass
class ExperimentVariant:
    """Experiment variant configuration."""
    id: str
    name: str
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    status: VariantStatus = VariantStatus.ACTIVE
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentResult:
    """Experiment result for a single test."""
    variant_id: str
    variant_name: str
    experiment_id: str
    session_id: str
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    score: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    feedback: Optional[Dict[str, Any]] = None


@dataclass
class Experiment:
    """A/B experiment definition."""
    id: str
    name: str
    description: str = ""
    type: ExperimentType = ExperimentType.CUSTOM
    status: ExperimentStatus = ExperimentStatus.DRAFT
    variants: List[ExperimentVariant] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    sample_size: int = 1000
    current_sample: int = 0
    target_metric: str = "score"
    min_detectable_effect: float = 0.05
    confidence_level: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"

    def is_running(self) -> bool:
        """Check if experiment is running."""
        if self.status != ExperimentStatus.RUNNING:
            return False

        if self.end_date and datetime.now() > self.end_date:
            return False

        if self.current_sample >= self.sample_size:
            return False

        return True

    def get_active_variants(self) -> List[ExperimentVariant]:
        """Get active variants."""
        return [v for v in self.variants if v.status == VariantStatus.ACTIVE]


class ABTestingFramework:
    """
    A/B testing framework for DocQA AI system.
    """

    def __init__(
        self,
        storage_dir: str = "./data/experiments",
        use_cache: bool = True,
        default_confidence_level: float = 0.95
    ):
        """
        Initialize A/B testing framework.

        Args:
            storage_dir: Directory to store experiment data
            use_cache: Whether to use caching
            default_confidence_level: Default confidence level for analysis
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.default_confidence_level = default_confidence_level

        # Cache
        self.cache_manager = CacheManager() if use_cache else None

        # Storage
        self._experiments: Dict[str, Experiment] = {}
        self._results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        self._load_data()

        # Statistics
        self.stats = {
            "total_experiments": 0,
            "running_experiments": 0,
            "completed_experiments": 0,
            "total_results": 0
        }

        self._update_stats()

        logger.info(f"A/B Testing Framework initialized: storage_dir={storage_dir}")

    def _load_data(self):
        """Load experiment data from storage."""
        # Load experiments
        experiments_file = self.storage_dir / "experiments.json"
        if experiments_file.exists():
            try:
                with open(experiments_file, 'r') as f:
                    data = json.load(f)
                    for exp_data in data:
                        experiment = self._deserialize_experiment(exp_data)
                        self._experiments[experiment.id] = experiment
                logger.info(f"Loaded {len(self._experiments)} experiments")
            except Exception as e:
                logger.error(f"Failed to load experiments: {e}")

        # Load results
        results_file = self.storage_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    for result_data in data:
                        result = self._deserialize_result(result_data)
                        self._results[result.experiment_id].append(result)
                logger.info(f"Loaded {sum(len(r) for r in self._results.values())} results")
            except Exception as e:
                logger.error(f"Failed to load results: {e}")

    def _save_data(self):
        """Save experiment data to storage."""
        # Save experiments
        experiments_data = [self._serialize_experiment(e) for e in self._experiments.values()]
        experiments_file = self.storage_dir / "experiments.json"
        try:
            with open(experiments_file, 'w') as f:
                json.dump(experiments_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save experiments: {e}")

        # Save results
        results_data = []
        for exp_results in self._results.values():
            for result in exp_results:
                results_data.append(self._serialize_result(result))

        results_file = self.storage_dir / "results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def _serialize_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """Serialize experiment to dict."""
        return {
            "id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "type": experiment.type.value,
            "status": experiment.status.value,
            "variants": [
                {
                    "id": v.id,
                    "name": v.name,
                    "description": v.description,
                    "config": v.config,
                    "status": v.status.value,
                    "weight": v.weight,
                    "metadata": v.metadata,
                    "created_at": v.created_at.isoformat(),
                    "updated_at": v.updated_at.isoformat()
                }
                for v in experiment.variants
            ],
            "start_date": experiment.start_date.isoformat() if experiment.start_date else None,
            "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
            "sample_size": experiment.sample_size,
            "current_sample": experiment.current_sample,
            "target_metric": experiment.target_metric,
            "min_detectable_effect": experiment.min_detectable_effect,
            "confidence_level": experiment.confidence_level,
            "metadata": experiment.metadata,
            "created_at": experiment.created_at.isoformat(),
            "updated_at": experiment.updated_at.isoformat(),
            "created_by": experiment.created_by
        }

    def _deserialize_experiment(self, data: Dict[str, Any]) -> Experiment:
        """Deserialize experiment from dict."""
        return Experiment(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            type=ExperimentType(data.get("type", "custom")),
            status=ExperimentStatus(data.get("status", "draft")),
            variants=[
                ExperimentVariant(
                    id=v["id"],
                    name=v["name"],
                    description=v.get("description", ""),
                    config=v.get("config", {}),
                    status=VariantStatus(v.get("status", "active")),
                    weight=v.get("weight", 1.0),
                    metadata=v.get("metadata", {}),
                    created_at=datetime.fromisoformat(v["created_at"]) if v.get("created_at") else datetime.now(),
                    updated_at=datetime.fromisoformat(v["updated_at"]) if v.get("updated_at") else datetime.now()
                )
                for v in data.get("variants", [])
            ],
            start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            sample_size=data.get("sample_size", 1000),
            current_sample=data.get("current_sample", 0),
            target_metric=data.get("target_metric", "score"),
            min_detectable_effect=data.get("min_detectable_effect", 0.05),
            confidence_level=data.get("confidence_level", 0.95),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
            created_by=data.get("created_by", "system")
        )

    def _serialize_result(self, result: ExperimentResult) -> Dict[str, Any]:
        """Serialize result to dict."""
        return {
            "variant_id": result.variant_id,
            "variant_name": result.variant_name,
            "experiment_id": result.experiment_id,
            "session_id": result.session_id,
            "user_id": result.user_id,
            "timestamp": result.timestamp.isoformat(),
            "success": result.success,
            "score": result.score,
            "latency_ms": result.latency_ms,
            "tokens_used": result.tokens_used,
            "metrics": result.metrics,
            "metadata": result.metadata,
            "feedback": result.feedback
        }

    def _deserialize_result(self, data: Dict[str, Any]) -> ExperimentResult:
        """Deserialize result from dict."""
        return ExperimentResult(
            variant_id=data["variant_id"],
            variant_name=data["variant_name"],
            experiment_id=data["experiment_id"],
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            success=data.get("success", False),
            score=data.get("score", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
            tokens_used=data.get("tokens_used", 0),
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {}),
            feedback=data.get("feedback")
        )

    def _update_stats(self):
        """Update statistics."""
        self.stats["total_experiments"] = len(self._experiments)
        self.stats["running_experiments"] = sum(1 for e in self._experiments.values() if e.is_running())
        self.stats["completed_experiments"] = sum(1 for e in self._experiments.values() if e.status == ExperimentStatus.COMPLETED)
        self.stats["total_results"] = sum(len(r) for r in self._results.values())

    def create_experiment(
        self,
        name: str,
        variants: List[Dict[str, Any]],
        experiment_type: Union[str, ExperimentType] = ExperimentType.CUSTOM,
        description: str = "",
        sample_size: int = 1000,
        target_metric: str = "score",
        min_detectable_effect: float = 0.05,
        confidence_level: float = 0.95,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: str = "system"
    ) -> Experiment:
        """
        Create a new A/B experiment.

        Args:
            name: Experiment name
            variants: List of variant configurations
            experiment_type: Type of experiment
            description: Experiment description
            sample_size: Target sample size
            target_metric: Metric to optimize
            min_detectable_effect: Minimum detectable effect size
            confidence_level: Confidence level for analysis
            metadata: Additional metadata
            created_by: Creator identifier

        Returns:
            Created Experiment object
        """
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"

        # Create variants
        variant_objects = []
        for i, v in enumerate(variants):
            variant = ExperimentVariant(
                id=f"var_{uuid.uuid4().hex[:6]}",
                name=v.get("name", f"Variant {i+1}"),
                description=v.get("description", ""),
                config=v.get("config", {}),
                weight=v.get("weight", 1.0),
                metadata=v.get("metadata", {})
            )
            variant_objects.append(variant)

        # Normalize weights
        total_weight = sum(v.weight for v in variant_objects)
        if total_weight > 0:
            for v in variant_objects:
                v.weight = v.weight / total_weight

        # Create experiment
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            type=ExperimentType(experiment_type) if isinstance(experiment_type, str) else experiment_type,
            variants=variant_objects,
            sample_size=sample_size,
            target_metric=target_metric,
            min_detectable_effect=min_detectable_effect,
            confidence_level=confidence_level,
            metadata=metadata or {},
            created_by=created_by
        )

        # Store experiment
        self._experiments[experiment_id] = experiment
        self._save_data()
        self._update_stats()

        logger.info(f"Created experiment: {experiment_id} ({name}) with {len(variants)} variants")
        return experiment

    def start_experiment(self, experiment_id: str) -> bool:
        """
        Start an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Success status
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            logger.warning(f"Experiment not found: {experiment_id}")
            return False

        if experiment.status != ExperimentStatus.DRAFT:
            logger.warning(f"Experiment {experiment_id} cannot be started (status: {experiment.status.value})")
            return False

        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.now()
        experiment.current_sample = 0
        experiment.updated_at = datetime.now()

        self._save_data()
        self._update_stats()

        logger.info(f"Started experiment: {experiment_id}")
        return True

    def stop_experiment(self, experiment_id: str) -> bool:
        """
        Stop an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Success status
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            logger.warning(f"Experiment not found: {experiment_id}")
            return False

        if experiment.status not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            logger.warning(f"Experiment {experiment_id} cannot be stopped (status: {experiment.status.value})")
            return False

        experiment.status = ExperimentStatus.STOPPED
        experiment.end_date = datetime.now()
        experiment.updated_at = datetime.now()

        self._save_data()
        self._update_stats()

        logger.info(f"Stopped experiment: {experiment_id}")
        return True

    def pause_experiment(self, experiment_id: str) -> bool:
        """
        Pause an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Success status
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            logger.warning(f"Experiment not found: {experiment_id}")
            return False

        if experiment.status != ExperimentStatus.RUNNING:
            logger.warning(f"Experiment {experiment_id} cannot be paused (status: {experiment.status.value})")
            return False

        experiment.status = ExperimentStatus.PAUSED
        experiment.updated_at = datetime.now()

        self._save_data()
        self._update_stats()

        logger.info(f"Paused experiment: {experiment_id}")
        return True

    def resume_experiment(self, experiment_id: str) -> bool:
        """
        Resume a paused experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Success status
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            logger.warning(f"Experiment not found: {experiment_id}")
            return False

        if experiment.status != ExperimentStatus.PAUSED:
            logger.warning(f"Experiment {experiment_id} cannot be resumed (status: {experiment.status.value})")
            return False

        experiment.status = ExperimentStatus.RUNNING
        experiment.updated_at = datetime.now()

        self._save_data()
        self._update_stats()

        logger.info(f"Resumed experiment: {experiment_id}")
        return True

    def get_variant_for_user(
        self,
        experiment_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[ExperimentVariant]:
        """
        Get a variant for a user based on assignment logic.

        Args:
            experiment_id: Experiment ID
            user_id: User ID
            session_id: Session ID

        Returns:
            Assigned variant or None
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment or not experiment.is_running():
            return None

        active_variants = experiment.get_active_variants()
        if not active_variants:
            return None

        # Use consistent assignment for users
        if user_id:
            assignment_key = f"{experiment_id}:{user_id}"
        elif session_id:
            assignment_key = f"{experiment_id}:{session_id}"
        else:
            assignment_key = f"{experiment_id}:{uuid.uuid4().hex}"

        # Hash for deterministic assignment
        hash_val = int(hashlib.md5(assignment_key.encode()).hexdigest(), 16)
        random_val = (hash_val % 10000) / 10000  # 0-1

        # Assign based on weights
        cumulative = 0.0
        for variant in active_variants:
            cumulative += variant.weight
            if random_val <= cumulative:
                return variant

        return active_variants[-1]  # Fallback

    def record_result(
        self,
        experiment_id: str,
        variant_id: str,
        session_id: str,
        success: bool = True,
        score: float = 0.0,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        feedback: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record a result for an experiment.

        Args:
            experiment_id: Experiment ID
            variant_id: Variant ID
            session_id: Session ID
            success: Whether the test was successful
            score: Quality score
            latency_ms: Response latency
            tokens_used: Tokens used
            metrics: Additional metrics
            metadata: Additional metadata
            user_id: User ID
            feedback: User feedback

        Returns:
            Success status
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            logger.warning(f"Experiment not found: {experiment_id}")
            return False

        # Find variant
        variant = None
        for v in experiment.variants:
            if v.id == variant_id:
                variant = v
                break

        if not variant:
            logger.warning(f"Variant not found: {variant_id}")
            return False

        # Create result
        result = ExperimentResult(
            variant_id=variant_id,
            variant_name=variant.name,
            experiment_id=experiment_id,
            session_id=session_id,
            user_id=user_id,
            success=success,
            score=score,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            metrics=metrics or {},
            metadata=metadata or {},
            feedback=feedback
        )

        # Store result
        self._results[experiment_id].append(result)
        experiment.current_sample += 1
        experiment.updated_at = datetime.now()

        # Check if sample size reached
        if experiment.current_sample >= experiment.sample_size:
            experiment.status = ExperimentStatus.COMPLETED
            experiment.end_date = datetime.now()
            logger.info(f"Experiment {experiment_id} completed (sample size reached)")

        self._save_data()
        self._update_stats()

        return True

    def get_experiment_results(
        self,
        experiment_id: str,
        include_details: bool = True
    ) -> Dict[str, Any]:
        """
        Get results for an experiment.

        Args:
            experiment_id: Experiment ID
            include_details: Whether to include detailed results

        Returns:
            Experiment results dictionary
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}

        results = self._results.get(experiment_id, [])

        # Group by variant
        variant_results = defaultdict(list)
        for result in results:
            variant_results[result.variant_id].append(result)

        # Calculate statistics per variant
        variant_stats = {}
        for variant in experiment.variants:
            variant_data = variant_results.get(variant.id, [])

            if not variant_data:
                variant_stats[variant.id] = {
                    "variant_name": variant.name,
                    "sample_size": 0,
                    "success_rate": 0.0,
                    "avg_score": 0.0,
                    "avg_latency_ms": 0.0,
                    "avg_tokens": 0.0,
                    "metrics": {}
                }
                continue

            scores = [r.score for r in variant_data]
            success_rates = [1 if r.success else 0 for r in variant_data]
            latencies = [r.latency_ms for r in variant_data]
            tokens = [r.tokens_used for r in variant_data]

            # Collect all metrics
            all_metrics = defaultdict(list)
            for r in variant_data:
                for key, value in r.metrics.items():
                    all_metrics[key].append(value)

            metric_stats = {}
            for key, values in all_metrics.items():
                metric_stats[key] = {
                    "mean": statistics.mean(values) if values else 0,
                    "median": statistics.median(values) if values else 0,
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0
                }

            variant_stats[variant.id] = {
                "variant_name": variant.name,
                "sample_size": len(variant_data),
                "success_rate": statistics.mean(success_rates),
                "avg_score": statistics.mean(scores),
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                "avg_tokens": statistics.mean(tokens) if tokens else 0,
                "metrics": metric_stats,
                "score_std": statistics.stdev(scores) if len(scores) > 1 else 0
            }

        # Calculate statistical significance
        significance = self._calculate_significance(experiment, variant_stats)

        # Build result
        result = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "status": experiment.status.value,
            "total_samples": len(results),
            "target_sample": experiment.sample_size,
            "variants": variant_stats,
            "significance": significance,
            "timestamp": datetime.now().isoformat()
        }

        if include_details:
            result["detailed_results"] = [
                {
                    "variant_id": r.variant_id,
                    "variant_name": r.variant_name,
                    "score": r.score,
                    "success": r.success,
                    "latency_ms": r.latency_ms,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results[-100:]  # Last 100 results
            ]

        return result

    def _calculate_significance(
        self,
        experiment: Experiment,
        variant_stats: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance between variants.

        Args:
            experiment: Experiment object
            variant_stats: Variant statistics

        Returns:
            Significance results
        """
        if len(variant_stats) < 2:
            return {"message": "Need at least 2 variants for comparison"}

        # Find the best performing variant
        best_variant = max(
            variant_stats.items(),
            key=lambda x: x[1].get(experiment.target_metric, 0)
        )

        # Compare each variant to the best
        comparisons = {}
        for variant_id, stats in variant_stats.items():
            if variant_id == best_variant[0]:
                continue

            # Simple t-test comparison
            diff = best_variant[1].get(experiment.target_metric, 0) - stats.get(experiment.target_metric, 0)

            # Check if difference is significant
            is_significant = diff > experiment.min_detectable_effect

            comparisons[variant_id] = {
                "compared_to": best_variant[0],
                "difference": diff,
                "is_significant": is_significant,
                "confidence_level": experiment.confidence_level
            }

        return {
            "best_variant": best_variant[0],
            "best_variant_name": best_variant[1]["variant_name"],
            "best_score": best_variant[1].get(experiment.target_metric, 0),
            "comparisons": comparisons,
            "target_metric": experiment.target_metric,
            "confidence_level": experiment.confidence_level
        }

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Success status
        """
        if experiment_id not in self._experiments:
            return False

        del self._experiments[experiment_id]
        if experiment_id in self._results:
            del self._results[experiment_id]

        self._save_data()
        self._update_stats()

        logger.info(f"Deleted experiment: {experiment_id}")
        return True

    def list_experiments(
        self,
        status: Optional[Union[str, ExperimentStatus]] = None,
        experiment_type: Optional[Union[str, ExperimentType]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List experiments with filters.

        Args:
            status: Filter by status
            experiment_type: Filter by type
            limit: Maximum number of experiments

        Returns:
            List of experiment summaries
        """
        experiments = list(self._experiments.values())

        if status:
            if isinstance(status, str):
                status = ExperimentStatus(status)
            experiments = [e for e in experiments if e.status == status]

        if experiment_type:
            if isinstance(experiment_type, str):
                experiment_type = ExperimentType(experiment_type)
            experiments = [e for e in experiments if e.type == experiment_type]

        # Sort by created_at descending
        experiments.sort(key=lambda x: x.created_at, reverse=True)

        # Limit
        experiments = experiments[:limit]

        return [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type.value,
                "status": e.status.value,
                "variants_count": len(e.variants),
                "sample_size": e.sample_size,
                "current_sample": e.current_sample,
                "created_at": e.created_at.isoformat(),
                "updated_at": e.updated_at.isoformat(),
                "progress": (e.current_sample / e.sample_size * 100) if e.sample_size > 0 else 0
            }
            for e in experiments
        ]

    def get_experiment_stats(self) -> Dict[str, Any]:
        """
        Get overall experiment statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "experiments_by_status": {
                status.value: sum(1 for e in self._experiments.values() if e.status == status)
                for status in ExperimentStatus
            },
            "experiments_by_type": {
                exp_type.value: sum(1 for e in self._experiments.values() if e.type == exp_type)
                for exp_type in ExperimentType
            },
            "total_results": self.stats["total_results"]
        }


# ============================================================
# Convenience Functions
# ============================================================

_ab_testing_framework: Optional[ABTestingFramework] = None


def get_ab_framework() -> ABTestingFramework:
    """Get global A/B testing framework instance."""
    global _ab_testing_framework
    if _ab_testing_framework is None:
        _ab_testing_framework = ABTestingFramework()
    return _ab_testing_framework


def create_model_comparison_experiment(
    name: str,
    models: List[Dict[str, Any]],
    sample_size: int = 1000,
    **kwargs
) -> Experiment:
    """
    Create a model comparison experiment.

    Args:
        name: Experiment name
        models: List of model configurations
        sample_size: Target sample size
        **kwargs: Additional experiment arguments

    Returns:
        Created Experiment
    """
    variants = []
    for i, model in enumerate(models):
        variants.append({
            "name": model.get("name", f"Model {i+1}"),
            "description": model.get("description", ""),
            "config": model,
            "weight": model.get("weight", 1.0)
        })

    framework = get_ab_framework()
    return framework.create_experiment(
        name=name,
        variants=variants,
        experiment_type=ExperimentType.MODEL_COMPARISON,
        sample_size=sample_size,
        target_metric="score",
        **kwargs
    )


def create_prompt_comparison_experiment(
    name: str,
    prompts: List[Dict[str, Any]],
    sample_size: int = 500,
    **kwargs
) -> Experiment:
    """
    Create a prompt comparison experiment.

    Args:
        name: Experiment name
        prompts: List of prompt configurations
        sample_size: Target sample size
        **kwargs: Additional experiment arguments

    Returns:
        Created Experiment
    """
    variants = []
    for i, prompt in enumerate(prompts):
        variants.append({
            "name": prompt.get("name", f"Prompt {i+1}"),
            "description": prompt.get("description", ""),
            "config": {"prompt": prompt},
            "weight": prompt.get("weight", 1.0)
        })

    framework = get_ab_framework()
    return framework.create_experiment(
        name=name,
        variants=variants,
        experiment_type=ExperimentType.PROMPT_COMPARISON,
        sample_size=sample_size,
        target_metric="score",
        **kwargs
    )


def create_retrieval_comparison_experiment(
    name: str,
    retrieval_configs: List[Dict[str, Any]],
    sample_size: int = 500,
    **kwargs
) -> Experiment:
    """
    Create a retrieval strategy comparison experiment.

    Args:
        name: Experiment name
        retrieval_configs: List of retrieval configurations
        sample_size: Target sample size
        **kwargs: Additional experiment arguments

    Returns:
        Created Experiment
    """
    variants = []
    for i, config in enumerate(retrieval_configs):
        variants.append({
            "name": config.get("name", f"Retrieval {i+1}"),
            "description": config.get("description", ""),
            "config": {"retrieval": config},
            "weight": config.get("weight", 1.0)
        })

    framework = get_ab_framework()
    return framework.create_experiment(
        name=name,
        variants=variants,
        experiment_type=ExperimentType.RETRIEVAL_COMPARISON,
        sample_size=sample_size,
        target_metric="score",
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    import asyncio
    import random

    async def test_ab_testing():
        """Test A/B testing framework."""
        logging.basicConfig(level=logging.INFO)

        print("Testing A/B Testing Framework...")
        print("=" * 60)

        # Create framework
        framework = get_ab_framework()

        # Create experiment
        print("\n📊 Creating experiment...")
        experiment = framework.create_experiment(
            name="GPT Model Comparison",
            description="Compare GPT-4 vs GPT-3.5 performance on QA",
            variants=[
                {
                    "name": "GPT-4",
                    "description": "Using GPT-4 model",
                    "config": {"model": "gpt-4", "temperature": 0.7},
                    "weight": 0.5
                },
                {
                    "name": "GPT-3.5",
                    "description": "Using GPT-3.5 model",
                    "config": {"model": "gpt-3.5-turbo", "temperature": 0.7},
                    "weight": 0.5
                }
            ],
            experiment_type=ExperimentType.MODEL_COMPARISON,
            sample_size=10,
            target_metric="score",
            created_by="test_user"
        )
        print(f"  Created experiment: {experiment.id}")

        # Start experiment
        print("\n▶️  Starting experiment...")
        framework.start_experiment(experiment.id)

        # Simulate test runs
        print("\n🔄 Simulating test runs...")
        for i in range(10):
            # Get variant for user
            variant = framework.get_variant_for_user(
                experiment.id,
                session_id=f"session_{i}"
            )
            if variant:
                # Simulate result
                score = random.uniform(0.5, 1.0)
                if variant.name == "GPT-4":
                    score = min(1.0, score + 0.15)  # GPT-4 performs better

                success = score > 0.7
                latency = random.uniform(100, 500)

                framework.record_result(
                    experiment_id=experiment.id,
                    variant_id=variant.id,
                    session_id=f"session_{i}",
                    success=success,
                    score=score,
                    latency_ms=latency,
                    tokens_used=random.randint(100, 500)
                )
                print(f"  Run {i+1}: {variant.name} -> score={score:.3f}")

        # Get results
        print("\n📈 Getting results...")
        results = framework.get_experiment_results(experiment.id)

        print(f"\n  Experiment: {results['experiment_name']}")
        print(f"  Status: {results['status']}")
        print(f"  Total samples: {results['total_samples']}")

        print("\n  Variant Performance:")
        for variant_id, stats in results['variants'].items():
            print(f"    {stats['variant_name']}:")
            print(f"      Sample size: {stats['sample_size']}")
            print(f"      Avg score: {stats['avg_score']:.3f}")
            print(f"      Success rate: {stats['success_rate']:.2%}")
            print(f"      Avg latency: {stats['avg_latency_ms']:.1f}ms")

        if results.get('significance'):
            sig = results['significance']
            print(f"\n  Best Variant: {sig['best_variant_name']}")
            print(f"  Best Score: {sig['best_score']:.3f}")
            print(f"  Target Metric: {sig['target_metric']}")

        print("\n✅ A/B Testing Framework ready!")

    asyncio.run(test_ab_testing())

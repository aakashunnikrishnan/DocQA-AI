#!/usr/bin/env python3
"""
Benchmark comparison script for DocQA AI system.
Compares different models, configurations, and system versions using standardized metrics.
"""

import os
import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger
from scripts.run_benchmarks import BenchmarkRunner, BenchmarkDataset, load_benchmark_dataset

logger = get_logger(__name__)


@dataclass
class ModelComparisonResult:
    """Result of a model comparison."""
    model_name: str
    model_config: Dict[str, Any]
    dataset_name: str
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    latency_metrics: Dict[str, float]
    faithfulness_metrics: Dict[str, float]
    cost_metrics: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_config": self.model_config,
            "dataset_name": self.dataset_name,
            "retrieval_metrics": self.retrieval_metrics,
            "generation_metrics": self.generation_metrics,
            "latency_metrics": self.latency_metrics,
            "faithfulness_metrics": self.faithfulness_metrics,
            "cost_metrics": self.cost_metrics,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelComparisonResult':
        """Create from dictionary."""
        return cls(
            model_name=data["model_name"],
            model_config=data["model_config"],
            dataset_name=data["dataset_name"],
            retrieval_metrics=data["retrieval_metrics"],
            generation_metrics=data["generation_metrics"],
            latency_metrics=data["latency_metrics"],
            faithfulness_metrics=data.get("faithfulness_metrics", {}),
            cost_metrics=data.get("cost_metrics", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )


class ModelComparator:
    """
    Compare multiple models and configurations.
    """

    def __init__(
        self,
        output_dir: str = "./results/comparisons",
        config_path: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize model comparator.

        Args:
            output_dir: Directory to save results
            config_path: Path to configuration file
            use_cache: Whether to use cached results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = config_path
        self.use_cache = use_cache

        # Results storage
        self.results: List[ModelComparisonResult] = []
        self._cache: Dict[str, ModelComparisonResult] = {}

        # Load existing results
        self._load_cached_results()

        logger.info(f"ModelComparator initialized: output_dir={output_dir}")

    def _load_cached_results(self):
        """Load cached comparison results."""
        cache_file = self.output_dir / "comparison_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    for result_data in data:
                        result = ModelComparisonResult.from_dict(result_data)
                        key = f"{result.model_name}_{result.dataset_name}"
                        self._cache[key] = result
                logger.info(f"Loaded {len(self._cache)} cached results")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        """Save comparison results to cache."""
        cache_file = self.output_dir / "comparison_cache.json"
        try:
            data = [r.to_dict() for r in self.results]
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.results)} results to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def run_comparison(
        self,
        model_configs: List[Dict[str, Any]],
        dataset: BenchmarkDataset,
        output_file: Optional[str] = None,
        force_rerun: bool = False
    ) -> List[ModelComparisonResult]:
        """
        Run comparison across multiple model configurations.

        Args:
            model_configs: List of model configurations
            dataset: Benchmark dataset
            output_file: Output file path
            force_rerun: Whether to force re-running benchmarks

        Returns:
            List of comparison results
        """
        results = []

        for config in model_configs:
            model_name = config.get("name", config.get("model", "unknown"))
            logger.info(f"Running benchmark for model: {model_name}")

            # Check cache
            cache_key = f"{model_name}_{dataset.name}"
            if not force_rerun and cache_key in self._cache:
                logger.info(f"Using cached result for {model_name}")
                results.append(self._cache[cache_key])
                continue

            try:
                # Create runner with config
                runner = BenchmarkRunner(
                    config_path=self.config_path,
                    output_dir=str(self.output_dir / "benchmarks"),
                    use_cache=self.use_cache
                )

                # Override config with model config
                if config.get("llm"):
                    runner.config.llm = type(runner.config.llm)(**config["llm"])
                if config.get("embedding"):
                    runner.config.embedding = type(runner.config.embedding)(**config["embedding"])
                if config.get("retrieval"):
                    runner.config.retrieval = type(runner.config.retrieval)(**config["retrieval"])

                # Run benchmark
                benchmark_result = runner.run_full_benchmark(
                    dataset=dataset,
                    max_samples=config.get("max_samples", 50),
                    top_k_values=config.get("top_k_values", [1, 3, 5, 10])
                )

                # Extract metrics
                result = ModelComparisonResult(
                    model_name=model_name,
                    model_config=config,
                    dataset_name=dataset.name,
                    retrieval_metrics=benchmark_result.retrieval_metrics,
                    generation_metrics=benchmark_result.generation_metrics,
                    latency_metrics=benchmark_result.latency_metrics,
                    faithfulness_metrics={},  # Will be filled later
                    cost_metrics={}  # Will be filled later
                )

                # Add cost metrics
                if "cost" in config:
                    result.cost_metrics = {
                        "cost_per_1k_input": config["cost"].get("input", 0),
                        "cost_per_1k_output": config["cost"].get("output", 0),
                        "estimated_cost": result.generation_metrics.get("total_tokens", 0) * config["cost"].get("output", 0) / 1000
                    }

                results.append(result)
                self.results.append(result)
                self._cache[cache_key] = result

                logger.info(f"Completed benchmark for {model_name}")

            except Exception as e:
                logger.error(f"Failed to run benchmark for {model_name}: {e}")
                continue

        # Save results
        self._save_cache()

        if output_file:
            self.save_results(output_file)

        return results

    def compare_models(
        self,
        models: List[str],
        dataset: BenchmarkDataset,
        **kwargs
    ) -> List[ModelComparisonResult]:
        """
        Compare specific models.

        Args:
            models: List of model names
            dataset: Benchmark dataset
            **kwargs: Additional arguments

        Returns:
            List of comparison results
        """
        model_configs = []
        for model in models:
            config = self._get_model_config(model)
            if config:
                model_configs.append(config)

        return self.run_comparison(model_configs, dataset, **kwargs)

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        # Built-in model configurations
        model_configs = {
            "gpt-4": {
                "name": "gpt-4",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "cost": {"input": 0.03, "output": 0.06},
                "max_samples": 50
            },
            "gpt-4o": {
                "name": "gpt-4o",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "cost": {"input": 0.005, "output": 0.015},
                "max_samples": 50
            },
            "gpt-4o-mini": {
                "name": "gpt-4o-mini",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "cost": {"input": 0.00015, "output": 0.0006},
                "max_samples": 50
            },
            "gpt-3.5-turbo": {
                "name": "gpt-3.5-turbo",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "cost": {"input": 0.0005, "output": 0.0015},
                "max_samples": 50
            },
            "claude-3-opus": {
                "name": "claude-3-opus",
                "llm": {
                    "provider": "anthropic",
                    "model": "claude-3-opus-20240229",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "cost": {"input": 0.015, "output": 0.075},
                "max_samples": 50
            },
            "claude-3-sonnet": {
                "name": "claude-3-sonnet",
                "llm": {
                    "provider": "anthropic",
                    "model": "claude-3-sonnet-20240229",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "cost": {"input": 0.003, "output": 0.015},
                "max_samples": 50
            },
            "claude-3-haiku": {
                "name": "claude-3-haiku",
                "llm": {
                    "provider": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "cost": {"input": 0.00025, "output": 0.00125},
                "max_samples": 50
            },
            "llama-2-7b": {
                "name": "llama-2-7b",
                "llm": {
                    "provider": "local",
                    "model": "meta-llama/Llama-2-7b-chat-hf",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "cost": {"input": 0, "output": 0},
                "max_samples": 30,
                "embedding": {
                    "model": "all-MiniLM-L6-v2",
                    "dimension": 384
                }
            },
            "llama-2-13b": {
                "name": "llama-2-13b",
                "llm": {
                    "provider": "local",
                    "model": "meta-llama/Llama-2-13b-chat-hf",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "cost": {"input": 0, "output": 0},
                "max_samples": 30,
                "embedding": {
                    "model": "all-MiniLM-L6-v2",
                    "dimension": 384
                }
            },
            "mistral-7b": {
                "name": "mistral-7b",
                "llm": {
                    "provider": "local",
                    "model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "cost": {"input": 0, "output": 0},
                "max_samples": 30,
                "embedding": {
                    "model": "all-MiniLM-L6-v2",
                    "dimension": 384
                }
            },
            "gemini-1.5-pro": {
                "name": "gemini-1.5-pro",
                "llm": {
                    "provider": "gemini",
                    "model": "gemini-1.5-pro",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "cost": {"input": 0.0025, "output": 0.0075},
                "max_samples": 50
            },
            "gemini-1.5-flash": {
                "name": "gemini-1.5-flash",
                "llm": {
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "cost": {"input": 0.00035, "output": 0.00105},
                "max_samples": 50
            }
        }

        return model_configs.get(model_name)

    def compare_embeddings(
        self,
        embedding_models: List[str],
        dataset: BenchmarkDataset,
        **kwargs
    ) -> List[ModelComparisonResult]:
        """
        Compare different embedding models.

        Args:
            embedding_models: List of embedding model names
            dataset: Benchmark dataset
            **kwargs: Additional arguments

        Returns:
            List of comparison results
        """
        model_configs = []
        for emb_model in embedding_models:
            config = {
                "name": emb_model,
                "embedding": {"model": emb_model},
                "llm": {"provider": "openai", "model": "gpt-4", "temperature": 0.3},
                "max_samples": 50
            }
            model_configs.append(config)

        return self.run_comparison(model_configs, dataset, **kwargs)

    def compare_retrieval_strategies(
        self,
        strategies: List[str],
        dataset: BenchmarkDataset,
        **kwargs
    ) -> List[ModelComparisonResult]:
        """
        Compare different retrieval strategies.

        Args:
            strategies: List of retrieval strategies
            dataset: Benchmark dataset
            **kwargs: Additional arguments

        Returns:
            List of comparison results
        """
        strategy_configs = {
            "vector": {
                "retrieval": {"enable_hybrid_search": False, "enable_reranking": False},
                "name": "vector_only"
            },
            "hybrid": {
                "retrieval": {"enable_hybrid_search": True, "enable_reranking": False},
                "name": "hybrid_search"
            },
            "reranked": {
                "retrieval": {"enable_hybrid_search": True, "enable_reranking": True},
                "name": "hybrid_reranked"
            }
        }

        model_configs = []
        for strategy in strategies:
            if strategy in strategy_configs:
                config = strategy_configs[strategy].copy()
                config["llm"] = {"provider": "openai", "model": "gpt-4", "temperature": 0.3}
                config["max_samples"] = 50
                model_configs.append(config)

        return self.run_comparison(model_configs, dataset, **kwargs)

    def compare_configurations(
        self,
        configs: List[Dict[str, Any]],
        dataset: BenchmarkDataset,
        **kwargs
    ) -> List[ModelComparisonResult]:
        """
        Compare custom configurations.

        Args:
            configs: List of configuration dictionaries
            dataset: Benchmark dataset
            **kwargs: Additional arguments

        Returns:
            List of comparison results
        """
        return self.run_comparison(configs, dataset, **kwargs)

    def save_results(self, filepath: str):
        """
        Save comparison results to file.

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in self.results]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.results)} results to {filepath}")

    def load_results(self, filepath: str) -> List[ModelComparisonResult]:
        """
        Load comparison results from file.

        Args:
            filepath: Input file path

        Returns:
            List of comparison results
        """
        results = []
        with open(filepath, 'r') as f:
            data = json.load(f)
            for item in data:
                results.append(ModelComparisonResult.from_dict(item))

        self.results.extend(results)
        return results

    def generate_report(
        self,
        output_format: str = "html",
        title: str = "Model Comparison Report"
    ) -> str:
        """
        Generate a comparison report.

        Args:
            output_format: Output format ('html', 'markdown', 'json')
            title: Report title

        Returns:
            Report content
        """
        if output_format == "html":
            return self._generate_html_report(title)
        elif output_format == "markdown":
            return self._generate_markdown_report(title)
        elif output_format == "json":
            return json.dumps([r.to_dict() for r in self.results], indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _generate_html_report(self, title: str) -> str:
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th {{ background: #3498db; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                tr:hover {{ background: #f5f5f5; }}
                .metric-good {{ color: #27ae60; font-weight: bold; }}
                .metric-bad {{ color: #e74c3c; }}
                .summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .chart-container {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total models compared: {len(self.results)}</p>
        """

        # Add summary statistics
        if self.results:
            html += self._generate_summary_html()

        # Add detailed tables
        html += self._generate_tables_html()

        html += """
            </div>
        </body>
        </html>
        """

        return html

    def _generate_summary_html(self) -> str:
        """Generate summary HTML."""
        best_bleu = max((r.generation_metrics.get('bleu', 0) for r in self.results), default=0)
        best_rouge = max((r.generation_metrics.get('rouge1_fmeasure', 0) for r in self.results), default=0)
        best_latency = min((r.latency_metrics.get('avg_generation_latency_ms', float('inf')) for r in self.results), default=0)

        html = """
            <div class="summary">
                <h3>Summary</h3>
                <ul>
        """
        html += f"<li><strong>Best BLEU Score:</strong> {best_bleu:.3f}</li>"
        html += f"<li><strong>Best ROUGE-L Score:</strong> {best_rouge:.3f}</li>"
        if best_latency != float('inf'):
            html += f"<li><strong>Best Latency:</strong> {best_latency:.1f}ms</li>"
        html += "</ul></div>"

        return html

    def _generate_tables_html(self) -> str:
        """Generate detailed tables HTML."""
        html = "<h2>Detailed Results</h2>"

        # Generation metrics
        html += "<h3>Generation Metrics</h3>"
        html += "<table><tr><th>Model</th><th>BLEU</th><th>ROUGE-1</th><th>ROUGE-L</th><th>METEOR</th><th>F1</th></tr>"
        for r in self.results:
            html += f"""
                <tr>
                    <td><strong>{r.model_name}</strong></td>
                    <td>{r.generation_metrics.get('bleu', 0):.3f}</td>
                    <td>{r.generation_metrics.get('rouge1_fmeasure', 0):.3f}</td>
                    <td>{r.generation_metrics.get('rougeL_fmeasure', 0):.3f}</td>
                    <td>{r.generation_metrics.get('meteor', 0):.3f}</td>
                    <td>{r.generation_metrics.get('f1', 0):.3f}</td>
                </tr>
            """
        html += "</table>"

        # Retrieval metrics
        html += "<h3>Retrieval Metrics</h3>"
        html += "<table><tr><th>Model</th><th>MRR</th><th>Recall@5</th><th>Precision@5</th><th>NDCG@5</th></tr>"
        for r in self.results:
            html += f"""
                <tr>
                    <td><strong>{r.model_name}</strong></td>
                    <td>{r.retrieval_metrics.get('mrr', 0):.3f}</td>
                    <td>{r.retrieval_metrics.get('recall@5', 0):.3f}</td>
                    <td>{r.retrieval_metrics.get('precision@5', 0):.3f}</td>
                    <td>{r.retrieval_metrics.get('ndcg@5', 0):.3f}</td>
                </tr>
            """
        html += "</table>"

        # Latency metrics
        html += "<h3>Latency Metrics</h3>"
        html += "<table><tr><th>Model</th><th>Avg (ms)</th><th>P95 (ms)</th><th>P99 (ms)</th></tr>"
        for r in self.results:
            html += f"""
                <tr>
                    <td><strong>{r.model_name}</strong></td>
                    <td>{r.latency_metrics.get('avg_generation_latency_ms', 0):.1f}</td>
                    <td>{r.latency_metrics.get('p95_generation_latency_ms', 0):.1f}</td>
                    <td>{r.latency_metrics.get('p99_generation_latency_ms', 0):.1f}</td>
                </tr>
            """
        html += "</table>"

        return html

    def _generate_markdown_report(self, title: str) -> str:
        """Generate Markdown report."""
        md = f"# {title}\n\n"
        md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md += f"Total models compared: {len(self.results)}\n\n"

        # Generation metrics
        md += "## Generation Metrics\n\n"
        md += "| Model | BLEU | ROUGE-1 | ROUGE-L | METEOR | F1 |\n"
        md += "|-------|------|---------|---------|--------|----|\n"
        for r in self.results:
            md += f"| {r.model_name} | {r.generation_metrics.get('bleu', 0):.3f} | "
            md += f"{r.generation_metrics.get('rouge1_fmeasure', 0):.3f} | "
            md += f"{r.generation_metrics.get('rougeL_fmeasure', 0):.3f} | "
            md += f"{r.generation_metrics.get('meteor', 0):.3f} | "
            md += f"{r.generation_metrics.get('f1', 0):.3f} |\n"

        # Retrieval metrics
        md += "\n## Retrieval Metrics\n\n"
        md += "| Model | MRR | Recall@5 | Precision@5 | NDCG@5 |\n"
        md += "|-------|-----|----------|-------------|--------|\n"
        for r in self.results:
            md += f"| {r.model_name} | {r.retrieval_metrics.get('mrr', 0):.3f} | "
            md += f"{r.retrieval_metrics.get('recall@5', 0):.3f} | "
            md += f"{r.retrieval_metrics.get('precision@5', 0):.3f} | "
            md += f"{r.retrieval_metrics.get('ndcg@5', 0):.3f} |\n"

        return md

    def create_visualizations(self, output_dir: Optional[str] = None):
        """
        Create comparison visualizations.

        Args:
            output_dir: Output directory for visualizations
        """
        if not self.results:
            logger.warning("No results to visualize")
            return

        output_dir = Path(output_dir) if output_dir else self.output_dir / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data
        models = [r.model_name for r in self.results]
        bleu_scores = [r.generation_metrics.get('bleu', 0) for r in self.results]
        rouge_scores = [r.generation_metrics.get('rouge1_fmeasure', 0) for r in self.results]
        mrr_scores = [r.retrieval_metrics.get('mrr', 0) for r in self.results]
        latencies = [r.latency_metrics.get('avg_generation_latency_ms', 0) for r in self.results]

        # Create bar chart for generation metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # BLEU scores
        axes[0, 0].bar(models, bleu_scores, color='steelblue')
        axes[0, 0].set_title('BLEU Scores by Model')
        axes[0, 0].set_ylabel('BLEU Score')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # ROUGE scores
        axes[0, 1].bar(models, rouge_scores, color='coral')
        axes[0, 1].set_title('ROUGE-1 Scores by Model')
        axes[0, 1].set_ylabel('ROUGE-1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # MRR scores
        axes[1, 0].bar(models, mrr_scores, color='seagreen')
        axes[1, 0].set_title('MRR Scores by Model')
        axes[1, 0].set_ylabel('MRR Score')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Latencies
        axes[1, 1].bar(models, latencies, color='crimson')
        axes[1, 1].set_title('Average Latency by Model')
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        fig.savefig(output_dir / 'comparison_charts.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Create radar chart
        self._create_radar_chart(output_dir)

        # Create heatmap
        self._create_heatmap(output_dir)

        logger.info(f"Visualizations saved to {output_dir}")

    def _create_radar_chart(self, output_dir: Path):
        """Create a radar chart for model comparison."""
        if len(self.results) < 2:
            return

        # Normalize metrics for radar chart
        metrics = ['bleu', 'rouge1_fmeasure', 'mrr', 'recall@5']
        normalized_data = {}

        for metric in metrics:
            values = [r.generation_metrics.get(metric, 0) if metric in ['bleu', 'rouge1_fmeasure']
                     else r.retrieval_metrics.get(metric, 0) for r in self.results]
            max_val = max(values) if max(values) > 0 else 1
            normalized_data[metric] = [v / max_val for v in values]

        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.results)))

        for i, result in enumerate(self.results):
            values = [normalized_data[m][i] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=result.model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Model Comparison Radar Chart')

        plt.tight_layout()
        fig.savefig(output_dir / 'radar_chart.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_heatmap(self, output_dir: Path):
        """Create a heatmap of model metrics."""
        if len(self.results) < 2:
            return

        # Prepare data for heatmap
        metrics = ['bleu', 'rouge1_fmeasure', 'rougeL_fmeasure', 'meteor', 'f1', 'mrr', 'recall@5']
        data = []

        for result in self.results:
            row = []
            for metric in metrics:
                if metric in result.generation_metrics:
                    row.append(result.generation_metrics[metric])
                elif metric in result.retrieval_metrics:
                    row.append(result.retrieval_metrics[metric])
                else:
                    row.append(0)
            data.append(row)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(self.results)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticklabels([r.model_name for r in self.results])

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(len(self.results)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i][j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Model Comparison Heatmap')

        plt.tight_layout()
        fig.savefig(output_dir / 'heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main entry point for model comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare models and configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare OpenAI models
  python compare_models.py --models gpt-4,gpt-4o,gpt-4o-mini --dataset squad_sample.json

  # Compare local models
  python compare_models.py --models llama-2-7b,mistral-7b --dataset squad_sample.json --max-samples 30

  # Compare embedding models
  python compare_models.py --embeddings all-MiniLM-L6-v2,all-mpnet-base-v2 --dataset natural_questions_sample.json

  # Compare retrieval strategies
  python compare_models.py --strategies vector,hybrid,reranked --dataset custom_qa.csv

  # Generate report
  python compare_models.py --models gpt-4,gpt-4o,claude-3-sonnet --dataset squad_sample.json --report --format html
        """
    )

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to compare"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        help="Comma-separated list of embedding models to compare"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        help="Comma-separated list of retrieval strategies to compare (vector,hybrid,reranked)"
    )
    parser.add_argument(
        "--configs",
        type=str,
        help="JSON file with custom configurations to compare"
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset file"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="auto",
        choices=["squad", "natural_questions", "csv", "auto"],
        help="Type of dataset (default: auto)"
    )
    parser.add_argument(

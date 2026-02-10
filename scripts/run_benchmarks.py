#!/usr/bin/env python3
"""
Benchmark and Evaluation Pipeline for DocQA AI System.
Runs comprehensive evaluations on retrieval and generation quality.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
import csv
import pickle

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_config
from src.retrieval.retriever import VectorRetriever, create_retriever
from src.retrieval.vector_store import FAISSVectorStore
from src.generation.llm_interface import LLMInterface
from src.generation.prompt_templates import get_rag_prompt
from src.generation.response_postprocess import postprocess_response
from src.evaluation.metrics import (
    EvaluationPipeline, calculate_metrics, calculate_retrieval_metrics,
    TextPreprocessor, MetricsResult
)
from src.ingestion.embedding_generator import BatchEmbeddingGenerator
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import ChunkingPipeline

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_name: str = ""
    model: str = ""
    embedding_model: str = ""
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    generation_metrics: Dict[str, float] = field(default_factory=dict)
    latency_metrics: Dict[str, float] = field(default_factory=dict)
    hallucination_metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    samples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, filepath: str):
        """Save results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'BenchmarkResult':
        """Load results from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class BenchmarkDataset:
    """Dataset for benchmarking."""

    def __init__(self, name: str, queries: List[str], references: List[str],
                 contexts: Optional[List[str]] = None,
                 relevant_indices: Optional[List[List[int]]] = None):
        """
        Initialize benchmark dataset.

        Args:
            name: Dataset name
            queries: List of query strings
            references: List of reference answers
            contexts: Optional list of context documents
            relevant_indices: Optional list of relevant document indices
        """
        self.name = name
        self.queries = queries
        self.references = references
        self.contexts = contexts or []
        self.relevant_indices = relevant_indices or []

        self._validate()

    def _validate(self):
        """Validate dataset."""
        assert len(self.queries) == len(self.references), "Queries and references must have same length"
        if self.contexts:
            assert len(self.queries) == len(self.contexts), "Queries and contexts must have same length"
        if self.relevant_indices:
            assert len(self.queries) == len(self.relevant_indices), "Queries and relevant indices must have same length"

    def size(self) -> int:
        """Get dataset size."""
        return len(self.queries)

    def get_item(self, idx: int) -> Dict[str, Any]:
        """Get a single dataset item."""
        item = {
            "query": self.queries[idx],
            "reference": self.references[idx],
            "idx": idx
        }
        if self.contexts:
            item["context"] = self.contexts[idx]
        if self.relevant_indices:
            item["relevant_indices"] = self.relevant_indices[idx]
        return item

    def get_items(self) -> List[Dict[str, Any]]:
        """Get all dataset items."""
        return [self.get_item(i) for i in range(self.size())]

    @classmethod
    def from_squad(cls, filepath: str, max_samples: Optional[int] = None) -> 'BenchmarkDataset':
        """Load dataset from SQuAD format."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        queries = []
        references = []
        contexts = []

        if 'data' in data:
            for article in data['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        if qa.get('answers'):
                            queries.append(qa['question'])
                            references.append(qa['answers'][0]['text'])
                            contexts.append(context)
        else:
            # Try flat format
            for item in data:
                queries.append(item.get('question', item.get('query', '')))
                references.append(item.get('answer', item.get('reference', '')))
                contexts.append(item.get('context', ''))

        if max_samples:
            queries = queries[:max_samples]
            references = references[:max_samples]
            contexts = contexts[:max_samples]

        return cls(
            name="squad",
            queries=queries,
            references=references,
            contexts=contexts
        )

    @classmethod
    def from_natural_questions(cls, filepath: str, max_samples: Optional[int] = None) -> 'BenchmarkDataset':
        """Load dataset from Natural Questions format."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        queries = []
        references = []
        contexts = []

        for item in data:
            if 'question' in item and 'answer' in item:
                queries.append(item['question'])
                references.append(item['answer'])
                contexts.append(item.get('context', ''))

        if max_samples:
            queries = queries[:max_samples]
            references = references[:max_samples]
            contexts = contexts[:max_samples]

        return cls(
            name="natural_questions",
            queries=queries,
            references=references,
            contexts=contexts
        )

    @classmethod
    def from_csv(cls, filepath: str, query_col: str = 'query',
                 reference_col: str = 'answer',
                 context_col: Optional[str] = None,
                 max_samples: Optional[int] = None) -> 'BenchmarkDataset':
        """Load dataset from CSV."""
        import pandas as pd
        df = pd.read_csv(filepath)

        queries = df[query_col].tolist()
        references = df[reference_col].tolist()
        contexts = df[context_col].tolist() if context_col and context_col in df.columns else []

        if max_samples:
            queries = queries[:max_samples]
            references = references[:max_samples]
            contexts = contexts[:max_samples] if contexts else []

        return cls(
            name=Path(filepath).stem,
            queries=queries,
            references=references,
            contexts=contexts
        )

    @classmethod
    def create_sample_dataset(cls, size: int = 10) -> 'BenchmarkDataset':
        """Create a sample dataset for testing."""
        queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "What are neural networks?",
            "Explain natural language processing.",
            "What is computer vision?",
            "How does reinforcement learning work?",
            "What are transformers in AI?",
            "Explain the concept of embeddings.",
            "What is transfer learning?",
            "How does attention mechanism work?"
        ][:size]

        references = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
            "Neural networks are computational systems inspired by biological neural networks.",
            "Natural language processing is a field of AI focused on understanding and generating human language.",
            "Computer vision enables machines to understand and interpret visual information.",
            "Reinforcement learning is a type of ML where agents learn by interacting with an environment.",
            "Transformers are neural network architectures that use attention mechanisms for sequence processing.",
            "Embeddings are dense vector representations of discrete objects like words or entities.",
            "Transfer learning allows models to apply knowledge from one task to another.",
            "Attention mechanisms allow models to focus on relevant parts of the input."
        ][:size]

        return cls(
            name="sample",
            queries=queries,
            references=references,
            contexts=[]  # Contexts will be retrieved from vector store
        )


class BenchmarkRunner:
    """Main benchmark runner."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "./benchmarks",
        use_cache: bool = True,
        verbose: bool = False
    ):
        """
        Initialize benchmark runner.

        Args:
            config_path: Path to configuration file
            output_dir: Directory to save results
            use_cache: Use cached results
            verbose: Verbose output
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.verbose = verbose

        # Load configuration
        self.config = get_config(config_path)

        # Initialize components
        self._init_components()

        # Initialize evaluation pipeline
        self.evaluator = EvaluationPipeline(
            metrics=['bleu', 'rouge', 'meteor', 'f1']
        )

        # Cache for results
        self.cache = {}

        logger.info(f"BenchmarkRunner initialized, output dir: {output_dir}")

    def _init_components(self):
        """Initialize system components for benchmarking."""
        try:
            # Initialize embedding generator
            self.embedding_generator = BatchEmbeddingGenerator(
                model=self.config.embedding.model,
                batch_size=self.config.embedding.batch_size,
                use_cache=self.config.embedding.cache_enabled,
                rate_limit_requests=50
            )

            # Initialize vector store
            vector_store_path = self.config.vector_store.index_path
            if vector_store_path and Path(vector_store_path).exists():
                self.vector_store = FAISSVectorStore(
                    dimension=self.config.vector_store.dimension,
                    index_type=self.config.vector_store.index_type,
                    index_path=vector_store_path
                )
            else:
                self.vector_store = FAISSVectorStore(
                    dimension=self.config.vector_store.dimension,
                    index_type=self.config.vector_store.index_type
                )

            # Initialize retriever
            self.retriever = create_retriever(
                retriever_type="vector",
                vector_store=self.vector_store,
                embedding_generator=self.embedding_generator,
                top_k=self.config.retrieval.top_k
            )

            # Initialize LLM
            self.llm = LLMInterface(
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens
            )

            logger.info("Components initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize components: {e}")
            self.vector_store = None
            self.retriever = None
            self.llm = None

    def prepare_dataset(self, dataset: BenchmarkDataset) -> BenchmarkDataset:
        """
        Prepare dataset by indexing documents if needed.

        Args:
            dataset: Benchmark dataset

        Returns:
            Prepared dataset with relevant indices
        """
        if dataset.contexts and self.vector_store and self.vector_store.get_size() == 0:
            logger.info(f"Indexing {len(dataset.contexts)} contexts from dataset...")

            # Chunk and embed contexts
            chunker = ChunkingPipeline(
                strategy="adaptive",
                chunk_size=800,
                chunk_overlap=150
            )

            all_chunks = []
            all_metadata = []

            for i, context in enumerate(tqdm(dataset.contexts, desc="Processing contexts")):
                chunks = chunker.chunk_document(context, {"doc_idx": i})
                for chunk in chunks:
                    all_chunks.append(chunk.text)
                    all_metadata.append(chunk.metadata)

            # Generate embeddings
            chunk_data = [
                {"text": text, "metadata": meta}
                for text, meta in zip(all_chunks, all_metadata)
            ]
            embeddings = self.embedding_generator.generate_embeddings(chunk_data)

            # Add to vector store
            self.vector_store.add_embeddings(
                embeddings=[e.embedding for e in embeddings],
                texts=all_chunks,
                metadata=all_metadata
            )

            logger.info(f"Indexed {len(all_chunks)} chunks from {len(dataset.contexts)} contexts")

        return dataset

    def run_retrieval_benchmark(
        self,
        dataset: BenchmarkDataset,
        top_k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Run retrieval benchmark.

        Args:
            dataset: Benchmark dataset
            top_k_values: List of K values to evaluate

        Returns:
            Dictionary of retrieval metrics
        """
        if not self.retriever:
            logger.error("Retriever not initialized")
            return {}

        if self.vector_store.get_size() == 0:
            logger.warning("Vector store is empty, skipping retrieval benchmark")
            return {}

        logger.info(f"Running retrieval benchmark on {dataset.size()} queries...")

        all_results = []
        all_retrieved_indices = []

        for item in tqdm(dataset.get_items(), desc="Running retrieval"):
            query = item["query"]

            # Retrieve documents
            start_time = time.time()
            results = self.retriever.retrieve(query, top_k=max(top_k_values))
            latency = time.time() - start_time

            # Store results
            retrieved_indices = []
            for result in results:
                # Try to find document index from metadata
                doc_idx = result.metadata.get("doc_idx", -1)
                if doc_idx == -1:
                    # Try to find by text match (fallback)
                    for i, context in enumerate(dataset.contexts):
                        if result.text in context:
                            doc_idx = i
                            break
                retrieved_indices.append(doc_idx)

            all_retrieved_indices.append(retrieved_indices)
            all_results.append({
                "query": query,
                "retrieved_count": len(results),
                "latency": latency,
                "top_scores": [r.score for r in results[:10]]
            })

        # Calculate metrics
        metrics = {}

        if dataset.relevant_indices:
            # Calculate retrieval metrics
            retrieval_metrics = calculate_retrieval_metrics(
                dataset.relevant_indices,
                all_retrieved_indices,
                [r["top_scores"] for r in all_results]
            )
            metrics.update(retrieval_metrics)

        # Calculate latency metrics
        latencies = [r["latency"] for r in all_results]
        metrics.update({
            "avg_retrieval_latency_ms": np.mean(latencies) * 1000,
            "p95_retrieval_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_retrieval_latency_ms": np.percentile(latencies, 99) * 1000,
            "max_retrieval_latency_ms": np.max(latencies) * 1000
        })

        # Store detailed results
        self._retrieval_results = all_results

        logger.info(f"Retrieval benchmark complete: {len(metrics)} metrics")
        return metrics

    def run_generation_benchmark(
        self,
        dataset: BenchmarkDataset,
        top_k: int = 5,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run generation benchmark.

        Args:
            dataset: Benchmark dataset
            top_k: Number of documents to retrieve
            max_samples: Maximum number of samples to process

        Returns:
            Dictionary of generation metrics
        """
        if not self.llm:
            logger.error("LLM not initialized")
            return {}

        if self.vector_store.get_size() == 0:
            logger.warning("Vector store is empty, skipping generation benchmark")
            return {}

        samples = dataset.get_items()
        if max_samples:
            samples = samples[:max_samples]

        logger.info(f"Running generation benchmark on {len(samples)} queries...")

        generated_responses = []
        generation_latencies = []
        all_retrieved = []

        for item in tqdm(samples, desc="Generating responses"):
            query = item["query"]

            # Retrieve documents
            start_time = time.time()
            retrieved = self.retriever.retrieve(query, top_k=top_k)
            retrieval_time = time.time() - start_time

            # Prepare context
            context_chunks = [
                {"text": r.text, "source": r.metadata.get("file_path", "Unknown")}
                for r in retrieved[:3]
            ]

            # Generate prompt
            prompt = get_rag_prompt(
                question=query,
                chunks=context_chunks
            )

            # Generate response
            start_time = time.time()
            response = self.llm.generate_simple(prompt)
            generation_time = time.time() - start_time

            # Post-process
            processed = postprocess_response(response, aggressive_cleaning=True)

            generated_responses.append(processed.cleaned_text)
            generation_latencies.append(generation_time)
            all_retrieved.append(retrieved)

        # Calculate metrics
        metrics = {}

        # Generation quality metrics
        if dataset.references:
            gen_metrics = calculate_metrics(
                generated_responses,
                dataset.references[:len(generated_responses)],
                metrics=['bleu', 'rouge', 'meteor', 'f1']
            )
            metrics.update(gen_metrics)

        # Latency metrics
        total_latency = np.mean(generation_latencies) * 1000
        metrics.update({
            "avg_generation_latency_ms": total_latency,
            "p95_generation_latency_ms": np.percentile(generation_latencies, 95) * 1000,
            "p99_generation_latency_ms": np.percentile(generation_latencies, 99) * 1000,
            "max_generation_latency_ms": np.max(generation_latencies) * 1000,
            "tokens_per_second": np.mean([len(r.split()) / l for r, l in zip(generated_responses, generation_latencies)])
        })

        # Store detailed results
        self._generation_results = {
            "responses": generated_responses,
            "latencies": generation_latencies,
            "retrieved": all_retrieved
        }

        logger.info(f"Generation benchmark complete: {len(metrics)} metrics")
        return metrics

    def run_full_benchmark(
        self,
        dataset: BenchmarkDataset,
        max_samples: Optional[int] = None,
        top_k_values: List[int] = [1, 3, 5, 10]
    ) -> BenchmarkResult:
        """
        Run full benchmark including retrieval and generation.

        Args:
            dataset: Benchmark dataset
            max_samples: Maximum number of samples to process
            top_k_values: List of K values for retrieval

        Returns:
            BenchmarkResult object
        """
        logger.info(f"Starting full benchmark on dataset: {dataset.name}")
        logger.info(f"Dataset size: {dataset.size()} queries")
        logger.info(f"Max samples: {max_samples or 'all'}")

        # Prepare dataset
        dataset = self.prepare_dataset(dataset)

        # Limit samples
        if max_samples and dataset.size() > max_samples:
            dataset = BenchmarkDataset(
                name=dataset.name,
                queries=dataset.queries[:max_samples],
                references=dataset.references[:max_samples],
                contexts=dataset.contexts[:max_samples] if dataset.contexts else [],
                relevant_indices=dataset.relevant_indices[:max_samples] if dataset.relevant_indices else []
            )

        # Run retrieval benchmark
        retrieval_metrics = self.run_retrieval_benchmark(dataset, top_k_values)

        # Run generation benchmark
        generation_metrics = self.run_generation_benchmark(
            dataset,
            top_k=max(top_k_values),
            max_samples=max_samples
        )

        # Combine results
        result = BenchmarkResult(
            dataset_name=dataset.name,
            model=self.config.llm.model,
            embedding_model=self.config.embedding.model,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            latency_metrics={
                **{k: v for k, v in retrieval_metrics.items() if 'latency' in k},
                **{k: v for k, v in generation_metrics.items() if 'latency' in k}
            },
            config={
                "chunk_size": self.config.processing.chunk_size,
                "chunk_overlap": self.config.processing.chunk_overlap,
                "top_k": self.config.retrieval.top_k,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens
            },
            samples=dataset.get_items()[:10]  # Store first 10 samples
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset.name}_{self.config.llm.model}_{timestamp}.json"
        filepath = self.output_dir / filename
        result.save(str(filepath))

        logger.info(f"Full benchmark complete. Results saved to {filepath}")

        return result

    def run_custom_benchmark(
        self,
        queries: List[str],
        references: List[str],
        contexts: Optional[List[str]] = None,
        dataset_name: str = "custom",
        max_samples: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Run benchmark on custom data.

        Args:
            queries: List of queries
            references: List of reference answers
            contexts: Optional list of contexts
            dataset_name: Name of the dataset
            max_samples: Maximum samples to process

        Returns:
            BenchmarkResult object
        """
        dataset = BenchmarkDataset(
            name=dataset_name,
            queries=queries,
            references=references,
            contexts=contexts or []
        )
        return self.run_full_benchmark(dataset, max_samples)


def load_benchmark_dataset(
    dataset_path: str,
    dataset_type: str = "auto",
    max_samples: Optional[int] = None
) -> BenchmarkDataset:
    """
    Load a benchmark dataset from file.

    Args:
        dataset_path: Path to dataset file
        dataset_type: Type of dataset ('squad', 'natural_questions', 'csv', 'auto')
        max_samples: Maximum samples to load

    Returns:
        BenchmarkDataset object
    """
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if dataset_type == "auto":
        # Auto-detect based on extension
        if path.suffix == ".json":
            # Try to detect format
            with open(path, 'r') as f:
                data = json.load(f)
            if 'data' in data and 'paragraphs' in data['data'][0]:
                dataset_type = "squad"
            else:
                dataset_type = "natural_questions"
        elif path.suffix == ".csv":
            dataset_type = "csv"
        else:
            raise ValueError(f"Could not auto-detect dataset type for {path.suffix}")

    if dataset_type == "squad":
        return BenchmarkDataset.from_squad(str(path), max_samples)
    elif dataset_type == "natural_questions":
        return BenchmarkDataset.from_natural_questions(str(path), max_samples)
    elif dataset_type == "csv":
        return BenchmarkDataset.from_csv(str(path), max_samples=max_samples)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks for DocQA AI system"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset file (JSON or CSV)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="auto",
        choices=["squad", "natural_questions", "csv", "auto"],
        help="Type of dataset"
    )
    parser.add_argument(
        "--query-col",
        type=str,
        default="query",
        help="Column name for queries in CSV"
    )
    parser.add_argument(
        "--reference-col",
        type=str,
        default="answer",
        help="Column name for references in CSV"
    )
    parser.add_argument(
        "--context-col",
        type=str,
        help="Column name for contexts in CSV"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )

    # Benchmark arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmarks",
        help="Directory to save results"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K for retrieval"
    )
    parser.add_argument(
        "--top-k-values",
        type=str,
        default="1,3,5,10",
        help="Comma-separated list of K values for retrieval evaluation"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation benchmark"
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use sample dataset for testing"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_dir="./logs")

    # Initialize benchmark runner
    runner = BenchmarkRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    # Load dataset
    if args.use_sample:
        logger.info("Using sample dataset")
        dataset = BenchmarkDataset.create_sample_dataset(
            size=min(args.max_samples or 10, 10)
        )
    elif args.dataset:
        logger.info(f"Loading dataset from {args.dataset}")
        dataset = load_benchmark_dataset(
            args.dataset,
            args.dataset_type,
            args.max_samples
        )
    else:
        logger.warning("No dataset provided, using sample dataset")
        dataset = BenchmarkDataset.create_sample_dataset(size=5)

    # Parse top-k values
    top_k_values = [int(x.strip()) for x in args.top_k_values.split(",")]

    # Run benchmark
    result = runner.run_full_benchmark(
        dataset=dataset,
        max_samples=args.max_samples,
        top_k_values=top_k_values
    )

    # Print results summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dataset: {result.dataset_name}")
    print(f"Model: {result.model}")
    print(f"Embedding Model: {result.embedding_model}")
    print("\nRetrieval Metrics:")
    for metric, value in result.retrieval_metrics.items():
        if 'latency' not in metric:
            print(f"  {metric}: {value:.4f}")
    print("\nGeneration Metrics:")
    for metric, value in result.generation_metrics.items():
        if 'latency' not in metric:
            print(f"  {metric}: {value:.4f}")
    print("\nLatency Metrics:")
    for metric, value in result.latency_metrics.items():
        print(f"  {metric}: {value:.2f}ms")
    print("\n" + "=" * 60)
    print(f"Results saved to: {args.output_dir}")

    return result


if __name__ == "__main__":
    main()

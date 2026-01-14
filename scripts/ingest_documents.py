#!/usr/bin/env python3
"""
Document ingestion pipeline script.
Loads documents, chunks them, generates embeddings, and stores in vector database.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import ChunkingPipeline, ChunkingStrategy
from src.ingestion.embedding_generator import EmbeddingGeneratorPipeline, OpenAIEmbeddingGenerator
from src.retrieval.vector_store import FAISSVectorStore, FAISSHybridStore
from src.utils.config import get_config, get_config_manager
from src.utils.logger import get_logger, setup_logging, performance_logger

logger = get_logger(__name__)


class DocumentIngestionPipeline:
    """Complete document ingestion pipeline."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: str = "recursive",
        batch_size: int = 20,
        use_cache: bool = True,
        vector_store_type: str = "faiss",
        index_type: str = "HNSW64"
    ):
        """
        Initialize ingestion pipeline.

        Args:
            config_path: Path to configuration file
            model: Embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            chunking_strategy: Chunking strategy to use
            batch_size: Batch size for embedding generation
            use_cache: Whether to use embedding cache
            vector_store_type: Type of vector store
            index_type: FAISS index type
        """
        # Load configuration
        if config_path:
            self.config = get_config(config_path)
        else:
            self.config = get_config()

        # Override with parameters
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.vector_store_type = vector_store_type
        self.index_type = index_type

        # Initialize components
        self.document_loader = DocumentLoader()
        self.chunking_pipeline = ChunkingPipeline(
            strategy=ChunkingStrategy(chunking_strategy),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Initialize embedding generator
        api_key = self.config.get_embedding_api_key()
        self.embedding_pipeline = EmbeddingGeneratorPipeline(
            model=model,
            api_key=api_key,
            batch_size=batch_size,
            use_cache=use_cache
        )

        # Initialize vector store
        dimension = self.embedding_pipeline.generator.get_embedding_dimension()
        self.vector_store = self._create_vector_store(dimension)

        # Statistics
        self.stats = {
            "documents_loaded": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "failed_documents": 0,
            "total_time_seconds": 0,
            "start_time": None,
            "end_time": None
        }

        logger.info(f"Initialized ingestion pipeline with model={model}, chunk_size={chunk_size}")

    def _create_vector_store(self, dimension: int):
        """Create vector store instance."""
        if self.vector_store_type == "faiss":
            return FAISSVectorStore(
                dimension=dimension,
                index_type=self.index_type,
                metric="cosine"
            )
        elif self.vector_store_type == "hybrid":
            return FAISSHybridStore(
                dimension=dimension,
                index_type=self.index_type,
                metric="cosine"
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")

    def load_documents(self, paths: List[str], recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Load documents from file paths or directories.

        Args:
            paths: List of file paths or directories
            recursive: Whether to scan directories recursively

        Returns:
            List of loaded documents
        """
        documents = []

        for path in paths:
            path_obj = Path(path)

            if not path_obj.exists():
                logger.warning(f"Path does not exist: {path}")
                continue

            if path_obj.is_file():
                try:
                    doc = self.document_loader.load_document(str(path_obj))
                    documents.append(doc)
                    logger.info(f"Loaded document: {path_obj.name}")
                except Exception as e:
                    logger.error(f"Failed to load {path_obj}: {e}")
                    self.stats["failed_documents"] += 1

            elif path_obj.is_dir():
                # Load all documents in directory
                extensions = self.config.processing.supported_extensions
                docs = self.document_loader.load_directory(str(path_obj), extensions)
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from directory: {path_obj}")

        self.stats["documents_loaded"] = len(documents)
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk documents into smaller pieces.

        Args:
            documents: List of document dictionaries

        Returns:
            List of chunk dictionaries with text and metadata
        """
        all_chunks = []

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Chunk the document
            chunks = self.chunking_pipeline.chunk_document(content, metadata)

            # Convert to dictionary format
            for chunk in chunks:
                chunk_dict = {
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "chunk_index": chunk.index,
                    "document_path": metadata.get("file_path", ""),
                    "document_name": Path(metadata.get("file_path", "")).name
                }
                all_chunks.append(chunk_dict)

        self.stats["chunks_created"] = len(all_chunks)
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        # Log chunk statistics
        stats = self.chunking_pipeline.get_chunk_stats([c for c in all_chunks])
        logger.info(f"Chunk stats: avg_size={stats['avg_size']:.0f}, "
                   f"min={stats['min_size']}, max={stats['max_size']}")

        return all_chunks

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of EmbeddingResult objects
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return []

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        with performance_logger(__name__, "embedding_generation"):
            embedding_results = self.embedding_pipeline.generate_embeddings(chunks)

        self.stats["embeddings_generated"] = len(embedding_results)

        # Log cost information
        stats = self.embedding_pipeline.get_stats()
        logger.info(f"Generated {len(embedding_results)} embeddings")
        logger.info(f"  Cache hits: {stats['cache_hits']}, misses: {stats['cache_misses']}")
        logger.info(f"  Total tokens: {stats['total_tokens']}, estimated cost: ${stats['total_cost']:.6f}")

        return embedding_results

    def store_embeddings(self, embedding_results: List) -> None:
        """
        Store embeddings in vector store.

        Args:
            embedding_results: List of EmbeddingResult objects
        """
        if not embedding_results:
            logger.warning("No embeddings to store")
            return

        # Extract data for vector store
        embeddings = [result.embedding for result in embedding_results]
        texts = [result.text for result in embedding_results]
        metadata_list = [result.metadata for result in embedding_results]
        chunk_ids = [f"chunk_{i}" for i in range(len(embedding_results))]

        # Add to vector store
        with performance_logger(__name__, "vector_store_insertion"):
            indices = self.vector_store.add_embeddings(
                embeddings=embeddings,
                texts=texts,
                metadata=metadata_list,
                chunk_ids=chunk_ids
            )

        logger.info(f"Stored {len(indices)} embeddings in vector store")

    def save_vector_store(self, output_path: str) -> None:
        """Save vector store to disk."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / "vector_index"
        self.vector_store.save(str(save_path))

        # Save metadata about the index
        metadata = {
            "created_at": datetime.now().isoformat(),
            "model": self.model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunking_strategy": self.chunking_strategy,
            "total_documents": self.stats["documents_loaded"],
            "total_chunks": self.stats["chunks_created"],
            "total_embeddings": self.stats["embeddings_generated"],
            "vector_store_type": self.vector_store_type,
            "index_type": self.index_type
        }

        metadata_path = output_dir / "index_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved vector store to {save_path}")
        logger.info(f"Saved index metadata to {metadata_path}")

    def load_vector_store(self, input_path: str) -> None:
        """Load vector store from disk."""
        index_path = Path(input_path) / "vector_index"

        if not index_path.exists():
            raise FileNotFoundError(f"Vector store not found: {index_path}")

        self.vector_store.load(str(index_path))

        # Load metadata
        metadata_path = Path(input_path) / "index_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                logger.info(f"Loaded index metadata: {metadata}")

        logger.info(f"Loaded vector store with {self.vector_store.get_size()} vectors")

    def run(
        self,
        input_paths: List[str],
        output_path: Optional[str] = None,
        save_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete ingestion pipeline.

        Args:
            input_paths: List of input files or directories
            output_path: Path to save vector store
            save_intermediate: Whether to save intermediate results

        Returns:
            Dictionary with pipeline statistics
        """
        logger.info("=" * 60)
        logger.info("Starting document ingestion pipeline")
        logger.info("=" * 60)

        self.stats["start_time"] = datetime.now()

        try:
            # Step 1: Load documents
            logger.info("\n[Step 1/4] Loading documents...")
            documents = self.load_documents(input_paths)

            if not documents:
                logger.error("No documents loaded. Exiting.")
                return self.stats

            # Optionally save loaded documents
            if save_intermediate:
                self._save_intermediate(documents, "loaded_documents.json")

            # Step 2: Chunk documents
            logger.info("\n[Step 2/4] Chunking documents...")
            chunks = self.chunk_documents(documents)

            if not chunks:
                logger.error("No chunks created. Exiting.")
                return self.stats

            if save_intermediate:
                self._save_intermediate(chunks, "chunks.json")

            # Step 3: Generate embeddings
            logger.info("\n[Step 3/4] Generating embeddings...")
            embeddings = self.generate_embeddings(chunks)

            if not embeddings:
                logger.error("No embeddings generated. Exiting.")
                return self.stats

            # Step 4: Store in vector database
            logger.info("\n[Step 4/4] Storing embeddings...")
            self.store_embeddings(embeddings)

            # Save vector store if output path provided
            if output_path:
                self.save_vector_store(output_path)

            self.stats["end_time"] = datetime.now()
            self.stats["total_time_seconds"] = (
                self.stats["end_time"] - self.stats["start_time"]
            ).total_seconds()

            # Print summary
            self._print_summary()

            return self.stats

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.stats["error"] = str(e)
            return self.stats

    def _save_intermediate(self, data: Any, filename: str) -> None:
        """Save intermediate data for debugging."""
        import pickle

        output_dir = Path("./data/intermediate")
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename

        if filename.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

        logger.debug(f"Saved intermediate data to {filepath}")

    def _print_summary(self) -> None:
        """Print pipeline summary."""
        logger.info("\n" + "=" * 60)
        logger.info("INGESTION PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Documents loaded:     {self.stats['documents_loaded']}")
        logger.info(f"Failed documents:      {self.stats['failed_documents']}")
        logger.info(f"Chunks created:        {self.stats['chunks_created']}")
        logger.info(f"Embeddings generated:  {self.stats['embeddings_generated']}")
        logger.info(f"Vector store size:     {self.vector_store.get_size()}")
        logger.info(f"Total time:            {self.stats['total_time_seconds']:.2f} seconds")

        # Model info
        model_info = self.embedding_pipeline.generator.get_model_info()
        logger.info(f"Embedding model:       {model_info['model']}")
        logger.info(f"Embedding dimension:   {model_info['dimension']}")

        # Cache stats
        cache_stats = self.embedding_pipeline.get_stats()
        if cache_stats['cache_hits'] + cache_stats['cache_misses'] > 0:
            hit_rate = cache_stats['cache_hits'] / (cache_stats['cache_hits'] + cache_stats['cache_misses'])
            logger.info(f"Cache hit rate:        {hit_rate:.2%}")

        logger.info("=" * 60)


def main():
    """Main entry point for ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the DocQA AI system"
    )

    # Input arguments
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files or directories to ingest"
    )
    parser.add_argument(
        "-o", "--output",
        default="./data/vector_store",
        help="Output directory for vector store (default: ./data/vector_store)"
    )

    # Processing options
    parser.add_argument(
        "-c", "--config",
        help="Path to configuration file"
    )
    parser.add_argument(
        "-m", "--model",
        default="text-embedding-3-small",
        choices=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        help="Embedding model to use"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks in characters (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)"
    )
    parser.add_argument(
        "--chunking-strategy",
        default="recursive",
        choices=["fixed_size", "sentence", "paragraph", "recursive", "sliding_window", "markdown"],
        help="Chunking strategy to use (default: recursive)"
    )

    # Embedding options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for embedding generation (default: 20)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache"
    )

    # Vector store options
    parser.add_argument(
        "--vector-store",
        default="faiss",
        choices=["faiss", "hybrid"],
        help="Type of vector store (default: faiss)"
    )
    parser.add_argument(
        "--index-type",
        default="HNSW64",
        choices=["FlatIP", "FlatL2", "HNSW32", "HNSW64", "IVF"],
        help="FAISS index type (default: HNSW64)"
    )

    # Other options
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan directories (default: True)"
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results for debugging"
    )
    parser.add_argument(
        "--load-existing",
        help="Load existing vector store from path (incremental ingestion)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_dir="./logs")

    logger.info("DocQA AI Document Ingestion Pipeline")
    logger.info(f"Inputs: {args.inputs}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")

    # Create pipeline
    pipeline = DocumentIngestionPipeline(
        config_path=args.config,
        model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.chunking_strategy,
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
        vector_store_type=args.vector_store,
        index_type=args.index_type
    )

    # Load existing vector store if specified
    if args.load_existing:
        try:
            pipeline.load_vector_store(args.load_existing)
            logger.info(f"Loaded existing vector store with {pipeline.vector_store.get_size()} vectors")
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {e}")

    # Run pipeline
    stats = pipeline.run(
        input_paths=args.inputs,
        output_path=args.output,
        save_intermediate=args.save_intermediate
    )

    # Exit with appropriate code
    if stats.get("error"):
        sys.exit(1)
    else:
        logger.info("\n✅ Ingestion completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

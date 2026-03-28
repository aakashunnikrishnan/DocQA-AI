"""
Integration tests for DocQA AI system.
"""

import pytest
import tempfile
from pathlib import Path

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import ChunkingPipeline
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import VectorRetriever
from src.generation.llm_interface import LLMInterface
from src.generation.prompt_templates import get_rag_prompt
from src.generation.response_postprocess import postprocess_response


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self, sample_documents, mock_embedding_generator, mock_llm_interface):
        """Test the full RAG pipeline."""
        # Step 1: Load documents (skip actual loading, use sample data)
        docs = sample_documents

        # Step 2: Chunk documents
        chunker = ChunkingPipeline(chunk_size=200, chunk_overlap=50)
        chunks = []
        for doc in docs:
            chunks.extend(chunker.chunk_document(doc["content"], {"source": doc["name"]}))

        assert len(chunks) > 0

        # Step 3: Generate embeddings
        chunk_data = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        embeddings = mock_embedding_generator.generate_embeddings(chunk_data)

        assert len(embeddings) == len(chunks)

        # Step 4: Store in vector store
        dimension = 384
        vector_store = FAISSVectorStore(dimension=dimension)
        vector_store.add_embeddings(
            embeddings=[e.embedding for e in embeddings],
            texts=[e.text for e in embeddings],
            metadata=[e.metadata for e in embeddings]
        )

        assert vector_store.get_size() > 0

        # Step 5: Create retriever
        retriever = VectorRetriever(
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=3
        )

        # Step 6: Query
        query = "What is machine learning?"
        results = retriever.retrieve(query)

        assert len(results) > 0

        # Step 7: Generate response
        context_chunks = [{"text": r.text, "source": "test"} for r in results]
        prompt = get_rag_prompt(question=query, chunks=context_chunks)
        response = mock_llm_interface.generate_simple(prompt)

        assert response is not None
        assert len(response) > 0

        # Step 8: Post-process
        processed = postprocess_response(response, aggressive_cleaning=True)
        assert processed.cleaned_text is not None

    def test_ingestion_pipeline(self, temp_dir, sample_documents):
        """Test the ingestion pipeline."""
        # Create sample files
        files = []
        for i, doc in enumerate(sample_documents[:2]):
            file_path = temp_dir / f"doc{i}.txt"
            file_path.write_text(doc["content"])
            files.append(str(file_path))

        # Load documents
        loader = DocumentLoader()
        loaded_docs = []
        for file_path in files:
            loaded_docs.append(loader.load_document(file_path))

        assert len(loaded_docs) == 2

        # Chunk documents
        chunker = ChunkingPipeline(chunk_size=300, chunk_overlap=50)
        chunks = []
        for doc in loaded_docs:
            chunks.extend(chunker.chunk_document(doc["content"], doc["metadata"]))

        assert len(chunks) > 0

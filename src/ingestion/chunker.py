"""
Text chunking strategies for splitting documents into manageable pieces for embedding and retrieval.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SLIDING_WINDOW = "sliding_window"
    MARKDOWN = "markdown"
    CODE = "code"


@dataclass
class Chunk:
    """Represents a single text chunk."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    index: int = 0
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char
        }


class BaseChunker:
    """Base class for all chunkers."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into chunks."""
        raise NotImplementedError

    def _create_chunk(self, text: str, index: int, start_char: int, end_char: int,
                     metadata: Optional[Dict[str, Any]] = None) -> Chunk:
        """Helper to create a chunk with metadata."""
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata["chunk_index"] = index
        chunk_metadata["chunk_size"] = len(text)

        return Chunk(
            text=text.strip(),
            metadata=chunk_metadata,
            index=index,
            start_char=start_char,
            end_char=end_char
        )


class FixedSizeChunker(BaseChunker):
    """Chunk text by fixed number of characters."""

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        chunks = []
        start = 0
        text_length = len(text)
        index = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            # Try to cut at a space or newline for better readability
            if end < text_length:
                # Look for last space within overlap region
                search_start = max(start, end - 50)
                last_space = text.rfind(' ', search_start, end)
                if last_space != -1 and last_space > start:
                    end = last_space

            chunk_text = text[start:end]
            chunks.append(self._create_chunk(chunk_text, index, start, end, metadata))

            start = end - self.chunk_overlap
            index += 1

        return chunks


class SentenceChunker(BaseChunker):
    """Chunk text by sentences."""

    def __init__(self, chunk_size: int = 5, chunk_overlap: int = 1, **kwargs):
        """
        Args:
            chunk_size: Number of sentences per chunk
            chunk_overlap: Number of overlapping sentences
        """
        super().__init__(chunk_size, chunk_overlap)
        self.sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        # Split into sentences
        sentences = re.split(self.sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        index = 0
        step = self.chunk_size - self.chunk_overlap

        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i:i + self.chunk_size]
            if not chunk_sentences:
                continue

            chunk_text = ' '.join(chunk_sentences)

            # Calculate character positions (approximate)
            start_char = text.find(chunk_sentences[0])
            end_char = text.find(chunk_sentences[-1]) + len(chunk_sentences[-1])

            chunks.append(self._create_chunk(chunk_text, index, start_char, end_char, metadata))
            index += 1

        return chunks


class ParagraphChunker(BaseChunker):
    """Chunk text by paragraphs."""

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_size = 0
        index = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > self.chunk_size and current_chunk:
                # Create chunk from accumulated paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))
                index += 1

                # Keep overlap paragraphs if specified
                overlap_paras = []
                overlap_size = 0
                for p in reversed(current_chunk):
                    if overlap_size + len(p) <= self.chunk_overlap:
                        overlap_paras.insert(0, p)
                        overlap_size += len(p)
                    else:
                        break
                current_chunk = overlap_paras
                current_size = overlap_size

            current_chunk.append(para)
            current_size += para_size

        # Add last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursively split text using a hierarchy of separators."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        return self._recursive_split(text, self.separators, metadata)

    def _recursive_split(self, text: str, separators: List[str],
                         metadata: Optional[Dict[str, Any]] = None,
                         start_pos: int = 0) -> List[Chunk]:
        """Recursively split text using separators."""
        chunks = []

        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(self._create_chunk(text, len(chunks), start_pos,
                                                start_pos + len(text), metadata))
            return chunks

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Final fallback - split by fixed size
            return FixedSizeChunker(self.chunk_size, self.chunk_overlap).chunk(text, metadata)

        splits = text.split(separator)

        current_chunk = []
        current_size = 0

        for split in splits:
            split_with_sep = split + separator if split != splits[-1] else split
            split_size = len(split_with_sep)

            if current_size + split_size > self.chunk_size and current_chunk:
                # Create chunk from accumulated splits
                chunk_text = separator.join(current_chunk)
                chunk_start = start_pos + text.find(current_chunk[0])
                chunk_end = chunk_start + len(chunk_text)
                chunks.append(self._create_chunk(chunk_text, len(chunks),
                                                chunk_start, chunk_end, metadata))

                # Handle overlap
                overlap_text = separator.join(current_chunk[-self.chunk_overlap:]) if self.chunk_overlap > 0 else ""
                if overlap_text and remaining_separators:
                    current_chunk = [overlap_text] if overlap_text else []
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(split)
            current_size += split_size

        # Add remaining text
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunk_start = start_pos + text.find(current_chunk[0])
            chunk_end = chunk_start + len(chunk_text)
            chunks.append(self._create_chunk(chunk_text, len(chunks),
                                            chunk_start, chunk_end, metadata))

        return chunks


class SemanticChunker(BaseChunker):
    """Chunk text based on semantic similarity between sentences."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 similarity_threshold: float = 0.5):
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        # Simple implementation using sentence boundaries and length
        # For full semantic chunking, would need embeddings model

        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_size = 0
        index = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # Check if adding sentence exceeds chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))
                index += 1

                # Keep last few sentences for overlap
                overlap_count = 0
                overlap_size = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                        overlap_count += 1
                    else:
                        break
                current_chunk = overlap_sentences
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))

        return chunks


class SlidingWindowChunker(BaseChunker):
    """Create overlapping chunks using a sliding window approach."""

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        chunks = []
        window_size = self.chunk_size
        step = window_size - self.chunk_overlap

        if step <= 0:
            step = window_size // 2
            logger.warning(f"Overlap larger than chunk size, using step={step}")

        for i in range(0, len(text), step):
            chunk_text = text[i:i + window_size]
            if len(chunk_text) < 50 and i > 0:
                continue  # Skip very small trailing chunks

            chunks.append(self._create_chunk(
                chunk_text, len(chunks), i, min(i + window_size, len(text)), metadata
            ))

        return chunks


class MarkdownChunker(BaseChunker):
    """Specialized chunker for Markdown documents respecting headers."""

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        chunks = []
        lines = text.split('\n')

        current_chunk = []
        current_header = "root"
        current_size = 0

        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # Save previous chunk if exists
                if current_chunk and current_size > 0:
                    chunk_text = '\n'.join(current_chunk)
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["header"] = current_header
                    chunks.append(self._create_chunk(chunk_text, len(chunks),
                                                    0, 0, chunk_metadata))

                # Start new chunk with header
                current_chunk = [line]
                current_header = header_match.group(2)
                current_size = len(line)
            else:
                # Check if adding line exceeds size
                if current_size + len(line) > self.chunk_size and current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["header"] = current_header
                    chunks.append(self._create_chunk(chunk_text, len(chunks),
                                                    0, 0, chunk_metadata))

                    # Keep last few lines for overlap
                    overlap_lines = []
                    overlap_size = 0
                    for l in reversed(current_chunk):
                        if overlap_size + len(l) <= self.chunk_overlap:
                            overlap_lines.insert(0, l)
                            overlap_size += len(l)
                        else:
                            break
                    current_chunk = overlap_lines
                    current_size = overlap_size

                current_chunk.append(line)
                current_size += len(line)

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["header"] = current_header
            chunks.append(self._create_chunk(chunk_text, len(chunks), 0, 0, chunk_metadata))

        return chunks


class CodeChunker(BaseChunker):
    """Specialized chunker for code files preserving semantic units."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200,
                 language: str = "python"):
        super().__init__(chunk_size, chunk_overlap)
        self.language = language

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        lines = text.split('\n')

        chunks = []
        current_chunk = []
        current_size = 0
        in_function = False
        function_indent = 0

        for i, line in enumerate(lines):
            # Detect function/class definitions
            if re.match(r'^(def|class)\s+', line.strip()):
                if current_chunk and current_size > 0:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, len(chunks),
                                                    0, 0, metadata))

                    # Keep overlap
                    if self.chunk_overlap > 0:
                        overlap_lines = current_chunk[-self.chunk_overlap//10:]
                        current_chunk = overlap_lines
                        current_size = sum(len(l) for l in overlap_lines)
                    else:
                        current_chunk = []
                        current_size = 0

                in_function = True
                function_indent = len(line) - len(line.lstrip())

            current_chunk.append(line)
            current_size += len(line)

            # Check size limit
            if current_size >= self.chunk_size and not in_function:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, len(chunks),
                                                0, 0, metadata))

                # Keep overlap
                overlap_size = 0
                overlap_lines = []
                for l in reversed(current_chunk):
                    if overlap_size + len(l) <= self.chunk_overlap:
                        overlap_lines.insert(0, l)
                        overlap_size += len(l)
                    else:
                        break
                current_chunk = overlap_lines
                current_size = overlap_size

            # Reset function flag when indentation returns
            if in_function and line.strip() and (len(line) - len(line.lstrip())) <= function_indent:
                in_function = False

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, len(chunks), 0, 0, metadata))

        return chunks


class ChunkingPipeline:
    """Main pipeline for chunking documents with different strategies."""

    def __init__(self, strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
                 chunk_size: int = 1000, chunk_overlap: int = 200,
                 **kwargs):
        """
        Initialize chunking pipeline.

        Args:
            strategy: Chunking strategy to use
            chunk_size: Size of each chunk (characters or units based on strategy)
            chunk_overlap: Overlap between chunks
            **kwargs: Additional strategy-specific parameters
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs
        self.chunker = self._create_chunker()

    def _create_chunker(self) -> BaseChunker:
        """Create chunker based on selected strategy."""
        strategies = {
            ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
            ChunkingStrategy.SENTENCE: SentenceChunker,
            ChunkingStrategy.PARAGRAPH: ParagraphChunker,
            ChunkingStrategy.SEMANTIC: SemanticChunker,
            ChunkingStrategy.RECURSIVE: RecursiveChunker,
            ChunkingStrategy.SLIDING_WINDOW: SlidingWindowChunker,
            ChunkingStrategy.MARKDOWN: MarkdownChunker,
            ChunkingStrategy.CODE: CodeChunker,
        }

        chunker_class = strategies.get(self.strategy)
        if not chunker_class:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return chunker_class(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            **self.kwargs
        )

    def chunk_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk a single document.

        Args:
            text: Document text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        logger.info(f"Chunking document with strategy {self.strategy.value}, "
                   f"size={self.chunk_size}, overlap={self.chunk_overlap}")

        chunks = self.chunker.chunk(text, metadata)

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def chunk_batch(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries with 'content' and optional 'metadata'

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})

            chunks = self.chunk_document(content, metadata)
            all_chunks.extend(chunks)

        return all_chunks

    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunks."""
        if not chunks:
            return {"total_chunks": 0, "avg_size": 0, "min_size": 0, "max_size": 0}

        sizes = [len(chunk.text) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_chars": sum(sizes)
        }


# Convenience function
def chunk_text(text: str, strategy: str = "recursive",
               chunk_size: int = 1000, chunk_overlap: int = 200,
               **kwargs) -> List[Chunk]:
    """
    Quick helper function to chunk text.

    Args:
        text: Text to chunk
        strategy: Chunking strategy ('fixed_size', 'sentence', 'paragraph', 'recursive')
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks

    Returns:
        List of Chunk objects
    """
    pipeline = ChunkingPipeline(
        strategy=ChunkingStrategy(strategy),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
    return pipeline.chunk_document(text)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    sample_text = """
    This is the first paragraph. It contains multiple sentences. Here's another sentence.
    
    This is the second paragraph. It has different content.
    
    And a third paragraph with more information to demonstrate chunking.
    """

    # Test different strategies
    strategies = ["fixed_size", "sentence", "paragraph", "recursive"]

    for strat in strategies:
        chunks = chunk_text(sample_text, strategy=strat, chunk_size=100, chunk_overlap=20)
        print(f"\n{strat.upper()}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:2]):
            print(f"  Chunk {i}: {chunk.text[:50]}...")

"""
Text chunking strategies for splitting documents into manageable pieces for embedding and retrieval.
FIXED: Chunk overlap calculation bug in FixedSizeChunker and RecursiveChunker.
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

        # Validate overlap
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(f"Chunk overlap ({chunk_overlap}) >= chunk size ({chunk_size}). "
                          f"Setting overlap to {chunk_size // 2}")
            self.chunk_overlap = chunk_size // 2

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
    """Chunk text by fixed number of characters with proper overlap calculation."""

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Split text into fixed-size chunks with overlap.

        FIX: Properly calculate overlap by ensuring we don't go backwards and
        correctly handle the overlap region.
        """
        if not text or not text.strip():
            return []

        chunks = []
        text_length = len(text)

        # Calculate step size (how much we move forward each time)
        step = self.chunk_size - self.chunk_overlap

        # Ensure we make progress
        if step <= 0:
            logger.warning(f"Step size ({step}) <= 0. Setting to chunk_size // 2")
            step = max(1, self.chunk_size // 2)

        index = 0
        start = 0

        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)

            # Try to find a good break point (space, newline, punctuation)
            if end < text_length:
                # Look for best break point within the last 50 characters
                search_start = max(start, end - 50)
                best_break = end

                # Priority: newline > space > punctuation
                for pattern in [r'\n', r'\s', r'[.!?]']:
                    matches = list(re.finditer(pattern, text[search_start:end]))
                    if matches:
                        last_match = matches[-1]
                        break_pos = search_start + last_match.start() + 1
                        if break_pos > start and break_pos < end:
                            best_break = break_pos
                            break

                end = best_break

            # Extract chunk
            chunk_text = text[start:end]

            # Only add non-empty chunks
            if chunk_text.strip():
                chunks.append(self._create_chunk(
                    chunk_text, index, start, end, metadata
                ))
                index += 1

            # Move start position with overlap
            # FIX: Properly calculate next start position
            if end >= text_length:
                break

            # Calculate overlap region
            overlap_size = min(self.chunk_overlap, end - start)
            start = end - overlap_size

            # Ensure we actually move forward (prevent infinite loop)
            if start <= end - overlap_size and start < text_length:
                # If we didn't move enough, force progress
                start = end - max(1, overlap_size // 2)

            # Final safety check - ensure progress
            if start <= 0 and text_length > 0:
                start = 1

            # Prevent infinite loop
            if start >= text_length:
                break

        # Verify no overlapping issues
        self._validate_chunks(chunks, text)

        return chunks

    def _validate_chunks(self, chunks: List[Chunk], original_text: str) -> None:
        """Validate chunk integrity and overlap."""
        if not chunks:
            return

        # Check for overlapping content
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i].text
            next_chunk = chunks[i + 1].text

            # Find overlap between chunks
            overlap_found = False

            # Check if end of current chunk appears in start of next chunk
            overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
            if overlap_text and overlap_text in next_chunk[:len(overlap_text)]:
                overlap_found = True

            if not overlap_found and self.chunk_overlap > 0:
                # Check for smaller overlap
                for overlap_len in [50, 100, 150]:
                    if current_chunk[-overlap_len:] in next_chunk[:overlap_len]:
                        overlap_found = True
                        break

            if not overlap_found and self.chunk_overlap > 0:
                logger.debug(f"Chunk {i} and {i+1} may not have expected overlap")

        # Check for content loss
        reconstructed = "".join([chunk.text for chunk in chunks])
        if len(reconstructed) < len(original_text) * 0.9:  # Allow some trimming
            logger.warning(f"Possible content loss: {len(reconstructed)} vs {len(original_text)}")


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

        # Validate sentence overlap
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(f"Sentence overlap ({chunk_overlap}) >= chunk size ({chunk_size}). "
                          f"Setting overlap to {max(1, chunk_size // 2)}")
            self.chunk_overlap = max(1, chunk_size // 2)

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into sentence-based chunks with proper overlap."""
        if not text or not text.strip():
            return []

        # Split into sentences
        sentences = re.split(self.sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks = []
        index = 0

        # Calculate step size (how many sentences to move forward)
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = max(1, self.chunk_size // 2)

        for i in range(0, len(sentences), step):
            # Get sentence window
            end_idx = min(i + self.chunk_size, len(sentences))
            chunk_sentences = sentences[i:end_idx]

            if not chunk_sentences:
                continue

            chunk_text = ' '.join(chunk_sentences)

            # Find approximate character positions
            try:
                start_char = text.find(chunk_sentences[0])
                end_char = text.find(chunk_sentences[-1]) + len(chunk_sentences[-1])
            except (ValueError, IndexError):
                start_char = 0
                end_char = 0

            chunks.append(self._create_chunk(
                chunk_text, index, start_char, end_char, metadata
            ))
            index += 1

            # Stop if we've reached the end
            if end_idx >= len(sentences):
                break

        return chunks


class ParagraphChunker(BaseChunker):
    """Chunk text by paragraphs with proper overlap."""

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into paragraph-based chunks."""
        if not text or not text.strip():
            return []

        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            # If no paragraphs, treat as single chunk
            return [self._create_chunk(text, 0, 0, len(text), metadata)]

        chunks = []
        current_chunk = []
        current_size = 0
        index = 0

        for para in paragraphs:
            para_size = len(para)

            # Check if adding this paragraph would exceed chunk size
            if current_size + para_size > self.chunk_size and current_chunk:
                # Create chunk from accumulated paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))
                index += 1

                # Calculate overlap paragraphs
                overlap_paras = []
                overlap_size = 0

                # Keep paragraphs from end that fit within overlap
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

        # Validate overlap
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(f"Overlap ({chunk_overlap}) >= chunk size ({chunk_size}). "
                          f"Setting overlap to {chunk_size // 3}")
            self.chunk_overlap = chunk_size // 3

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Recursively split text with proper overlap handling."""
        if not text or not text.strip():
            return []

        return self._recursive_split(text, self.separators, metadata)

    def _recursive_split(self, text: str, separators: List[str],
                         metadata: Optional[Dict[str, Any]] = None,
                         start_pos: int = 0, depth: int = 0) -> List[Chunk]:
        """Recursively split text using separators with overlap."""
        chunks = []

        # Base case: text fits in one chunk
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(self._create_chunk(
                    text, len(chunks), start_pos, start_pos + len(text), metadata
                ))
            return chunks

        # No more separators to try - fall back to fixed size
        if not separators:
            return self._fixed_size_split_with_overlap(text, metadata, start_pos)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Handle empty separator (character-level splitting)
        if separator == "":
            return self._fixed_size_split_with_overlap(text, metadata, start_pos)

        # Split by current separator
        splits = text.split(separator)

        # Rebuild with separators
        parts = []
        for i, split in enumerate(splits):
            parts.append(split)
            if i < len(splits) - 1:
                parts.append(separator)

        # Group into chunks
        current_chunk_parts = []
        current_size = 0
        chunk_start_pos = start_pos

        for i, part in enumerate(parts):
            part_size = len(part)

            # Check if adding this part would exceed chunk size
            if current_size + part_size > self.chunk_size and current_chunk_parts:
                # Create chunk from accumulated parts
                chunk_text = ''.join(current_chunk_parts)
                chunk_end_pos = chunk_start_pos + len(chunk_text)

                chunks.append(self._create_chunk(
                    chunk_text, len(chunks), chunk_start_pos, chunk_end_pos, metadata
                ))

                # FIX: Properly calculate overlap for recursive chunker
                # Keep overlapping parts from the end of current chunk
                overlap_parts = []
                overlap_size = 0

                # Build overlap by adding parts from end until overlap size reached
                for part_idx in range(len(current_chunk_parts) - 1, -1, -1):
                    part_to_add = current_chunk_parts[part_idx]
                    if overlap_size + len(part_to_add) <= self.chunk_overlap:
                        overlap_parts.insert(0, part_to_add)
                        overlap_size += len(part_to_add)
                    else:
                        # If we can't add the whole part, try to add part of it
                        remaining = self.chunk_overlap - overlap_size
                        if remaining > 0:
                            overlap_parts.insert(0, part_to_add[:remaining])
                            overlap_size += remaining
                        break

                # Update for next chunk
                current_chunk_parts = overlap_parts
                current_size = overlap_size
                chunk_start_pos = chunk_end_pos - overlap_size if overlap_size > 0 else chunk_end_pos

                # If we have overlapping content, continue with next part
                # Don't add the current part again if it's already in overlap
                if overlap_parts and i < len(parts):
                    # Check if current part is already included in overlap
                    last_overlap = ''.join(overlap_parts)
                    if part in last_overlap:
                        continue

            current_chunk_parts.append(part)
            current_size += part_size

        # Add remaining content
        if current_chunk_parts:
            chunk_text = ''.join(current_chunk_parts)
            chunk_end_pos = chunk_start_pos + len(chunk_text)
            chunks.append(self._create_chunk(
                chunk_text, len(chunks), chunk_start_pos, chunk_end_pos, metadata
            ))

        # Recursively process any chunks that are still too large
        final_chunks = []
        for chunk in chunks:
            if len(chunk.text) > self.chunk_size:
                # Recursively split this chunk with next separator
                sub_chunks = self._recursive_split(
                    chunk.text, remaining_separators, chunk.metadata,
                    chunk.start_char, depth + 1
                )
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _fixed_size_split_with_overlap(self, text: str, metadata: Optional[Dict[str, Any]],
                                       start_pos: int) -> List[Chunk]:
        """Fallback to fixed-size splitting with overlap."""
        chunks = []
        text_length = len(text)
        step = self.chunk_size - self.chunk_overlap

        if step <= 0:
            step = max(1, self.chunk_size // 2)

        for i in range(0, text_length, step):
            chunk_end = min(i + self.chunk_size, text_length)
            chunk_text = text[i:chunk_end]

            if chunk_text.strip():
                chunks.append(self._create_chunk(
                    chunk_text, len(chunks), start_pos + i, start_pos + chunk_end, metadata
                ))

            if chunk_end >= text_length:
                break

        return chunks


class SlidingWindowChunker(BaseChunker):
    """Create overlapping chunks using a sliding window approach."""

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text using sliding window with proper step calculation."""
        if not text or not text.strip():
            return []

        chunks = []
        text_length = len(text)

        # Calculate step size
        step = self.chunk_size - self.chunk_overlap

        # FIX: Ensure step is positive
        if step <= 0:
            logger.warning(f"Step size ({step}) invalid. Using step = {self.chunk_size // 2}")
            step = max(1, self.chunk_size // 2)

        for start in range(0, text_length, step):
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]

            # Skip very small chunks at the end (less than 10% of chunk size)
            if len(chunk_text) < self.chunk_size * 0.1 and start > 0:
                # If it's too small and not the first chunk, skip
                continue

            if chunk_text.strip():
                chunks.append(self._create_chunk(
                    chunk_text, len(chunks), start, end, metadata
                ))

            # Stop if we've reached the end
            if end >= text_length:
                break

        return chunks


class MarkdownChunker(BaseChunker):
    """Specialized chunker for Markdown documents respecting headers."""

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split markdown while preserving header structure."""
        if not text or not text.strip():
            return []

        lines = text.split('\n')

        chunks = []
        current_chunk = []
        current_header = "root"
        current_size = 0
        index = 0

        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # Save previous chunk if it exists and has content
                if current_chunk and current_size > 0:
                    chunk_text = '\n'.join(current_chunk)
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["header"] = current_header
                    chunks.append(self._create_chunk(
                        chunk_text, index, 0, 0, chunk_metadata
                    ))
                    index += 1

                    # Calculate overlap for markdown (keep previous header if it fits)
                    if self.chunk_overlap > 0 and current_header != "root":
                        # Keep the header line in overlap
                        header_line = current_chunk[0] if current_chunk else ""
                        if len(header_line) <= self.chunk_overlap:
                            current_chunk = [header_line]
                            current_size = len(header_line)
                        else:
                            current_chunk = []
                            current_size = 0
                    else:
                        current_chunk = []
                        current_size = 0

                # Start new chunk with header
                current_chunk = [line]
                current_header = header_match.group(2)
                current_size = len(line)
            else:
                # Check if adding line would exceed chunk size
                if current_size + len(line) > self.chunk_size and current_chunk:
                    # Create chunk before adding this line
                    chunk_text = '\n'.join(current_chunk)
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["header"] = current_header
                    chunks.append(self._create_chunk(
                        chunk_text, index, 0, 0, chunk_metadata
                    ))
                    index += 1

                    # Calculate overlap for content
                    overlap_size = 0
                    overlap_lines = []
                    for l in reversed(current_chunk):
                        if overlap_size + len(l) <= self.chunk_overlap:
                            overlap_lines.insert(0, l)
                            overlap_size += len(l)
                        else:
                            # Add part of the line if needed
                            remaining = self.chunk_overlap - overlap_size
                            if remaining > 0:
                                overlap_lines.insert(0, l[:remaining])
                                overlap_size += remaining
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
            chunks.append(self._create_chunk(chunk_text, index, 0, 0, chunk_metadata))

        return chunks


class CodeChunker(BaseChunker):
    """Specialized chunker for code files preserving semantic units."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200,
                 language: str = "python"):
        super().__init__(chunk_size, chunk_overlap)
        self.language = language

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split code while preserving function/class boundaries."""
        if not text or not text.strip():
            return []

        lines = text.split('\n')

        chunks = []
        current_chunk = []
        current_size = 0
        in_function = False
        function_indent = 0
        index = 0

        for i, line in enumerate(lines):
            # Detect function/class definitions
            if re.match(r'^(def|class)\s+', line.strip()):
                # Save previous chunk before starting new function
                if current_chunk and current_size > 0:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text, index, 0, 0, metadata
                    ))
                    index += 1

                    # Calculate overlap for code
                    if self.chunk_overlap > 0:
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
                chunks.append(self._create_chunk(
                    chunk_text, index, 0, 0, metadata
                ))
                index += 1

                # Calculate overlap for code when not in function
                if self.chunk_overlap > 0:
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
                else:
                    current_chunk = []
                    current_size = 0

            # Reset function flag when indentation returns to outer level
            if in_function and line.strip() and (len(line) - len(line.lstrip())) <= function_indent:
                if i > 0 and lines[i-1].strip() and not lines[i-1].strip().endswith(':'):
                    in_function = False

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text, index, 0, 0, metadata
            ))

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

        # Validate overlap
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(f"Overlap ({chunk_overlap}) >= chunk size ({chunk_size})")

        self.chunker = self._create_chunker()

    def _create_chunker(self) -> BaseChunker:
        """Create chunker based on selected strategy."""
        strategies = {
            ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
            ChunkingStrategy.SENTENCE: SentenceChunker,
            ChunkingStrategy.PARAGRAPH: ParagraphChunker,
            ChunkingStrategy.SEMANTIC: FixedSizeChunker,  # Fallback to fixed size
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

        # Validate overlaps
        self._validate_overlaps(chunks)

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def _validate_overlaps(self, chunks: List[Chunk]) -> None:
        """Validate that overlaps are working correctly."""
        if len(chunks) < 2 or self.chunk_overlap == 0:
            return

        # Check a few random chunk pairs for overlap
        sample_size = min(5, len(chunks) - 1)
        for i in range(sample_size):
            current = chunks[i].text
            next_chunk = chunks[i + 1].text

            # Check if there's any overlap
            overlap_found = False

            # Look for common text at boundaries
            overlap_len = min(100, self.chunk_overlap)
            if overlap_len > 0:
                current_end = current[-overlap_len:] if len(current) > overlap_len else current
                next_start = next_chunk[:overlap_len] if len(next_chunk) > overlap_len else next_chunk

                # Check if end of current appears in start of next
                if len(current_end) > 20 and current_end in next_start:
                    overlap_found = True

            if not overlap_found and self.chunk_overlap > 50:
                logger.debug(f"Chunk {i} and {i+1} may have less overlap than expected")

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
            "total_chars": sum(sizes),
            "overlap_setting": self.chunk_overlap
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
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    sample_text = """
    This is the first paragraph. It contains multiple sentences. Here's another sentence.
    
    This is the second paragraph. It has different content.
    
    And a third paragraph with more information to demonstrate chunking and overlap.
    """

    # Test fixed size with overlap
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=30)
    chunks = chunker.chunk(sample_text)

    print(f"\nFixed Size Chunker: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk.text)} chars - '{chunk.text[:50]}...'")

    # Test overlap validation
    print(f"\nOverlap test: {chunk_size=}, {chunk_overlap=}")
    print(f"Chunk 0 end: '{chunks[0].text[-30:]}'")
    print(f"Chunk 1 start: '{chunks[1].text[:30]}'")

    # Test various strategies
    strategies = ["fixed_size", "sentence", "paragraph", "recursive", "sliding_window"]

    for strat in strategies:
        chunks = chunk_text(sample_text, strategy=strat, chunk_size=80, chunk_overlap=20)
        print(f"\n{strat.upper()}: {len(chunks)} chunks")

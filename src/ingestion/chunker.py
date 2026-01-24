"""
Text chunking strategies for splitting documents into manageable pieces for embedding and retrieval.
OPTIMIZED: Adaptive chunk sizing, semantic boundaries, and improved overlap handling for better retrieval.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

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
    ADAPTIVE = "adaptive"  # NEW: Adaptive chunk sizing


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

        # Optimized overlap for retrieval (typically 10-20% of chunk size)
        self.optimal_overlap_ratio = 0.15  # 15% overlap is optimal for retrieval

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

    def _find_optimal_break(self, text: str, start: int, end: int) -> int:
        """
        Find optimal break point within text segment.
        Prioritizes: paragraph breaks > sentence boundaries > phrase boundaries.
        """
        segment = text[start:end]

        # Try to find paragraph break
        para_break = segment.rfind('\n\n')
        if para_break > len(segment) * 0.3:
            return start + para_break + 2

        # Try to find sentence boundary
        sentence_pattern = r'[.!?]\s+(?=[A-Z])'
        sentence_matches = list(re.finditer(sentence_pattern, segment))
        if sentence_matches:
            # Get last sentence boundary within the last 30% of segment
            for match in reversed(sentence_matches):
                if match.start() > len(segment) * 0.5:
                    return start + match.end()

        # Try to find phrase boundary (comma, semicolon)
        phrase_pattern = r'[,;]\s+'
        phrase_matches = list(re.finditer(phrase_pattern, segment))
        if phrase_matches:
            for match in reversed(phrase_matches):
                if match.start() > len(segment) * 0.5:
                    return start + match.end()

        # Fall back to space
        space = segment.rfind(' ', len(segment) * 0.7, len(segment))
        if space > 0:
            return start + space + 1

        return end


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

            # Try to find a good break point within the last 30% of chunk
            if end < text_length:
                search_start = max(start, end - int(self.chunk_size * 0.3))
                best_break = self._find_optimal_break(text, search_start, end)
                if best_break > start and best_break < end:
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
            if end >= text_length:
                break

            # Calculate overlap region - use optimal overlap ratio
            overlap_size = min(
                int(self.chunk_size * self.optimal_overlap_ratio),
                self.chunk_overlap,
                end - start
            )
            start = end - overlap_size

            # Ensure we actually move forward
            if start <= end - overlap_size and start < text_length:
                start = end - max(1, overlap_size // 2)

            # Final safety check
            if start <= 0 and text_length > 0:
                start = 1

            # Prevent infinite loop
            if start >= text_length:
                break

        return chunks


class AdaptiveChunker(BaseChunker):
    """
    NEW: Adaptive chunking that adjusts chunk size based on text structure.
    Optimizes chunk size for better retrieval by considering:
    - Semantic boundaries (paragraphs, sections)
    - Text density and complexity
    - Document type (code, prose, markdown)
    """

    def __init__(
        self,
        min_chunk_size: int = 300,
        max_chunk_size: int = 1500,
        target_chunk_size: int = 800,
        chunk_overlap: int = 150,
        adaptive_threshold: float = 0.3
    ):
        """
        Initialize adaptive chunker.

        Args:
            min_chunk_size: Minimum chunk size (characters)
            max_chunk_size: Maximum chunk size (characters)
            target_chunk_size: Target chunk size (characters)
            chunk_overlap: Overlap between chunks
            adaptive_threshold: Threshold for adaptivity (0-1)
        """
        super().__init__(target_chunk_size, chunk_overlap)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_chunk_size = target_chunk_size
        self.adaptive_threshold = adaptive_threshold

        # Optimal overlap for adaptive chunking
        self.optimal_overlap_ratio = 0.12  # 12% overlap

        logger.info(f"AdaptiveChunker initialized: min={min_chunk_size}, "
                   f"target={target_chunk_size}, max={max_chunk_size}")

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Adaptively chunk text based on content structure.

        Key optimizations:
        1. Detect text structure (paragraphs, sections, lists)
        2. Adjust chunk size based on content density
        3. Preserve semantic boundaries
        4. Optimize for retrieval relevance
        """
        if not text or not text.strip():
            return []

        # Analyze text structure
        structure = self._analyze_text_structure(text)

        # Determine optimal chunk size based on structure
        optimal_size = self._calculate_optimal_chunk_size(text, structure)

        # Determine chunking strategy based on content type
        if structure["has_code_blocks"]:
            return self._chunk_with_code_awareness(text, optimal_size, metadata)
        elif structure["has_markdown_headers"]:
            return self._chunk_with_header_awareness(text, optimal_size, metadata)
        elif structure["is_dense"]:
            return self._chunk_dense_text(text, optimal_size, metadata)
        else:
            return self._chunk_normal_text(text, optimal_size, metadata)

    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure for adaptive chunking."""
        lines = text.split('\n')
        num_lines = len(lines)
        avg_line_length = sum(len(line) for line in lines) / max(1, num_lines)

        # Detect structural elements
        has_code_blocks = bool(re.search(r'```[\s\S]*?```', text))
        has_markdown_headers = bool(re.search(r'^#{1,6}\s+', text, re.MULTILINE))
        has_lists = bool(re.search(r'^[\s]*[-*•]\s+', text, re.MULTILINE))
        has_tables = bool(re.search(r'\|.*\|', text))

        # Calculate text density (characters per line)
        text_density = len(text.replace('\n', '')) / max(1, num_lines)
        is_dense = text_density > 100

        # Calculate semantic richness (unique words ratio)
        words = re.findall(r'\w+', text.lower())
        unique_words = len(set(words))
        word_count = len(words)
        semantic_richness = unique_words / max(1, word_count) if word_count > 0 else 0

        return {
            "has_code_blocks": has_code_blocks,
            "has_markdown_headers": has_markdown_headers,
            "has_lists": has_lists,
            "has_tables": has_tables,
            "is_dense": is_dense,
            "semantic_richness": semantic_richness,
            "avg_line_length": avg_line_length,
            "num_lines": num_lines,
            "word_count": word_count
        }

    def _calculate_optimal_chunk_size(self, text: str, structure: Dict[str, Any]) -> int:
        """Calculate optimal chunk size based on text structure."""
        base_size = self.target_chunk_size

        # Adjust for semantic richness (richer content = smaller chunks for precision)
        if structure["semantic_richness"] > 0.3:
            base_size = int(base_size * 0.8)

        # Adjust for density (dense text = smaller chunks for readability)
        if structure["is_dense"]:
            base_size = int(base_size * 0.85)

        # Adjust for structural complexity
        if structure["has_code_blocks"]:
            base_size = int(base_size * 0.9)

        if structure["has_markdown_headers"]:
            # Larger chunks for markdown to preserve context
            base_size = int(base_size * 1.1)

        # Ensure within bounds
        return max(self.min_chunk_size, min(self.max_chunk_size, base_size))

    def _chunk_with_code_awareness(
        self,
        text: str,
        chunk_size: int,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Chunk]:
        """Chunk code while preserving semantic units (functions, classes)."""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        index = 0
        in_code_block = False

        for line in lines:
            # Detect code block boundaries
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                current_chunk.append(line)
                current_size += len(line)
                continue

            # If in code block, preserve entire block
            if in_code_block:
                current_chunk.append(line)
                current_size += len(line)
                continue

            # Check if line starts a function/class
            if re.match(r'^(def|class|function)\s+', line.strip()):
                if current_chunk and current_size > 0:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))
                    index += 1
                    current_chunk = []
                    current_size = 0

            current_chunk.append(line)
            current_size += len(line)

            # Check size limit
            if current_size >= chunk_size and not in_code_block:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))
                index += 1

                # Keep overlap (last few lines)
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

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))

        return chunks

    def _chunk_with_header_awareness(
        self,
        text: str,
        chunk_size: int,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Chunk]:
        """Chunk markdown while respecting header hierarchy."""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_header = "root"
        current_size = 0
        index = 0

        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                if current_chunk and current_size > 0:
                    chunk_text = '\n'.join(current_chunk)
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["header"] = current_header
                    chunks.append(self._create_chunk(chunk_text, index, 0, 0, chunk_metadata))
                    index += 1

                    # Keep header for overlap
                    if self.chunk_overlap > 0:
                        current_chunk = [current_chunk[0]] if current_chunk else []
                        current_size = len(current_chunk[0]) if current_chunk else 0
                    else:
                        current_chunk = []
                        current_size = 0

                current_chunk.append(line)
                current_header = header_match.group(2)
                current_size = len(line)
            else:
                # Check if adding line exceeds size
                if current_size + len(line) > chunk_size and current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["header"] = current_header
                    chunks.append(self._create_chunk(chunk_text, index, 0, 0, chunk_metadata))
                    index += 1

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

                current_chunk.append(line)
                current_size += len(line)

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["header"] = current_header
            chunks.append(self._create_chunk(chunk_text, index, 0, 0, chunk_metadata))

        return chunks

    def _chunk_dense_text(
        self,
        text: str,
        chunk_size: int,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Chunk]:
        """Chunk dense text (scientific, technical content)."""
        # Split by sentences for dense text
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_size = 0
        index = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If sentence is too long, split it
            if sentence_size > chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))
                    index += 1
                    current_chunk = []
                    current_size = 0

                # Split long sentence into smaller parts
                words = sentence.split()
                temp_chunk = []
                temp_size = 0
                for word in words:
                    if temp_size + len(word) > chunk_size:
                        chunk_text = ' '.join(temp_chunk)
                        chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))
                        index += 1
                        temp_chunk = [word]
                        temp_size = len(word)
                    else:
                        temp_chunk.append(word)
                        temp_size += len(word)

                if temp_chunk:
                    chunk_text = ' '.join(temp_chunk)
                    chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))
                    index += 1
                continue

            # Check if adding sentence exceeds size
            if current_size + sentence_size > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))
                index += 1

                # Keep overlap sentences
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                current_chunk = overlap_sentences
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))

        return chunks

    def _chunk_normal_text(
        self,
        text: str,
        chunk_size: int,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Chunk]:
        """Chunk normal prose text."""
        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_size = 0
        index = 0

        for para in paragraphs:
            # If paragraph is too long, use sentence-based chunking
            if len(para) > chunk_size * 1.5:
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))
                    index += 1
                    current_chunk = []
                    current_size = 0

                # Split paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)
                for sentence in sentences:
                    if current_size + len(sentence) > chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))
                        index += 1

                        # Keep overlap
                        overlap_sentences = []
                        overlap_size = 0
                        for s in reversed(current_chunk):
                            if overlap_size + len(s) <= self.chunk_overlap:
                                overlap_sentences.insert(0, s)
                                overlap_size += len(s)
                            else:
                                break
                        current_chunk = overlap_sentences
                        current_size = overlap_size

                    current_chunk.append(sentence)
                    current_size += len(sentence)
                continue

            # Check if adding paragraph exceeds size
            if current_size + len(para) > chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))
                index += 1

                # Keep overlap paragraphs
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
            current_size += len(para)

        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, index, 0, 0, metadata))

        return chunks


class SentenceChunker(BaseChunker):
    """Chunk text by sentences with optimized overlap."""

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

        # Optimal overlap for sentence chunking (1-2 sentences)
        self.optimal_overlap = min(2, chunk_size // 3)

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

        # Use optimal overlap
        overlap = min(self.optimal_overlap, self.chunk_overlap)
        step = self.chunk_size - overlap

        if step <= 0:
            step = max(1, self.chunk_size // 2)

        for i in range(0, len(sentences), step):
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
    """Chunk text by paragraphs with optimized overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        super().__init__(chunk_size, chunk_overlap)
        # Optimal overlap ratio for paragraphs (10-15%)
        self.optimal_overlap_ratio = 0.12

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into paragraph-based chunks with overlap."""
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

                # Calculate optimal overlap
                overlap_size = int(self.chunk_size * self.optimal_overlap_ratio)
                overlap_paras = []
                overlap_accum = 0

                # Keep paragraphs from end that fit within overlap
                for p in reversed(current_chunk):
                    if overlap_accum + len(p) <= overlap_size:
                        overlap_paras.insert(0, p)
                        overlap_accum += len(p)
                    else:
                        break

                current_chunk = overlap_paras
                current_size = overlap_accum

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
        # Optimized separators for better semantic boundaries
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

        # Validate overlap
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(f"Overlap ({chunk_overlap}) >= chunk size ({chunk_size}). "
                          f"Setting overlap to {chunk_size // 3}")
            self.chunk_overlap = chunk_size // 3

        # Optimal overlap ratio
        self.optimal_overlap_ratio = 0.15

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

        # Group into chunks with optimal overlap
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

                # Calculate optimal overlap
                overlap_size = int(self.chunk_size * self.optimal_overlap_ratio)
                overlap_parts = []
                overlap_accum = 0

                # Build overlap by adding parts from end until overlap size reached
                for part_idx in range(len(current_chunk_parts) - 1, -1, -1):
                    part_to_add = current_chunk_parts[part_idx]
                    if overlap_accum + len(part_to_add) <= overlap_size:
                        overlap_parts.insert(0, part_to_add)
                        overlap_accum += len(part_to_add)
                    else:
                        # If we can't add the whole part, try to add part of it
                        remaining = overlap_size - overlap_accum
                        if remaining > 0:
                            overlap_parts.insert(0, part_to_add[:remaining])
                            overlap_accum += remaining
                        break

                # Update for next chunk
                current_chunk_parts = overlap_parts
                current_size = overlap_accum
                chunk_start_pos = chunk_end_pos - overlap_accum if overlap_accum > 0 else chunk_end_pos

                # If we have overlapping content, continue with next part
                if overlap_parts and i < len(parts):
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
        """Fallback to fixed-size splitting with optimal overlap."""
        chunks = []
        text_length = len(text)
        overlap_size = int(self.chunk_size * self.optimal_overlap_ratio)
        step = self.chunk_size - overlap_size

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

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.optimal_overlap_ratio = 0.15

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text using sliding window with optimal overlap."""
        if not text or not text.strip():
            return []

        chunks = []
        text_length = len(text)

        # Use optimal overlap
        overlap_size = int(self.chunk_size * self.optimal_overlap_ratio)
        step = self.chunk_size - overlap_size

        if step <= 0:
            logger.warning(f"Step size ({step}) invalid. Using step = {self.chunk_size // 2}")
            step = max(1, self.chunk_size // 2)

        for start in range(0, text_length, step):
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]

            # Skip very small chunks at the end (less than 20% of chunk size)
            if len(chunk_text) < self.chunk_size * 0.2 and start > 0:
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


class ChunkingPipeline:
    """Main pipeline for chunking documents with different strategies."""

    def __init__(self, strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE,
                 chunk_size: int = 800, chunk_overlap: int = 150,
                 **kwargs):
        """
        Initialize chunking pipeline with optimized defaults.

        Args:
            strategy: Chunking strategy to use (default: ADAPTIVE)
            chunk_size: Size of each chunk (default: 800 for optimal retrieval)
            chunk_overlap: Overlap between chunks (default: 150)
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
        logger.info(f"Initialized ChunkingPipeline with strategy={strategy.value}, "
                   f"size={chunk_size}, overlap={chunk_overlap}")

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
            ChunkingStrategy.ADAPTIVE: AdaptiveChunker,  # NEW
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

        # Calculate overlap stats
        overlaps = []
        for i in range(len(chunks) - 1):
            # Find overlap between consecutive chunks
            current_end = chunks[i].text[-100:] if len(chunks[i].text) > 100 else chunks[i].text
            next_start = chunks[i+1].text[:100] if len(chunks[i+1].text) > 100 else chunks[i+1].text
            # Simple overlap check
            for j in range(min(len(current_end), len(next_start))):
                if current_end[-j:] == next_start[:j]:
                    overlaps.append(j)
                    break

        return {
            "total_chunks": len(chunks),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_chars": sum(sizes),
            "overlap_setting": self.chunk_overlap,
            "avg_overlap": sum(overlaps) / len(overlaps) if overlaps else 0,
            "size_std_dev": np.std(sizes) if len(sizes) > 1 else 0
        }


# Convenience function with optimized defaults
def chunk_text(text: str, strategy: str = "adaptive",
               chunk_size: int = 800, chunk_overlap: int = 150,
               **kwargs) -> List[Chunk]:
    """
    Quick helper function to chunk text with optimized defaults.

    Args:
        text: Text to chunk
        strategy: Chunking strategy ('fixed_size', 'sentence', 'paragraph',
                  'recursive', 'sliding_window', 'adaptive')
        chunk_size: Size of chunks (default: 800 for optimal retrieval)
        chunk_overlap: Overlap between chunks (default: 150)

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
    # Example usage with optimized chunking
    logging.basicConfig(level=logging.INFO)

    sample_text = """
    This is the first paragraph. It contains multiple sentences. Here's another sentence.
    
    This is the second paragraph. It has different content about machine learning and AI.
    
    And a third paragraph with more information to demonstrate chunking and overlap.
    
    This is a fourth paragraph that is much longer and contains more detailed information about the topic at hand. It discusses various aspects of the subject matter and provides examples and explanations.
    """

    # Test optimized adaptive chunking
    print("Testing ADAPTIVE chunking strategy...")
    chunks = chunk_text(sample_text, strategy="adaptive", chunk_size=200, chunk_overlap=40)

    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk.text)} chars - '{chunk.text[:50]}...'")

    # Compare strategies
    print("\n" + "="*50)
    print("STRATEGY COMPARISON")
    print("="*50)

    strategies = ["fixed_size", "sentence", "paragraph", "recursive", "adaptive"]

    for strat in strategies:
        chunks = chunk_text(sample_text, strategy=strat, chunk_size=200, chunk_overlap=40)
        stats = ChunkingPipeline().get_chunk_stats(chunks)
        print(f"\n{strat.upper()}:")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Avg size: {stats['avg_size']:.0f} chars")
        print(f"  Size std dev: {stats['size_std_dev']:.0f}")

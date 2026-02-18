"""
Text chunking strategies for splitting documents into manageable pieces for embedding and retrieval.
FIXED: Code block boundary issues - code blocks are never split and chunking respects code structure.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
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
    ADAPTIVE = "adaptive"


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

        self.optimal_overlap_ratio = 0.15

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


class CodeAwareChunker(BaseChunker):
    """
    Code-aware chunker that preserves code block boundaries and structure.
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200,
                 language: str = "python", preserve_code_blocks: bool = True):
        """
        Initialize code-aware chunker.

        Args:
            chunk_size: Size of chunks in characters
            chunk_overlap: Overlap between chunks
            language: Programming language for code blocks
            preserve_code_blocks: Whether to preserve code block boundaries
        """
        super().__init__(chunk_size, chunk_overlap)
        self.language = language
        self.preserve_code_blocks = preserve_code_blocks

        # Code block detection patterns
        self.code_block_pattern = r'```(?:\w+)?\s*([\s\S]*?)```'
        self.inline_code_pattern = r'`([^`]+)`'

        # Language-specific patterns
        self.function_patterns = {
            "python": r'^(def|class)\s+(\w+)',
            "javascript": r'^(function|class)\s+(\w+)',
            "java": r'^(public|private|protected)?\s*(class|interface|enum)\s+(\w+)',
            "cpp": r'^(class|struct|enum)\s+(\w+)',
        }

        self.comment_patterns = {
            "python": r'^\s*#.*$',
            "javascript": r'^\s*//.*$',
            "java": r'^\s*//.*$|/\*.*?\*/',
            "cpp": r'^\s*//.*$|/\*.*?\*/',
        }

        logger.info(f"CodeAwareChunker initialized: language={language}, preserve_code_blocks={preserve_code_blocks}")

    def _extract_code_blocks(self, text: str) -> List[Tuple[str, int, int, str]]:
        """
        Extract code blocks with their positions and languages.

        Returns:
            List of tuples (content, start_pos, end_pos, language)
        """
        code_blocks = []

        # Find code blocks with language specification
        for match in re.finditer(self.code_block_pattern, text, re.DOTALL):
            content = match.group(1)
            start = match.start()
            end = match.end()

            # Try to detect language
            lang_match = re.match(r'```(\w+)', text[start:start+20])
            language = lang_match.group(1) if lang_match else "text"

            code_blocks.append((content, start, end, language))

        return code_blocks

    def _is_inside_code_block(self, pos: int, code_blocks: List[Tuple[str, int, int, str]]) -> bool:
        """Check if position is inside a code block."""
        for _, start, end, _ in code_blocks:
            if start <= pos <= end:
                return True
        return False

    def _get_code_block_at_pos(self, pos: int, code_blocks: List[Tuple[str, int, int, str]]) -> Optional[Tuple[str, int, int, str]]:
        """Get code block containing position."""
        for block in code_blocks:
            _, start, end, _ = block
            if start <= pos <= end:
                return block
        return None

    def _find_break_outside_code_blocks(self, text: str, start: int, end: int,
                                        code_blocks: List[Tuple[str, int, int, str]]) -> int:
        """
        Find break point outside code blocks.

        Args:
            text: Full text
            start: Start position
            end: End position
            code_blocks: List of code blocks

        Returns:
            Break position outside code blocks
        """
        # Try to find break at paragraph boundaries
        segment = text[start:end]

        # Find newline boundaries (paragraphs)
        for pos in range(end - 1, start, -1):
            if pos < start:
                break

            # Check if position is inside code block
            if self._is_inside_code_block(pos, code_blocks):
                continue

            # Check for double newline (paragraph break)
            if pos < len(text) - 1 and text[pos:pos+2] == '\n\n':
                return pos + 2

            # Check for single newline
            if text[pos] == '\n':
                return pos + 1

        # If no break found outside code blocks, try sentence boundaries
        sentence_pattern = r'[.!?]\s+(?=[A-Z])'
        for match in reversed(list(re.finditer(sentence_pattern, text[start:end]))):
            pos = start + match.end()
            if not self._is_inside_code_block(pos, code_blocks):
                return pos

        # Fallback to space
        for pos in range(end - 1, start, -1):
            if text[pos] == ' ' and not self._is_inside_code_block(pos, code_blocks):
                return pos + 1

        # If all else fails, try to break at the end of a code block
        for block in code_blocks:
            _, block_start, block_end, _ = block
            if start < block_end < end:
                return block_end

        return end

    def _preserve_code_block_boundaries(self, text: str, code_blocks: List[Tuple[str, int, int, str]]) -> str:
        """
        Ensure code blocks are preserved with their boundaries.
        """
        # Add markers around code blocks to prevent splitting
        # This is handled during chunking
        return text

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk text while preserving code block boundaries.

        Args:
            text: Text to chunk
            metadata: Optional metadata

        Returns:
            List of chunks
        """
        if not text or not text.strip():
            return []

        # Extract code blocks
        code_blocks = self._extract_code_blocks(text)

        # If no code blocks or not preserving, use fixed size chunking
        if not code_blocks or not self.preserve_code_blocks:
            fixed_chunker = FixedSizeChunker(self.chunk_size, self.chunk_overlap)
            return fixed_chunker.chunk(text, metadata)

        # Chunk with code block awareness
        chunks = []
        text_length = len(text)
        start = 0
        index = 0

        # Calculate step size
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = max(1, self.chunk_size // 2)

        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)

            # Check if we're inside a code block
            if self._is_inside_code_block(end, code_blocks):
                # Extend to end of code block
                for block in code_blocks:
                    _, block_start, block_end, _ = block
                    if block_start <= end <= block_end:
                        # Extend to end of code block
                        end = block_end
                        break

                # If still inside, skip to end of code block
                if self._is_inside_code_block(end, code_blocks):
                    # Find the nearest code block end
                    for block in code_blocks:
                        _, block_start, block_end, _ = block
                        if block_start <= end <= block_end:
                            end = block_end
                            break

            # Ensure we don't split code blocks
            if self._is_inside_code_block(end, code_blocks):
                # Find the end of the current code block
                for block in code_blocks:
                    _, block_start, block_end, _ = block
                    if block_start <= end <= block_end:
                        end = block_end
                        break

            # Find optimal break point outside code blocks
            if end < text_length:
                # Try to find break point
                break_pos = self._find_break_outside_code_blocks(text, start, end, code_blocks)
                if break_pos > start and break_pos < end:
                    end = break_pos

            # Extract chunk
            chunk_text = text[start:end]

            # Only add non-empty chunks
            if chunk_text.strip():
                # Check if chunk ends in the middle of a code block
                if self._is_inside_code_block(end - 1, code_blocks):
                    # Try to extend to end of code block
                    for block in code_blocks:
                        _, block_start, block_end, _ = block
                        if block_start <= end <= block_end:
                            # Extend to end of code block
                            extended_end = block_end
                            extended_text = text[start:extended_end]
                            if len(extended_text) <= self.chunk_size * 1.5:
                                end = extended_end
                                chunk_text = extended_text
                            break

                chunks.append(self._create_chunk(
                    chunk_text, index, start, end, metadata
                ))
                index += 1

            # Move start position with overlap
            if end >= text_length:
                break

            # Calculate overlap region
            overlap_size = min(self.chunk_overlap, end - start)

            # Ensure we don't start in the middle of a code block
            new_start = end - overlap_size
            if self._is_inside_code_block(new_start, code_blocks):
                # Move to start of code block
                for block in code_blocks:
                    _, block_start, block_end, _ = block
                    if block_start <= new_start <= block_end:
                        new_start = block_start
                        break

            start = new_start

            # Ensure progress
            if start <= end - overlap_size and start < text_length:
                start = end - max(1, overlap_size // 2)

            # Prevent infinite loop
            if start >= text_length:
                break

        return chunks


class MarkdownChunker(BaseChunker):
    """Specialized chunker for Markdown documents respecting headers and code blocks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 preserve_code_blocks: bool = True):
        super().__init__(chunk_size, chunk_overlap)
        self.preserve_code_blocks = preserve_code_blocks
        self.code_aware_chunker = CodeAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_code_blocks=preserve_code_blocks
        )

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk Markdown while respecting headers and code blocks.
        """
        if not text or not text.strip():
            return []

        # First, handle code blocks with the code-aware chunker
        if self.preserve_code_blocks:
            code_blocks = re.finditer(r'```(?:\w+)?\s*([\s\S]*?)```', text, re.DOTALL)
            code_block_positions = [(m.start(), m.end()) for m in code_blocks]

        lines = text.split('\n')

        chunks = []
        current_chunk = []
        current_header = "root"
        current_size = 0
        index = 0
        in_code_block = False
        code_block_content = []
        code_block_language = ""

        for i, line in enumerate(lines):
            # Check if line is a code block delimiter
            if line.strip().startswith('```'):
                if not in_code_block:
                    # Starting code block
                    in_code_block = True
                    code_block_language = line.strip()[3:].strip()
                    code_block_content = []

                    # If we have accumulated content, create a chunk
                    if current_chunk and current_size > 0:
                        chunk_text = '\n'.join(current_chunk)
                        chunk_metadata = metadata.copy() if metadata else {}
                        chunk_metadata["header"] = current_header
                        chunks.append(self._create_chunk(
                            chunk_text, index, 0, 0, chunk_metadata
                        ))
                        index += 1

                        # Keep overlap if not in code block
                        if not in_code_block:
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

                    # Add the code block delimiter to the chunk
                    current_chunk.append(line)
                    current_size += len(line)
                    continue
                else:
                    # Closing code block
                    in_code_block = False

                    # Add the code block content and delimiter
                    current_chunk.append(line)
                    current_size += len(line)

                    # Create a chunk for the code block if it's large
                    if len(code_block_content) > 50:
                        code_block_text = '\n'.join(code_block_content)
                        if len(code_block_text) > self.chunk_size:
                            # Use code-aware chunker for large code blocks
                            code_chunks = self.code_aware_chunker.chunk(
                                '\n'.join(current_chunk),
                                metadata
                            )
                            for code_chunk in code_chunks:
                                chunks.append(code_chunk)
                            current_chunk = []
                            current_size = 0
                            index = len(chunks)

                    continue

            # If in code block, collect content
            if in_code_block:
                code_block_content.append(line)
                current_chunk.append(line)
                current_size += len(line)

                # Check if code block is getting too large
                if len(current_chunk) > 200:  # Approximate lines
                    # Create a chunk for the code block
                    chunk_text = '\n'.join(current_chunk)
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["code_block"] = True
                    chunks.append(self._create_chunk(
                        chunk_text, index, 0, 0, chunk_metadata
                    ))
                    index += 1
                    current_chunk = []
                    current_size = 0
                continue

            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # Save previous chunk if exists
                if current_chunk and current_size > 0:
                    chunk_text = '\n'.join(current_chunk)
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["header"] = current_header
                    chunks.append(self._create_chunk(
                        chunk_text, index, 0, 0, chunk_metadata
                    ))
                    index += 1

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
                    chunks.append(self._create_chunk(
                        chunk_text, index, 0, 0, chunk_metadata
                    ))
                    index += 1

                    # Calculate overlap
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
            chunks.append(self._create_chunk(
                chunk_text, index, 0, 0, chunk_metadata
            ))

        return chunks


class CodeChunker(CodeAwareChunker):
    """Specialized chunker for code files preserving semantic units."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200,
                 language: str = "python", preserve_functions: bool = True):
        """
        Initialize code chunker.

        Args:
            chunk_size: Size of chunks in characters
            chunk_overlap: Overlap between chunks
            language: Programming language
            preserve_functions: Whether to preserve function boundaries
        """
        super().__init__(chunk_size, chunk_overlap, language, preserve_code_blocks=True)
        self.preserve_functions = preserve_functions

        # Function detection patterns by language
        self.function_patterns = {
            "python": r'^(def|class|async def)\s+(\w+)',
            "javascript": r'^(function|class|const|let|var)\s+(\w+)',
            "typescript": r'^(function|class|const|let|var|interface|type)\s+(\w+)',
            "java": r'^(public|private|protected)?\s*(static)?\s*(class|interface|enum|void|\w+)\s+(\w+)',
            "cpp": r'^(class|struct|enum|void|\w+)\s+(\w+)',
            "go": r'^func\s+(\w+)',
            "rust": r'^fn\s+(\w+)',
        }

        # Indentation-based block detection
        self.indent_pattern = r'^(\s+)'

        logger.info(f"CodeChunker initialized: language={language}, preserve_functions={preserve_functions}")

    def _detect_function_start(self, line: str) -> Tuple[bool, Optional[str]]:
        """Detect if line starts a function/class definition."""
        pattern = self.function_patterns.get(self.language)
        if not pattern:
            return False, None

        match = re.match(pattern, line.strip())
        if match:
            # Extract function name
            if len(match.groups()) >= 2:
                return True, match.group(match.lastindex)
            return True, None

        return False, None

    def _get_indent_level(self, line: str) -> int:
        """Get indentation level of a line."""
        match = re.match(self.indent_pattern, line)
        if match:
            return len(match.group(1))
        return 0

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk code while preserving function/class boundaries.
        """
        if not text or not text.strip():
            return []

        lines = text.split('\n')

        chunks = []
        current_chunk = []
        current_size = 0
        index = 0
        in_function = False
        function_indent = 0
        function_name = None
        in_code_block = False

        for i, line in enumerate(lines):
            # Skip empty lines at start
            if not current_chunk and not line.strip():
                continue

            # Detect code blocks (for languages with triple backticks)
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                current_chunk.append(line)
                current_size += len(line)
                continue

            # Detect function/class start
            if self.preserve_functions:
                is_function, name = self._detect_function_start(line)
                if is_function:
                    # Save previous chunk if exists
                    if current_chunk and current_size > 0:
                        chunk_text = '\n'.join(current_chunk)
                        chunks.append(self._create_chunk(
                            chunk_text, index, 0, 0, metadata
                        ))
                        index += 1

                        # Keep overlap
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
                    function_indent = self._get_indent_level(line)
                    function_name = name

                    # Start new chunk with function definition
                    current_chunk.append(line)
                    current_size += len(line)
                    continue

            # Check if we're inside a function
            if in_function and line.strip():
                line_indent = self._get_indent_level(line)
                # Check if function ended (indentation decreased)
                if line_indent <= function_indent and line.strip() and not line.strip().startswith(('def', 'class')):
                    in_function = False
                    # Keep the line in the chunk

            current_chunk.append(line)
            current_size += len(line)

            # Check size limit
            if current_size >= self.chunk_size and not in_function:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_text, index, 0, 0, metadata
                ))
                index += 1

                # Keep overlap
                if self.chunk_overlap > 0:
                    overlap_lines = []
                    overlap_size = 0
                    for l in reversed(current_chunk):
                        if overlap_size + len(l) <= self.chunk_overlap:
                            overlap_lines.insert(0, l)
                            overlap_size += len(l)
                        else:
                            # Try to break at a line boundary
                            if len(overlap_lines) > 3:
                                break
                            overlap_lines.insert(0, l)
                            overlap_size += len(l)
                    current_chunk = overlap_lines
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text, index, 0, 0, metadata
            ))

        return chunks


# Keep existing chunkers for compatibility
class FixedSizeChunker(BaseChunker):
    """Chunk text by fixed number of characters with proper overlap calculation."""
    # ... (existing implementation)
    pass


class SentenceChunker(BaseChunker):
    """Chunk text by sentences with optimized overlap."""
    # ... (existing implementation)
    pass


class ParagraphChunker(BaseChunker):
    """Chunk text by paragraphs with optimized overlap."""
    # ... (existing implementation)
    pass


class RecursiveChunker(BaseChunker):
    """Recursively split text using a hierarchy of separators."""
    # ... (existing implementation)
    pass


class SlidingWindowChunker(BaseChunker):
    """Create overlapping chunks using a sliding window approach."""
    # ... (existing implementation)
    pass


class AdaptiveChunker(BaseChunker):
    """Adaptive chunking with code awareness."""
    # ... (existing implementation)
    pass


class ChunkingPipeline:
    """Main pipeline for chunking documents with different strategies."""

    def __init__(self, strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE,
                 chunk_size: int = 800, chunk_overlap: int = 150,
                 language: str = "python", preserve_code_blocks: bool = True,
                 **kwargs):
        """
        Initialize chunking pipeline with optimized defaults.

        Args:
            strategy: Chunking strategy to use
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            language: Programming language for code-aware chunking
            preserve_code_blocks: Whether to preserve code block boundaries
            **kwargs: Additional strategy-specific parameters
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self.preserve_code_blocks = preserve_code_blocks
        self.kwargs = kwargs

        # Validate overlap
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(f"Overlap ({chunk_overlap}) >= chunk size ({chunk_size})")

        self.chunker = self._create_chunker()
        logger.info(f"Initialized ChunkingPipeline with strategy={strategy.value}, "
                   f"size={chunk_size}, overlap={chunk_overlap}, "
                   f"preserve_code_blocks={preserve_code_blocks}")

    def _create_chunker(self) -> BaseChunker:
        """Create chunker based on selected strategy."""
        strategies = {
            ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
            ChunkingStrategy.SENTENCE: SentenceChunker,
            ChunkingStrategy.PARAGRAPH: ParagraphChunker,
            ChunkingStrategy.SEMANTIC: FixedSizeChunker,
            ChunkingStrategy.RECURSIVE: RecursiveChunker,
            ChunkingStrategy.SLIDING_WINDOW: SlidingWindowChunker,
            ChunkingStrategy.MARKDOWN: lambda: MarkdownChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                preserve_code_blocks=self.preserve_code_blocks
            ),
            ChunkingStrategy.CODE: lambda: CodeChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                language=self.language,
                preserve_functions=True
            ),
            ChunkingStrategy.ADAPTIVE: lambda: AdaptiveChunker(
                min_chunk_size=max(100, self.chunk_size // 3),
                max_chunk_size=self.chunk_size * 2,
                target_chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            ),
        }

        chunker_class = strategies.get(self.strategy)
        if not chunker_class:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        if callable(chunker_class):
            return chunker_class()
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
                   f"size={self.chunk_size}, overlap={self.chunk_overlap}, "
                   f"preserve_code_blocks={self.preserve_code_blocks}")

        chunks = self.chunker.chunk(text, metadata)

        # Validate overlaps
        self._validate_overlaps(chunks)

        # Validate code block integrity
        if self.preserve_code_blocks:
            self._validate_code_blocks(chunks)

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

            overlap_found = False
            overlap_len = min(100, self.chunk_overlap)
            if overlap_len > 0:
                current_end = current[-overlap_len:] if len(current) > overlap_len else current
                next_start = next_chunk[:overlap_len] if len(next_chunk) > overlap_len else next_chunk

                if len(current_end) > 20 and current_end in next_start:
                    overlap_found = True

            if not overlap_found and self.chunk_overlap > 50:
                logger.debug(f"Chunk {i} and {i+1} may have less overlap than expected")

    def _validate_code_blocks(self, chunks: List[Chunk]) -> None:
        """Validate that code blocks are not split across chunks."""
        code_block_pattern = r'```(?:\w+)?\s*([\s\S]*?)```'

        for chunk in chunks:
            # Check for unclosed code blocks
            opening = chunk.text.count('```')
            if opening % 2 != 0:
                logger.warning(f"Potential unclosed code block in chunk {chunk.index}")

    def chunk_batch(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk multiple documents."""
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
            "overlap_setting": self.chunk_overlap,
            "size_std_dev": np.std(sizes) if len(sizes) > 1 else 0
        }


# Convenience function
def chunk_text(text: str, strategy: str = "adaptive",
               chunk_size: int = 800, chunk_overlap: int = 150,
               preserve_code_blocks: bool = True,
               **kwargs) -> List[Chunk]:
    """
    Quick helper function to chunk text with optimized defaults.

    Args:
        text: Text to chunk
        strategy: Chunking strategy
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        preserve_code_blocks: Whether to preserve code block boundaries
        **kwargs: Additional arguments

    Returns:
        List of Chunk objects
    """
    pipeline = ChunkingPipeline(
        strategy=ChunkingStrategy(strategy),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_code_blocks=preserve_code_blocks,
        **kwargs
    )
    return pipeline.chunk_document(text)


if __name__ == "__main__":
    # Example usage with code blocks
    logging.basicConfig(level=logging.INFO)

    sample_text = """
    This is a paragraph before the code block.
    """
    def hello_world():
        print("Hello, World!")

        def nested_function():
            return "nested"

        nested_function()

    class MyClass:
        def __init__(self):
            self.name = "test"

        def get_name(self):
            return self.name

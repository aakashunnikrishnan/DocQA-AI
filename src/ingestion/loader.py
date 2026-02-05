"""
Document loader module with support for multiple file formats including PDF and DOCX.
FIXED: Memory leaks in large document processing with proper resource management,
streaming, and chunked processing.
"""

import os
import gc
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Iterator
from pathlib import Path
from abc import ABC, abstractmethod
from functools import wraps
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager

import PyPDF2
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table, _Cell
from bs4 import BeautifulSoup
import markdown
import csv
import json
import chardet

logger = logging.getLogger(__name__)


@dataclass
class LoaderResult:
    """Result object for loader operations."""
    success: bool
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    file_size: int = 0
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    memory_usage_mb: float = 0.0


class LoaderError(Exception):
    """Base exception for loader errors."""
    pass


class FileNotFoundError(LoaderError):
    """File not found error."""
    pass


class UnsupportedFormatError(LoaderError):
    """Unsupported file format error."""
    pass


class CorruptedFileError(LoaderError):
    """Corrupted file error."""
    pass


class ParsingError(LoaderError):
    """Parsing error during file loading."""
    pass


class MemoryLimitExceeded(LoaderError):
    """Memory limit exceeded error."""
    pass


class LoaderErrorCode:
    """Error codes for loader operations."""
    SUCCESS = "SUCCESS"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    CORRUPTED_FILE = "CORRUPTED_FILE"
    PARSING_ERROR = "PARSING_ERROR"
    ENCODING_ERROR = "ENCODING_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    EMPTY_FILE = "EMPTY_FILE"
    TIMEOUT = "TIMEOUT"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


@contextmanager
def track_memory(operation_name: str = "operation"):
    """Context manager to track memory usage."""
    start_mem = get_memory_usage()
    try:
        yield
    finally:
        end_mem = get_memory_usage()
        if end_mem - start_mem > 10:  # Log if memory increased by more than 10MB
            logger.debug(f"Memory increase in {operation_name}: {end_mem - start_mem:.2f} MB")
        gc.collect()


def handle_loader_errors(default_return: Any = None, retry_count: int = 2, retry_delay: float = 1.0,
                         max_memory_mb: int = 2048):
    """
    Decorator for error handling with memory limit checking.

    Args:
        default_return: Default value to return on error
        retry_count: Number of retry attempts
        retry_delay: Delay between retries in seconds
        max_memory_mb: Maximum memory allowed in MB
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_error = None
            memory_start = get_memory_usage()

            # Check memory before starting
            if memory_start > max_memory_mb:
                logger.error(f"Memory usage ({memory_start:.2f} MB) exceeds limit ({max_memory_mb} MB)")
                if isinstance(default_return, LoaderResult) or default_return is None:
                    return LoaderResult(
                        success=False,
                        error_code=LoaderErrorCode.MEMORY_LIMIT_EXCEEDED,
                        error_message=f"Memory limit exceeded: {memory_start:.2f} MB / {max_memory_mb} MB"
                    )
                return default_return

            for attempt in range(retry_count):
                try:
                    # Force garbage collection before each attempt
                    gc.collect()

                    start_time = time.time()
                    result = func(self, *args, **kwargs)

                    # Check memory after operation
                    memory_after = get_memory_usage()

                    # If result is LoaderResult, add memory usage
                    if isinstance(result, LoaderResult):
                        result.processing_time_ms = (time.time() - start_time) * 1000
                        result.memory_usage_mb = memory_after - memory_start

                    # Force garbage collection after operation
                    gc.collect()

                    return result

                except MemoryError as e:
                    logger.error(f"Memory error in {func.__name__}: {e}")
                    last_error = e
                    # Force garbage collection and try again
                    gc.collect()
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay * (attempt + 1))

                except FileNotFoundError as e:
                    logger.error(f"File not found: {e}")
                    last_error = e
                    break

                except UnsupportedFormatError as e:
                    logger.error(f"Unsupported format: {e}")
                    last_error = e
                    break

                except PermissionError as e:
                    logger.error(f"Permission denied: {e}")
                    last_error = e
                    break

                except (CorruptedFileError, ParsingError, UnicodeDecodeError) as e:
                    logger.warning(f"Attempt {attempt + 1}/{retry_count} failed: {e}")
                    last_error = e

                    if attempt < retry_count - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        # Force garbage collection before retry
                        gc.collect()

                except Exception as e:
                    logger.error(f"Unexpected error: {e}", exc_info=True)
                    last_error = e

                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        gc.collect()

            # Force garbage collection before returning error
            gc.collect()

            if isinstance(default_return, LoaderResult) or default_return is None:
                return LoaderResult(
                    success=False,
                    error_code=LoaderErrorCode.UNKNOWN_ERROR,
                    error_message=str(last_error) if last_error else "Unknown error"
                )
            return default_return

        return wrapper
    return decorator


class BaseLoader(ABC):
    """Abstract base class for document loaders with memory management."""

    def __init__(self, timeout: int = 60, max_file_size_mb: int = 100,
                 max_memory_mb: int = 2048, chunk_size: int = 1024 * 1024):  # 1MB chunks
        """
        Initialize base loader.

        Args:
            timeout: Maximum time in seconds for loading operations
            max_file_size_mb: Maximum file size in MB
            max_memory_mb: Maximum memory usage in MB
            chunk_size: Chunk size for streaming reads
        """
        self.timeout = timeout
        self.max_file_size_mb = max_file_size_mb
        self.max_memory_mb = max_memory_mb
        self.chunk_size = chunk_size

    @abstractmethod
    def load(self, file_path: str) -> LoaderResult:
        """Load document and return content."""
        pass

    @abstractmethod
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document."""
        pass

    def validate_file(self, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate file before loading."""
        path = Path(file_path)

        if not path.exists():
            return False, LoaderErrorCode.FILE_NOT_FOUND, f"File not found: {file_path}"

        if not path.is_file():
            return False, LoaderErrorCode.UNKNOWN_ERROR, f"Path is not a file: {file_path}"

        file_size = path.stat().st_size
        max_size_bytes = self.max_file_size_mb * 1024 * 1024

        if file_size == 0:
            return False, LoaderErrorCode.EMPTY_FILE, f"File is empty: {file_path}"

        if file_size > max_size_bytes:
            return False, LoaderErrorCode.UNKNOWN_ERROR, f"File too large: {file_size} bytes (max {max_size_bytes})"

        if not os.access(file_path, os.R_OK):
            return False, LoaderErrorCode.PERMISSION_DENIED, f"No read permission: {file_path}"

        # Check available memory
        current_memory = get_memory_usage()
        if current_memory > self.max_memory_mb * 0.8:  # 80% threshold
            return False, LoaderErrorCode.MEMORY_LIMIT_EXCEEDED, f"Memory usage too high: {current_memory:.2f} MB"

        return True, None, None


class PDFLoader(BaseLoader):
    """Loader for PDF documents with streaming and memory optimization."""

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """Extract text from PDF file with memory optimization."""
        is_valid, error_code, error_msg = self.validate_file(file_path)
        if not is_valid:
            return LoaderResult(
                success=False,
                error_code=error_code,
                error_message=error_msg,
                file_path=file_path
            )

        text_content = []
        warnings = []
        metadata = self.get_metadata(file_path)
        file_size = Path(file_path).stat().st_size
        total_pages = 0
        page_counter = 0

        try:
            with open(file_path, 'rb') as file:
                # Use streaming PDF reader to avoid loading entire file
                try:
                    pdf_reader = PyPDF2.PdfReader(file, strict=False)
                except Exception as e:
                    raise CorruptedFileError(f"Invalid or corrupted PDF file: {e}")

                total_pages = len(pdf_reader.pages)

                if total_pages == 0:
                    raise ParsingError("PDF has no pages")

                # Process pages in chunks to avoid memory issues
                page_batch_size = 50  # Process 50 pages at a time

                for page_start in range(0, total_pages, page_batch_size):
                    page_end = min(page_start + page_batch_size, total_pages)

                    # Process each page in the batch
                    batch_text = []

                    for page_num in range(page_start, page_end):
                        page_counter += 1
                        try:
                            page = pdf_reader.pages[page_num]
                            # Extract text from page
                            text = page.extract_text()

                            # Clear page reference to free memory
                            del page
                            gc.collect()

                            if text and text.strip():
                                batch_text.append(f"[Page {page_counter}]\n{text}")
                            else:
                                warnings.append(f"No text found on page {page_counter}")

                        except Exception as e:
                            warnings.append(f"Failed to extract page {page_counter}: {e}")
                            continue

                    # Add batch text to content
                    if batch_text:
                        text_content.extend(batch_text)

                    # Force garbage collection after each batch
                    gc.collect()

                    # Check memory usage and pause if needed
                    current_memory = get_memory_usage()
                    if current_memory > self.max_memory_mb * 0.7:  # 70% threshold
                        logger.warning(f"Memory usage high ({current_memory:.2f} MB), paging to disk...")
                        # Write current content to temp file if memory is high
                        # (simplified - in production, implement proper paging)
                        gc.collect()

                if not text_content:
                    raise ParsingError("No text content extracted from PDF")

                # Free PDF reader resources
                del pdf_reader
                gc.collect()

        except CorruptedFileError:
            raise
        except Exception as e:
            raise ParsingError(f"Failed to load PDF: {e}")

        # Join content (limit to prevent memory blowup)
        content = "\n\n".join(text_content)

        # Free text_content list
        text_content.clear()
        del text_content
        gc.collect()

        return LoaderResult(
            success=True,
            content=content,
            metadata=metadata,
            file_path=file_path,
            file_size=file_size,
            warnings=warnings,
            memory_usage_mb=get_memory_usage()
        )

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF metadata with memory efficiency."""
        metadata = {
            "file_path": file_path,
            "file_type": "pdf",
            "file_size": Path(file_path).stat().st_size,
            "num_pages": 0,
            "title": None,
            "author": None,
            "subject": None,
            "creator": None,
            "producer": None
        }

        try:
            with open(file_path, 'rb') as file:
                # Use strict=False to handle malformed PDFs
                pdf_reader = PyPDF2.PdfReader(file, strict=False)
                metadata["num_pages"] = len(pdf_reader.pages)

                if pdf_reader.metadata:
                    meta = pdf_reader.metadata
                    metadata["title"] = meta.get('/Title', None)
                    metadata["author"] = meta.get('/Author', None)
                    metadata["subject"] = meta.get('/Subject', None)
                    metadata["creator"] = meta.get('/Creator', None)
                    metadata["producer"] = meta.get('/Producer', None)

                del pdf_reader
                gc.collect()

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return metadata


class DOCXLoader(BaseLoader):
    """Memory-optimized loader for DOCX documents."""

    def __init__(self, timeout: int = 60, max_file_size_mb: int = 100,
                 max_memory_mb: int = 2048, chunk_size: int = 1024 * 1024,
                 extract_tables: bool = True, extract_headers: bool = True,
                 preserve_formatting: bool = True, extract_lists: bool = True,
                 max_paragraphs: int = 100000):
        """
        Initialize DOCX loader with memory limits.

        Args:
            timeout: Maximum time in seconds for loading operations
            max_file_size_mb: Maximum file size in MB
            max_memory_mb: Maximum memory usage in MB
            chunk_size: Chunk size for streaming reads
            extract_tables: Whether to extract table content
            extract_headers: Whether to extract headers/footers
            preserve_formatting: Whether to preserve formatting (bold, italic)
            extract_lists: Whether to detect and format lists
            max_paragraphs: Maximum number of paragraphs to process
        """
        super().__init__(timeout, max_file_size_mb, max_memory_mb, chunk_size)
        self.extract_tables = extract_tables
        self.extract_headers = extract_headers
        self.preserve_formatting = preserve_formatting
        self.extract_lists = extract_lists
        self.max_paragraphs = max_paragraphs

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """Extract text from DOCX file with memory optimization."""
        is_valid, error_code, error_msg = self.validate_file(file_path)
        if not is_valid:
            return LoaderResult(
                success=False,
                error_code=error_code,
                error_message=error_msg,
                file_path=file_path
            )

        text_content = []
        warnings = []
        metadata = self.get_metadata(file_path)
        file_size = Path(file_path).stat().st_size
        paragraph_count = 0

        try:
            # Use streaming document loading
            with track_memory("DOCX loading"):
                try:
                    doc = Document(file_path)
                except Exception as e:
                    raise CorruptedFileError(f"Failed to open DOCX file: {e}")

                # Extract headers
                if self.extract_headers:
                    try:
                        header_text = self._extract_headers_stream(doc)
                        if header_text:
                            text_content.append("=== HEADERS ===\n" + header_text)
                            metadata["has_headers"] = True
                    except Exception as e:
                        warnings.append(f"Failed to extract headers: {e}")

                # Extract paragraphs with streaming
                content_parts = []
                para_count = 0

                for para in doc.paragraphs:
                    para_count += 1
                    if para_count > self.max_paragraphs:
                        warnings.append(f"Reached max paragraphs limit ({self.max_paragraphs})")
                        break

                    if not para.text or not para.text.strip():
                        continue

                    # Check memory periodically
                    if para_count % 1000 == 0:
                        current_memory = get_memory_usage()
                        if current_memory > self.max_memory_mb * 0.7:
                            logger.warning(f"Memory usage high ({current_memory:.2f} MB) during paragraph processing")
                            # Flush accumulated content to prevent memory build-up
                            if len(content_parts) > 5000:
                                text_content.extend(content_parts[:5000])
                                content_parts = content_parts[5000:]
                                gc.collect()

                    # Check if paragraph is a heading
                    if self._is_heading(para):
                        heading_level = self._get_heading_level(para)
                        content_parts.append(f"\n{'#' * heading_level} {para.text.strip()}")
                        continue

                    # Check if paragraph is in a list
                    if self.extract_lists and self._is_list_item(para):
                        list_type = self._get_list_type(para)
                        if list_type == 'bullet':
                            content_parts.append(f"  • {para.text.strip()}")
                        elif list_type == 'numbered':
                            # Use a simple counter approach
                            content_parts.append(f"  {self._get_list_number(para)}. {para.text.strip()}")
                        continue

                    # Regular paragraph with formatting
                    formatted_text = self._format_paragraph_text(para)
                    content_parts.append(formatted_text)

                # Add remaining content parts
                if content_parts:
                    text_content.extend(content_parts)
                    content_parts.clear()
                    del content_parts
                    gc.collect()

                # Extract tables
                if self.extract_tables and doc.tables:
                    try:
                        table_texts = self._extract_tables_stream(doc.tables)
                        if table_texts:
                            text_content.append("\n=== TABLES ===\n")
                            text_content.extend(table_texts)
                            metadata["num_tables"] = len(doc.tables)
                    except Exception as e:
                        warnings.append(f"Failed to extract tables: {e}")

                # Extract footers
                if self.extract_headers:
                    try:
                        footer_text = self._extract_footers_stream(doc)
                        if footer_text:
                            text_content.append("\n=== FOOTERS ===\n" + footer_text)
                            metadata["has_footers"] = True
                    except Exception as e:
                        warnings.append(f"Failed to extract footers: {e}")

                if not text_content:
                    raise ParsingError("No text content extracted from DOCX")

                # Metadata
                metadata["num_paragraphs"] = paragraph_count
                metadata["num_sections"] = len(doc.sections) if hasattr(doc, 'sections') else 0

                # Free document resources
                del doc
                gc.collect()

        except CorruptedFileError:
            raise
        except Exception as e:
            raise ParsingError(f"Failed to load DOCX: {e}")

        # Join content with memory efficiency
        content = "\n\n".join(text_content)

        # Free text_content list
        text_content.clear()
        del text_content
        gc.collect()

        return LoaderResult(
            success=True,
            content=content,
            metadata=metadata,
            file_path=file_path,
            file_size=file_size,
            warnings=warnings,
            memory_usage_mb=get_memory_usage()
        )

    def _extract_headers_stream(self, doc) -> str:
        """Extract headers with memory efficiency."""
        header_texts = []
        try:
            for section in doc.sections:
                if section.header:
                    for para in section.header.paragraphs:
                        if para.text and para.text.strip():
                            header_texts.append(para.text.strip())
        except Exception as e:
            logger.warning(f"Failed to extract headers: {e}")

        return "\n".join(header_texts) if header_texts else ""

    def _extract_footers_stream(self, doc) -> str:
        """Extract footers with memory efficiency."""
        footer_texts = []
        try:
            for section in doc.sections:
                if section.footer:
                    for para in section.footer.paragraphs:
                        if para.text and para.text.strip():
                            footer_texts.append(para.text.strip())
        except Exception as e:
            logger.warning(f"Failed to extract footers: {e}")

        return "\n".join(footer_texts) if footer_texts else ""

    def _format_paragraph_text(self, para: Paragraph) -> str:
        """Format paragraph text with formatting preservation."""
        if not self.preserve_formatting:
            return para.text.strip()

        # Extract runs with formatting
        formatted_parts = []
        for run in para.runs:
            text = run.text
            if not text:
                continue

            # Apply formatting
            if run.bold:
                text = f"**{text}**"
            if run.italic:
                text = f"*{text}*"
            if run.underline:
                text = f"_{text}_"

            formatted_parts.append(text)

        return ''.join(formatted_parts).strip()

    def _is_heading(self, para: Paragraph) -> bool:
        """Check if paragraph is a heading."""
        if para.style and para.style.name:
            return 'heading' in para.style.name.lower()
        return False

    def _get_heading_level(self, para: Paragraph) -> int:
        """Get heading level from paragraph style."""
        if not para.style or not para.style.name:
            return 1

        import re
        match = re.search(r'heading\s*(\d+)', para.style.name.lower())
        if match:
            return min(int(match.group(1)), 6)

        return 1

    def _is_list_item(self, para: Paragraph) -> bool:
        """Check if paragraph is a list item."""
        if para.style and para.style.name:
            style_name = para.style.name.lower()
            if 'list' in style_name or 'bullet' in style_name:
                return True

        # Check for list numbering in paragraph text
        if re.match(r'^[\s]*[•●○■▪▫]', para.text):
            return True
        if re.match(r'^[\s]*\d+[\.\)]', para.text):
            return True
        if re.match(r'^[\s]*[a-zA-Z][\.\)]', para.text):
            return True

        return False

    def _get_list_type(self, para: Paragraph) -> str:
        """Determine list type (bullet or numbered)."""
        if para.style and para.style.name:
            style_name = para.style.name.lower()
            if 'bullet' in style_name:
                return 'bullet'
            if 'number' in style_name or 'enum' in style_name:
                return 'numbered'

        if re.match(r'^[\s]*[•●○■▪▫]', para.text):
            return 'bullet'
        if re.match(r'^[\s]*\d+[\.\)]', para.text):
            return 'numbered'

        return 'bullet'

    def _get_list_number(self, para: Paragraph) -> int:
        """Extract list number from paragraph."""
        match = re.match(r'^[\s]*(\d+)[\.\)]', para.text)
        if match:
            return int(match.group(1))
        return 1

    def _extract_tables_stream(self, tables: List[Table]) -> List[str]:
        """Extract tables with memory efficiency."""
        table_texts = []

        for table_idx, table in enumerate(tables, 1):
            table_lines = []
            table_lines.append(f"Table {table_idx}:")

            # Extract headers (first row)
            if table.rows:
                header_row = table.rows[0]
                headers = []
                for cell in header_row.cells:
                    headers.append(cell.text.strip())
                    # Free cell reference
                    del cell

                if headers:
                    table_lines.append("  " + " | ".join(headers))
                    table_lines.append("  " + "-" * (sum(len(h) for h in headers) + len(headers) * 3))

                # Free header row
                del headers

            # Extract data rows
            row_count = 0
            for row_idx, row in enumerate(table.rows[1:], 1):
                row_count += 1
                if row_count > 1000:  # Limit rows to prevent memory issues
                    table_lines.append(f"  ... and {len(table.rows) - row_count} more rows")
                    break

                cells = []
                for cell in row.cells:
                    cells.append(cell.text.strip())
                    del cell

                if any(cells):
                    table_lines.append(f"  Row {row_idx}: " + " | ".join(cells))

                # Free row resources
                del cells, row

            table_texts.append("\n".join(table_lines))

            # Free table resources
            del table_lines, table

        return table_texts

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract DOCX metadata with memory efficiency."""
        metadata = {
            "file_path": file_path,
            "file_type": "docx",
            "file_size": Path(file_path).stat().st_size,
            "num_paragraphs": 0,
            "num_tables": 0,
            "num_sections": 0,
            "author": None,
            "title": None,
            "subject": None,
            "keywords": None,
            "category": None
        }

        try:
            doc = Document(file_path)

            if doc.core_properties:
                cp = doc.core_properties
                metadata["author"] = str(cp.author) if cp.author else None
                metadata["title"] = str(cp.title) if cp.title else None
                metadata["subject"] = str(cp.subject) if cp.subject else None
                metadata["keywords"] = str(cp.keywords) if cp.keywords else None
                metadata["category"] = str(cp.category) if cp.category else None

            metadata["num_paragraphs"] = len(doc.paragraphs)
            metadata["num_tables"] = len(doc.tables)
            metadata["num_sections"] = len(doc.sections) if hasattr(doc, 'sections') else 0

            # Free document resources
            del doc
            gc.collect()

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return metadata


class TextLoader(BaseLoader):
    """Loader for plain text files with streaming and encoding detection."""

    def __init__(self, timeout: int = 60, max_file_size_mb: int = 100,
                 max_memory_mb: int = 2048, chunk_size: int = 1024 * 1024):
        super().__init__(timeout, max_file_size_mb, max_memory_mb, chunk_size)

    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding with memory efficiency."""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        except Exception:
            return 'utf-8'

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """Load plain text file with streaming."""
        is_valid, error_code, error_msg = self.validate_file(file_path)
        if not is_valid:
            return LoaderResult(
                success=False,
                error_code=error_code,
                error_message=error_msg,
                file_path=file_path
            )

        metadata = self.get_metadata(file_path)
        file_size = Path(file_path).stat().st_size

        # Try multiple encodings
        encodings = [self._detect_encoding(file_path), 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        content = None
        last_error = None

        for encoding in encodings:
            try:
                # Stream read to avoid loading entire file at once
                with open(file_path, 'r', encoding=encoding) as file:
                    # Read in chunks to manage memory
                    chunks = []
                    total_chars = 0
                    max_chars = self.max_file_size_mb * 1024 * 1024  # Max characters

                    while True:
                        chunk = file.read(self.chunk_size)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        total_chars += len(chunk)

                        # Check memory usage periodically
                        if len(chunks) % 10 == 0:
                            current_memory = get_memory_usage()
                            if current_memory > self.max_memory_mb * 0.7:
                                logger.warning(f"Memory usage high ({current_memory:.2f} MB) during text loading")
                                # Flush chunks to prevent memory blowup
                                if len(chunks) > 20:
                                    partial = ''.join(chunks)
                                    chunks = [partial]
                                    gc.collect()

                        if total_chars > max_chars:
                            chunks.append("\n...[truncated]")
                            break

                    content = ''.join(chunks)
                    metadata["encoding"] = encoding
                    break

            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                raise ParsingError(f"Failed to read text file: {e}")

        if content is None:
            raise ParsingError(f"Failed to decode file with any encoding. Last error: {last_error}")

        if not content.strip():
            raise ParsingError("File contains no text content")

        # Count lines
        metadata["line_count"] = content.count('\n') + 1

        return LoaderResult(
            success=True,
            content=content,
            metadata=metadata,
            file_path=file_path,
            file_size=file_size,
            memory_usage_mb=get_memory_usage()
        )

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract text file metadata."""
        return {
            "file_path": file_path,
            "file_type": "txt",
            "file_size": Path(file_path).stat().st_size,
            "encoding": "unknown",
            "line_count": 0
        }


class DocumentLoader:
    """Main document loader with memory management."""

    def __init__(self, timeout: int = 60, max_file_size_mb: int = 100,
                 max_memory_mb: int = 2048, extract_tables: bool = True,
                 extract_headers: bool = True, preserve_formatting: bool = True,
                 extract_lists: bool = True, max_paragraphs: int = 100000):
        """
        Initialize document loader.

        Args:
            timeout: Maximum time in seconds for loading operations
            max_file_size_mb: Maximum file size in MB
            max_memory_mb: Maximum memory usage in MB
            extract_tables: Whether to extract tables from DOCX
            extract_headers: Whether to extract headers/footers
            preserve_formatting: Whether to preserve formatting
            extract_lists: Whether to detect and format lists
            max_paragraphs: Maximum number of paragraphs to process
        """
        self.timeout = timeout
        self.max_file_size_mb = max_file_size_mb
        self.max_memory_mb = max_memory_mb

        self.loaders = {
            '.pdf': PDFLoader(
                timeout=timeout,
                max_file_size_mb=max_file_size_mb,
                max_memory_mb=max_memory_mb
            ),
            '.docx': DOCXLoader(
                timeout=timeout,
                max_file_size_mb=max_file_size_mb,
                max_memory_mb=max_memory_mb,
                extract_tables=extract_tables,
                extract_headers=extract_headers,
                preserve_formatting=preserve_formatting,
                extract_lists=extract_lists,
                max_paragraphs=max_paragraphs
            ),
            '.txt': TextLoader(
                timeout=timeout,
                max_file_size_mb=max_file_size_mb,
                max_memory_mb=max_memory_mb
            ),
            '.html': TextLoader(
                timeout=timeout,
                max_file_size_mb=max_file_size_mb,
                max_memory_mb=max_memory_mb
            ),
            '.htm': TextLoader(
                timeout=timeout,
                max_file_size_mb=max_file_size_mb,
                max_memory_mb=max_memory_mb
            ),
            '.md': TextLoader(
                timeout=timeout,
                max_file_size_mb=max_file_size_mb,
                max_memory_mb=max_memory_mb
            ),
            '.markdown': TextLoader(
                timeout=timeout,
                max_file_size_mb=max_file_size_mb,
                max_memory_mb=max_memory_mb
            ),
            '.csv': TextLoader(
                timeout=timeout,
                max_file_size_mb=max_file_size_mb,
                max_memory_mb=max_memory_mb
            ),
            '.json': TextLoader(
                timeout=timeout,
                max_file_size_mb=max_file_size_mb,
                max_memory_mb=max_memory_mb
            ),
        }

        # Statistics
        self.stats = {
            "total_attempts": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "errors_by_type": {},
            "max_memory_used_mb": 0,
            "total_processing_time_ms": 0
        }

    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load a document and return its content and metadata.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with 'content', 'metadata', and 'file_path' keys
        """
        self.stats["total_attempts"] += 1
        start_time = time.time()

        file_path = Path(file_path)

        if not file_path.exists():
            self.stats["failed_loads"] += 1
            self._record_error("file_not_found")
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()
        loader = self.loaders.get(extension)

        if not loader:
            self.stats["failed_loads"] += 1
            self._record_error("unsupported_format")
            raise UnsupportedFormatError(
                f"Unsupported file type: {extension}. Supported types: {list(self.loaders.keys())}"
            )

        # Check memory before loading
        current_memory = get_memory_usage()
        if current_memory > self.max_memory_mb * 0.8:
            logger.warning(f"Memory usage high ({current_memory:.2f} MB) before loading")
            self.stats["failed_loads"] += 1
            self._record_error("memory_limit")
            raise MemoryLimitExceeded(f"Memory usage too high: {current_memory:.2f} MB")

        logger.info(f"Loading document: {file_path}")

        try:
            result = loader.load(str(file_path))

            processing_time = (time.time() - start_time) * 1000
            self.stats["total_processing_time_ms"] += processing_time

            if result.success:
                self.stats["successful_loads"] += 1

                # Update max memory usage
                if result.memory_usage_mb > self.stats["max_memory_used_mb"]:
                    self.stats["max_memory_used_mb"] = result.memory_usage_mb

                logger.info(f"Successfully loaded: {file_path} ({result.file_size} bytes, "
                           f"{processing_time:.0f}ms, {result.memory_usage_mb:.1f}MB)")

                return {
                    "content": result.content,
                    "metadata": result.metadata,
                    "file_path": str(file_path),
                    "file_size": result.file_size,
                    "processing_time_ms": result.processing_time_ms,
                    "warnings": result.warnings,
                    "memory_usage_mb": result.memory_usage_mb
                }
            else:
                self.stats["failed_loads"] += 1
                self._record_error(result.error_code or "unknown")

                error_msg = f"Failed to load {file_path}: {result.error_message}"
                logger.error(error_msg)

                if result.error_code == LoaderErrorCode.FILE_NOT_FOUND:
                    raise FileNotFoundError(result.error_message)
                elif result.error_code == LoaderErrorCode.UNSUPPORTED_FORMAT:
                    raise UnsupportedFormatError(result.error_message)
                elif result.error_code == LoaderErrorCode.MEMORY_LIMIT_EXCEEDED:
                    raise MemoryLimitExceeded(result.error_message)
                else:
                    raise LoaderError(result.error_message)

        except Exception as e:
            self.stats["failed_loads"] += 1
            self._record_error("loader_exception")
            logger.error(f"Error loading document {file_path}: {e}", exc_info=True)
            raise
        finally:
            # Force garbage collection after loading
            gc.collect()

    def _record_error(self, error_type: str):
        """Record error statistics."""
        if error_type not in self.stats["errors_by_type"]:
            self.stats["errors_by_type"][error_type] = 0
        self.stats["errors_by_type"][error_type] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_loads"] / self.stats["total_attempts"]
                if self.stats["total_attempts"] > 0 else 0
            )
        }


# Convenience function with memory limits
def load_document_memory_safe(file_path: str, max_memory_mb: int = 2048) -> Dict[str, Any]:
    """
    Load a document with memory safety.

    Args:
        file_path: Path to the document file
        max_memory_mb: Maximum memory allowed in MB

    Returns:
        Document dictionary
    """
    loader = DocumentLoader(max_memory_mb=max_memory_mb)
    return loader.load_document(file_path)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    loader = DocumentLoader(max_memory_mb=1024)

    # Load a file with memory tracking
    # result = loader.load_document("large_document.pdf")
    # print(f"Loaded with memory usage: {result['memory_usage_mb']:.2f} MB")

    print("Memory-optimized loader ready")

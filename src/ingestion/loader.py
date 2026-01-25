"""
Document loader module with support for multiple file formats including PDF and DOCX.
IMPROVED: Enhanced DOCX support with table extraction, header preservation, and metadata.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from abc import ABC, abstractmethod
from functools import wraps
from enum import Enum
from dataclasses import dataclass

import PyPDF2
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table, _Cell
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.oxml import parse_xml
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
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


def handle_loader_errors(default_return: Any = None, retry_count: int = 3, retry_delay: float = 1.0):
    """
    Decorator for error handling in loader methods.

    Args:
        default_return: Default value to return on error
        retry_count: Number of retry attempts
        retry_delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_error = None

            for attempt in range(retry_count):
                try:
                    start_time = time.time()
                    result = func(self, *args, **kwargs)

                    # If result is LoaderResult, add processing time
                    if isinstance(result, LoaderResult):
                        result.processing_time_ms = (time.time() - start_time) * 1000

                    return result

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

                except Exception as e:
                    logger.error(f"Unexpected error: {e}", exc_info=True)
                    last_error = e

                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)

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
    """Abstract base class for document loaders."""

    def __init__(self, timeout: int = 30, max_file_size_mb: int = 100):
        self.timeout = timeout
        self.max_file_size_mb = max_file_size_mb

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

        return True, None, None


class PDFLoader(BaseLoader):
    """Loader for PDF documents."""

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """Extract text from PDF file."""
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

        try:
            with open(file_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                except Exception as e:
                    raise CorruptedFileError(f"Invalid or corrupted PDF file: {e}")

                if len(pdf_reader.pages) == 0:
                    raise ParsingError("PDF has no pages")

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text_content.append(f"[Page {page_num}]\n{text}")
                        else:
                            warnings.append(f"No text found on page {page_num}")
                    except Exception as e:
                        warnings.append(f"Failed to extract page {page_num}: {e}")
                        continue

                if not text_content:
                    raise ParsingError("No text content extracted from PDF")

        except CorruptedFileError:
            raise
        except Exception as e:
            raise ParsingError(f"Failed to load PDF: {e}")

        content = "\n\n".join(text_content)

        return LoaderResult(
            success=True,
            content=content,
            metadata=metadata,
            file_path=file_path,
            file_size=file_size,
            warnings=warnings
        )

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF metadata."""
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
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["num_pages"] = len(pdf_reader.pages)

                if pdf_reader.metadata:
                    meta = pdf_reader.metadata
                    metadata["title"] = meta.get('/Title', None)
                    metadata["author"] = meta.get('/Author', None)
                    metadata["subject"] = meta.get('/Subject', None)
                    metadata["creator"] = meta.get('/Creator', None)
                    metadata["producer"] = meta.get('/Producer', None)

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return metadata


class DOCXLoader(BaseLoader):
    """
    Enhanced loader for DOCX documents with full support for:
    - Text extraction with formatting preservation
    - Table extraction with structure preservation
    - Header/footer extraction
    - List detection and formatting
    - Metadata extraction
    - Image placeholder handling
    """

    def __init__(self, timeout: int = 30, max_file_size_mb: int = 100,
                 extract_tables: bool = True, extract_headers: bool = True,
                 preserve_formatting: bool = True, extract_lists: bool = True):
        """
        Initialize DOCX loader.

        Args:
            timeout: Maximum time in seconds for loading operations
            max_file_size_mb: Maximum file size in MB
            extract_tables: Whether to extract table content
            extract_headers: Whether to extract headers/footers
            preserve_formatting: Whether to preserve formatting (bold, italic)
            extract_lists: Whether to detect and format lists
        """
        super().__init__(timeout, max_file_size_mb)
        self.extract_tables = extract_tables
        self.extract_headers = extract_headers
        self.preserve_formatting = preserve_formatting
        self.extract_lists = extract_lists

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """Extract text from DOCX file with enhanced features."""
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

        try:
            try:
                doc = Document(file_path)
            except Exception as e:
                raise CorruptedFileError(f"Failed to open DOCX file: {e}")

            # Extract metadata
            metadata = self._extract_docx_metadata(doc, file_path, file_size)

            # Extract headers
            if self.extract_headers:
                header_text = self._extract_headers(doc)
                if header_text:
                    text_content.append("=== HEADERS ===\n" + header_text)
                    metadata["has_headers"] = True

            # Extract main content with formatting
            content_parts = self._extract_paragraphs(doc)
            text_content.extend(content_parts)

            # Extract tables
            if self.extract_tables and doc.tables:
                table_texts = self._extract_tables(doc.tables)
                if table_texts:
                    text_content.append("\n=== TABLES ===\n")
                    text_content.extend(table_texts)
                    metadata["num_tables"] = len(doc.tables)

            # Extract footers
            if self.extract_headers:
                footer_text = self._extract_footers(doc)
                if footer_text:
                    text_content.append("\n=== FOOTERS ===\n" + footer_text)
                    metadata["has_footers"] = True

            if not text_content:
                raise ParsingError("No text content extracted from DOCX")

            # Combine content
            content = "\n\n".join(text_content)

            # Additional metadata
            metadata["num_paragraphs"] = len(doc.paragraphs)
            metadata["num_sections"] = len(doc.sections) if hasattr(doc, 'sections') else 0

            return LoaderResult(
                success=True,
                content=content,
                metadata=metadata,
                file_path=file_path,
                file_size=file_size,
                warnings=warnings
            )

        except CorruptedFileError:
            raise
        except Exception as e:
            raise ParsingError(f"Failed to load DOCX: {e}")

    def _extract_docx_metadata(self, doc: Document, file_path: str, file_size: int) -> Dict[str, Any]:
        """Extract DOCX metadata including core properties and custom properties."""
        metadata = {
            "file_path": file_path,
            "file_type": "docx",
            "file_size": file_size,
            "num_paragraphs": len(doc.paragraphs),
            "num_tables": len(doc.tables),
            "num_sections": len(doc.sections) if hasattr(doc, 'sections') else 0,
            "author": None,
            "title": None,
            "subject": None,
            "keywords": None,
            "category": None,
            "comments": None,
            "created": None,
            "modified": None,
            "last_modified_by": None,
            "revision": None
        }

        try:
            if doc.core_properties:
                cp = doc.core_properties
                metadata["author"] = str(cp.author) if cp.author else None
                metadata["title"] = str(cp.title) if cp.title else None
                metadata["subject"] = str(cp.subject) if cp.subject else None
                metadata["keywords"] = str(cp.keywords) if cp.keywords else None
                metadata["category"] = str(cp.category) if cp.category else None
                metadata["comments"] = str(cp.comments) if cp.comments else None
                metadata["created"] = str(cp.created) if cp.created else None
                metadata["modified"] = str(cp.modified) if cp.modified else None
                metadata["last_modified_by"] = str(cp.last_modified_by) if cp.last_modified_by else None
                metadata["revision"] = str(cp.revision) if cp.revision else None

        except Exception as e:
            logger.warning(f"Failed to extract core properties from {file_path}: {e}")

        # Try to extract custom properties
        try:
            if hasattr(doc, 'custom_properties'):
                custom_props = {}
                for prop in doc.custom_properties:
                    custom_props[prop.name] = prop.value
                if custom_props:
                    metadata["custom_properties"] = custom_props
        except Exception as e:
            logger.warning(f"Failed to extract custom properties: {e}")

        return metadata

    def _extract_paragraphs(self, doc: Document) -> List[str]:
        """Extract paragraphs with formatting and list detection."""
        content_parts = []
        list_counter = 0
        in_list = False
        list_type = None  # 'bullet' or 'numbered'

        for para in doc.paragraphs:
            if not para.text or not para.text.strip():
                continue

            # Check if paragraph is a heading
            if self._is_heading(para):
                # Format heading
                heading_level = self._get_heading_level(para)
                content_parts.append(f"\n{'#' * heading_level} {para.text.strip()}")
                in_list = False
                list_counter = 0
                continue

            # Check if paragraph is in a list
            if self.extract_lists and self._is_list_item(para):
                list_item_type = self._get_list_type(para)

                if not in_list or list_item_type != list_type:
                    in_list = True
                    list_type = list_item_type
                    list_counter = 1

                # Format list item
                if list_type == 'bullet':
                    content_parts.append(f"  • {para.text.strip()}")
                elif list_type == 'numbered':
                    content_parts.append(f"  {list_counter}. {para.text.strip()}")
                    list_counter += 1
                continue
            else:
                if in_list:
                    in_list = False
                    list_counter = 0
                    list_type = None

            # Regular paragraph with formatting
            formatted_text = self._format_paragraph_text(para)
            content_parts.append(formatted_text)

        return content_parts

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

        # Extract number from "Heading 1", "Heading 2", etc.
        import re
        match = re.search(r'heading\s*(\d+)', para.style.name.lower())
        if match:
            return min(int(match.group(1)), 6)  # Max heading level 6

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

        # Check text pattern
        if re.match(r'^[\s]*[•●○■▪▫]', para.text):
            return 'bullet'
        if re.match(r'^[\s]*\d+[\.\)]', para.text):
            return 'numbered'

        return 'bullet'  # Default

    def _extract_tables(self, tables: List[Table]) -> List[str]:
        """Extract tables with structure preservation."""
        table_texts = []

        for table_idx, table in enumerate(tables, 1):
            table_lines = []
            table_lines.append(f"Table {table_idx}:")

            # Extract headers (first row)
            if table.rows:
                header_row = table.rows[0]
                headers = [cell.text.strip() for cell in header_row.cells]
                if headers:
                    table_lines.append("  " + " | ".join(headers))
                    table_lines.append("  " + "-" * (sum(len(h) for h in headers) + len(headers) * 3))

            # Extract data rows
            for row_idx, row in enumerate(table.rows[1:], 1):
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):  # Only add non-empty rows
                    table_lines.append(f"  Row {row_idx}: " + " | ".join(cells))

            table_texts.append("\n".join(table_lines))

        return table_texts

    def _extract_headers(self, doc: Document) -> str:
        """Extract header content from document."""
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

    def _extract_footers(self, doc: Document) -> str:
        """Extract footer content from document."""
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

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract DOCX metadata with error handling."""
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

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return metadata


class TextLoader(BaseLoader):
    """Loader for plain text files with encoding detection."""

    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read(10000)
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        except Exception:
            return 'utf-8'

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """Load plain text file with encoding detection."""
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
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
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
        metadata["line_count"] = len(content.splitlines())

        return LoaderResult(
            success=True,
            content=content,
            metadata=metadata,
            file_path=file_path,
            file_size=file_size
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
    """Main document loader that routes to appropriate loader based on file extension."""

    def __init__(self, timeout: int = 30, max_file_size_mb: int = 100,
                 extract_tables: bool = True, extract_headers: bool = True,
                 preserve_formatting: bool = True, extract_lists: bool = True):
        """
        Initialize document loader.

        Args:
            timeout: Maximum time in seconds for loading operations
            max_file_size_mb: Maximum file size in MB
            extract_tables: Whether to extract tables from DOCX
            extract_headers: Whether to extract headers/footers
            preserve_formatting: Whether to preserve formatting
            extract_lists: Whether to detect and format lists
        """
        self.timeout = timeout
        self.max_file_size_mb = max_file_size_mb

        self.loaders = {
            '.pdf': PDFLoader(timeout=timeout, max_file_size_mb=max_file_size_mb),
            '.docx': DOCXLoader(
                timeout=timeout,
                max_file_size_mb=max_file_size_mb,
                extract_tables=extract_tables,
                extract_headers=extract_headers,
                preserve_formatting=preserve_formatting,
                extract_lists=extract_lists
            ),
            '.txt': TextLoader(timeout=timeout, max_file_size_mb=max_file_size_mb),
            '.html': HTMLLoader(timeout=timeout, max_file_size_mb=max_file_size_mb),
            '.htm': HTMLLoader(timeout=timeout, max_file_size_mb=max_file_size_mb),
            '.md': MarkdownLoader(timeout=timeout, max_file_size_mb=max_file_size_mb),
            '.markdown': MarkdownLoader(timeout=timeout, max_file_size_mb=max_file_size_mb),
            '.csv': CSVLoader(timeout=timeout, max_file_size_mb=max_file_size_mb),
            '.json': JSONLoader(timeout=timeout, max_file_size_mb=max_file_size_mb),
        }

        # Statistics
        self.stats = {
            "total_attempts": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "errors_by_type": {}
        }

    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load a document and return its content and metadata.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with 'content', 'metadata', and 'file_path' keys

        Raises:
            FileNotFoundError: If file doesn't exist
            UnsupportedFormatError: If file type is not supported
            LoaderError: For other loading errors
        """
        self.stats["total_attempts"] += 1

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

        logger.info(f"Loading document: {file_path}")

        try:
            result = loader.load(str(file_path))

            if result.success:
                self.stats["successful_loads"] += 1
                logger.info(f"Successfully loaded: {file_path} ({result.file_size} bytes, "
                           f"{result.processing_time_ms:.0f}ms)")

                return {
                    "content": result.content,
                    "metadata": result.metadata,
                    "file_path": str(file_path),
                    "file_size": result.file_size,
                    "processing_time_ms": result.processing_time_ms,
                    "warnings": result.warnings
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
                else:
                    raise LoaderError(result.error_message)

        except Exception as e:
            self.stats["failed_loads"] += 1
            self._record_error("loader_exception")
            logger.error(f"Error loading document {file_path}: {e}", exc_info=True)
            raise

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


if __name__ == "__main__":
    # Example usage with DOCX
    logging.basicConfig(level=logging.INFO)

    loader = DocumentLoader(
        extract_tables=True,
        extract_headers=True,
        preserve_formatting=True,
        extract_lists=True
    )

    # Load a DOCX file
    # result = loader.load_document("sample.docx")
    # print(f"Content length: {len(result['content'])}")
    # print(f"Metadata: {result['metadata']}")

    print("DOCX Loader ready with enhanced features")

"""
Document loader module with support for multiple file formats including PDF.
IMPROVED: Enhanced error handling, retry logic, and graceful degradation.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from abc import ABC, abstractmethod
from functools import wraps
from enum import Enum

import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import markdown
import csv
import json
import chardet

logger = logging.getLogger(__name__)


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


class LoaderErrorCode(Enum):
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


@dataclass
class LoaderResult:
    """Result object for loader operations."""
    success: bool
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_code: Optional[LoaderErrorCode] = None
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    file_size: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "content": self.content,
            "metadata": self.metadata,
            "error_code": self.error_code.value if self.error_code else None,
            "error_message": self.error_message,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "processing_time_ms": self.processing_time_ms
        }


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
                    break  # Don't retry file not found

                except UnsupportedFormatError as e:
                    logger.error(f"Unsupported format: {e}")
                    last_error = e
                    break  # Don't retry unsupported format

                except PermissionError as e:
                    logger.error(f"Permission denied: {e}")
                    last_error = e
                    break  # Don't retry permission errors

                except (CorruptedFileError, ParsingError, UnicodeDecodeError) as e:
                    logger.warning(f"Attempt {attempt + 1}/{retry_count} failed: {e}")
                    last_error = e

                    if attempt < retry_count - 1:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

                except Exception as e:
                    logger.error(f"Unexpected error: {e}", exc_info=True)
                    last_error = e

                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)

            # Return default value on error
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
        """
        Initialize base loader.

        Args:
            timeout: Maximum time in seconds for loading operations
            max_file_size_mb: Maximum file size in MB
        """
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

    def validate_file(self, file_path: str) -> Tuple[bool, Optional[LoaderErrorCode], Optional[str]]:
        """
        Validate file before loading.

        Returns:
            Tuple of (is_valid, error_code, error_message)
        """
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            return False, LoaderErrorCode.FILE_NOT_FOUND, f"File not found: {file_path}"

        # Check if it's a file (not directory)
        if not path.is_file():
            return False, LoaderErrorCode.UNKNOWN_ERROR, f"Path is not a file: {file_path}"

        # Check file size
        file_size = path.stat().st_size
        max_size_bytes = self.max_file_size_mb * 1024 * 1024

        if file_size == 0:
            return False, LoaderErrorCode.EMPTY_FILE, f"File is empty: {file_path}"

        if file_size > max_size_bytes:
            return False, LoaderErrorCode.UNKNOWN_ERROR, f"File too large: {file_size} bytes (max {max_size_bytes})"

        # Check read permission
        if not os.access(file_path, os.R_OK):
            return False, LoaderErrorCode.PERMISSION_DENIED, f"No read permission: {file_path}"

        return True, None, None


class PDFLoader(BaseLoader):
    """Loader for PDF documents with improved error handling."""

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """Extract text from PDF file with error handling."""
        # Validate file
        is_valid, error_code, error_msg = self.validate_file(file_path)
        if not is_valid:
            return LoaderResult(
                success=False,
                error_code=error_code,
                error_message=error_msg,
                file_path=file_path
            )

        text_content = []
        metadata = self.get_metadata(file_path)
        file_size = Path(file_path).stat().st_size

        try:
            with open(file_path, 'rb') as file:
                # Check if file is valid PDF
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                except Exception as e:
                    raise CorruptedFileError(f"Invalid or corrupted PDF file: {e}")

                if len(pdf_reader.pages) == 0:
                    raise ParsingError("PDF has no pages")

                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text_content.append(f"[Page {page_num}]\n{text}")
                        else:
                            logger.debug(f"No text found on page {page_num} of {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num} from {file_path}: {e}")
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
            file_size=file_size
        )

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF metadata with error handling."""
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
    """Loader for DOCX documents with improved error handling."""

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """Extract text from DOCX file with error handling."""
        is_valid, error_code, error_msg = self.validate_file(file_path)
        if not is_valid:
            return LoaderResult(
                success=False,
                error_code=error_code,
                error_message=error_msg,
                file_path=file_path
            )

        text_content = []
        metadata = self.get_metadata(file_path)
        file_size = Path(file_path).stat().st_size

        try:
            try:
                doc = Document(file_path)
            except Exception as e:
                raise CorruptedFileError(f"Failed to open DOCX file: {e}")

            # Extract paragraphs
            paragraph_count = 0
            for paragraph in doc.paragraphs:
                if paragraph.text and paragraph.text.strip():
                    text_content.append(paragraph.text)
                    paragraph_count += 1

            # Extract tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text and cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_content.append(" | ".join(row_text))

            if not text_content:
                raise ParsingError("No text content extracted from DOCX")

            metadata["num_paragraphs"] = paragraph_count
            metadata["num_tables"] = len(doc.tables)

        except CorruptedFileError:
            raise
        except Exception as e:
            raise ParsingError(f"Failed to load DOCX: {e}")

        content = "\n\n".join(text_content)

        return LoaderResult(
            success=True,
            content=content,
            metadata=metadata,
            file_path=file_path,
            file_size=file_size
        )

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract DOCX metadata with error handling."""
        metadata = {
            "file_path": file_path,
            "file_type": "docx",
            "file_size": Path(file_path).stat().st_size,
            "num_paragraphs": 0,
            "num_tables": 0,
            "author": None,
            "title": None,
            "subject": None,
            "keywords": None
        }

        try:
            doc = Document(file_path)

            if doc.core_properties:
                metadata["author"] = str(doc.core_properties.author) if doc.core_properties.author else None
                metadata["title"] = str(doc.core_properties.title) if doc.core_properties.title else None
                metadata["subject"] = str(doc.core_properties.subject) if doc.core_properties.subject else None
                metadata["keywords"] = str(doc.core_properties.keywords) if doc.core_properties.keywords else None

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return metadata


class TextLoader(BaseLoader):
    """Loader for plain text files with encoding detection."""

    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read(10000)  # Read first 10KB for detection
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


class HTMLLoader(BaseLoader):
    """Loader for HTML documents with improved error handling."""

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """Extract text from HTML file with error handling."""
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

        try:
            # Try multiple encodings
            content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        html_content = file.read()
                        content = self._extract_text_from_html(html_content)
                        if content and content.strip():
                            break
                except UnicodeDecodeError:
                    continue

            if content is None or not content.strip():
                raise ParsingError("No text content extracted from HTML")

        except Exception as e:
            raise ParsingError(f"Failed to load HTML: {e}")

        return LoaderResult(
            success=True,
            content=content,
            metadata=metadata,
            file_path=file_path,
            file_size=file_size
        )

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract text from HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text

        except Exception as e:
            raise ParsingError(f"Failed to parse HTML: {e}")

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract HTML metadata with error handling."""
        metadata = {
            "file_path": file_path,
            "file_type": "html",
            "file_size": Path(file_path).stat().st_size,
            "title": None,
            "description": None,
            "keywords": None
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')

                if soup.title:
                    metadata["title"] = soup.title.string

                # Extract meta tags
                for meta in soup.find_all('meta'):
                    if meta.get('name') == 'description':
                        metadata["description"] = meta.get('content')
                    elif meta.get('name') == 'keywords':
                        metadata["keywords"] = meta.get('content')

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return metadata


class DocumentLoader:
    """Main document loader that routes to appropriate loader based on file extension."""

    def __init__(self, timeout: int = 30, max_file_size_mb: int = 100):
        """
        Initialize document loader.

        Args:
            timeout: Maximum time in seconds for loading operations
            max_file_size_mb: Maximum file size in MB
        """
        self.timeout = timeout
        self.max_file_size_mb = max_file_size_mb

        self.loaders = {
            '.pdf': PDFLoader(timeout=timeout, max_file_size_mb=max_file_size_mb),
            '.docx': DOCXLoader(timeout=timeout, max_file_size_mb=max_file_size_mb),
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

        # Check if file exists
        if not file_path.exists():
            self.stats["failed_loads"] += 1
            self._record_error("file_not_found")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get loader based on extension
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
                    "processing_time_ms": result.processing_time_ms
                }
            else:
                self.stats["failed_loads"] += 1
                self._record_error(result.error_code.value if result.error_code else "unknown")

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

    def load_document_safe(self, file_path: str) -> LoaderResult:
        """
        Load document safely without raising exceptions.

        Args:
            file_path: Path to the document file

        Returns:
            LoaderResult object (always returned, never raises)
        """
        try:
            result = self.load_document(file_path)
            return LoaderResult(
                success=True,
                content=result["content"],
                metadata=result["metadata"],
                file_path=result["file_path"],
                file_size=result["file_size"],
                processing_time_ms=result["processing_time_ms"]
            )
        except FileNotFoundError as e:
            return LoaderResult(
                success=False,
                error_code=LoaderErrorCode.FILE_NOT_FOUND,
                error_message=str(e),
                file_path=str(file_path)
            )
        except UnsupportedFormatError as e:
            return LoaderResult(
                success=False,
                error_code=LoaderErrorCode.UNSUPPORTED_FORMAT,
                error_message=str(e),
                file_path=str(file_path)
            )
        except Exception as e:
            return LoaderResult(
                success=False,
                error_code=LoaderErrorCode.UNKNOWN_ERROR,
                error_message=str(e),
                file_path=str(file_path)
            )

    def load_directory(
        self,
        directory_path: str,
        extensions: Optional[List[str]] = None,
        skip_errors: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load all documents from a directory.

        Args:
            directory_path: Path to directory containing documents
            extensions: List of extensions to filter (e.g., ['.pdf', '.txt'])
            skip_errors: Whether to skip files that cause errors

        Returns:
            List of document dictionaries
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not directory_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory_path}")

        documents = []
        failed_files = []

        # Default to all supported extensions if not specified
        if extensions is None:
            extensions = list(self.loaders.keys())

        total_files = 0
        for extension in extensions:
            for file_path in directory_path.glob(f"**/*{extension}"):
                total_files += 1

                try:
                    doc = self.load_document(str(file_path))
                    documents.append(doc)
                    logger.debug(f"Loaded: {file_path}")
                except Exception as e:
                    error_msg = f"Failed to load {file_path}: {e}"
                    if skip_errors:
                        logger.warning(error_msg)
                        failed_files.append(str(file_path))
                    else:
                        logger.error(error_msg)
                        raise

        logger.info(f"Loaded {len(documents)}/{total_files} documents from {directory_path}")

        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files[:5]}...")

        return documents

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

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_attempts": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "errors_by_type": {}
        }


# Add missing loader classes
class MarkdownLoader(BaseLoader):
    """Loader for Markdown files."""

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """Convert markdown to text."""
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

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                content = soup.get_text()

                if not content.strip():
                    raise ParsingError("No text content extracted from Markdown")

        except Exception as e:
            raise ParsingError(f"Failed to load Markdown: {e}")

        return LoaderResult(
            success=True,
            content=content,
            metadata=metadata,
            file_path=file_path,
            file_size=file_size
        )

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract markdown metadata."""
        return {
            "file_path": file_path,
            "file_type": "md",
            "file_size": Path(file_path).stat().st_size
        }


class CSVLoader(BaseLoader):
    """Loader for CSV files."""

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """Convert CSV to readable text format."""
        is_valid, error_code, error_msg = self.validate_file(file_path)
        if not is_valid:
            return LoaderResult(
                success=False,
                error_code=error_code,
                error_message=error_msg,
                file_path=file_path
            )

        text_content = []
        metadata = self.get_metadata(file_path)
        file_size = Path(file_path).stat().st_size

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)

                if not rows:
                    raise ParsingError("CSV file has no rows")

                headers = rows[0] if rows else []
                text_content.append("Columns: " + ", ".join(headers))

                for row_num, row in enumerate(rows[1:], 1):
                    row_text = f"Row {row_num}: " + ", ".join(
                        f"{headers[i] if i < len(headers) else f'col{i}'}: {val}"
                        for i, val in enumerate(row)
                    )
                    text_content.append(row_text)

                metadata["num_rows"] = len(rows)
                metadata["num_columns"] = len(headers)

        except Exception as e:
            raise ParsingError(f"Failed to load CSV: {e}")

        content = "\n".join(text_content)

        return LoaderResult(
            success=True,
            content=content,
            metadata=metadata,
            file_path=file_path,
            file_size=file_size
        )

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract CSV metadata."""
        metadata = {
            "file_path": file_path,
            "file_type": "csv",
            "file_size": Path(file_path).stat().st_size,
            "num_rows": 0,
            "num_columns": 0
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)
                metadata["num_rows"] = len(rows)
                if rows:
                    metadata["num_columns"] = len(rows[0])

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return metadata


class JSONLoader(BaseLoader):
    """Loader for JSON files."""

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """Load and format JSON content."""
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

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                content = json.dumps(data, indent=2)

        except json.JSONDecodeError as e:
            raise ParsingError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ParsingError(f"Failed to load JSON: {e}")

        return LoaderResult(
            success=True,
            content=content,
            metadata=metadata,
            file_path=file_path,
            file_size=file_size
        )

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract JSON metadata."""
        return {
            "file_path": file_path,
            "file_type": "json",
            "file_size": Path(file_path).stat().st_size
        }


# Convenience function for quick loading
def load_documents_safe(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load multiple documents safely, skipping failed ones.

    Args:
        file_paths: List of file paths to load

    Returns:
        List of successfully loaded document dictionaries
    """
    loader = DocumentLoader()
    documents = []

    for file_path in file_paths:
        result = loader.load_document_safe(file_path)
        if result.success:
            documents.append({
                "content": result.content,
                "metadata": result.metadata,
                "file_path": result.file_path,
                "file_size": result.file_size
            })

    return documents


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    loader = DocumentLoader()

    # Load a single file safely
    result = loader.load_document_safe("sample.pdf")
    if result.success:
        print(f"Loaded: {len(result.content)} characters")
    else:
        print(f"Error: {result.error_message}")

    # Load directory with error handling
    docs = loader.load_directory("./data/raw/", skip_errors=True)
    print(f"Loaded {len(docs)} documents")

    # Get statistics
    stats = loader.get_stats()
    print(f"Success rate: {stats['success_rate']:.2%}")

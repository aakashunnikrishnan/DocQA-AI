"""
Document loader module with support for multiple file formats including PDF, DOCX, and HTML.
ENHANCED: Full HTML support with BeautifulSoup parsing, metadata extraction, and content cleaning.
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
import re
from urllib.parse import urlparse, urljoin

import PyPDF2
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table, _Cell
from bs4 import BeautifulSoup, Comment, NavigableString
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
        if end_mem - start_mem > 10:
            logger.debug(f"Memory increase in {operation_name}: {end_mem - start_mem:.2f} MB")
        gc.collect()


def handle_loader_errors(default_return: Any = None, retry_count: int = 2, retry_delay: float = 1.0,
                         max_memory_mb: int = 2048):
    """
    Decorator for error handling with memory limit checking.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_error = None
            memory_start = get_memory_usage()

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
                    gc.collect()

                    start_time = time.time()
                    result = func(self, *args, **kwargs)

                    memory_after = get_memory_usage()

                    if isinstance(result, LoaderResult):
                        result.processing_time_ms = (time.time() - start_time) * 1000
                        result.memory_usage_mb = memory_after - memory_start

                    gc.collect()
                    return result

                except MemoryError as e:
                    logger.error(f"Memory error in {func.__name__}: {e}")
                    last_error = e
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
                        gc.collect()

                except Exception as e:
                    logger.error(f"Unexpected error: {e}", exc_info=True)
                    last_error = e

                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        gc.collect()

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
                 max_memory_mb: int = 2048, chunk_size: int = 1024 * 1024):
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

        current_memory = get_memory_usage()
        if current_memory > self.max_memory_mb * 0.8:
            return False, LoaderErrorCode.MEMORY_LIMIT_EXCEEDED, f"Memory usage too high: {current_memory:.2f} MB"

        return True, None, None


# ============================================================
# HTML Loader - NEW ENHANCED IMPLEMENTATION
# ============================================================

class HTMLLoader(BaseLoader):
    """
    Enhanced HTML document loader with support for:
    - HTML parsing with BeautifulSoup
    - Metadata extraction (title, meta tags, Open Graph)
    - Content cleaning (remove scripts, styles, navigation)
    - Table extraction
    - Link extraction
    - Semantic structure preservation
    - Encoding detection
    """

    def __init__(
        self,
        timeout: int = 60,
        max_file_size_mb: int = 100,
        max_memory_mb: int = 2048,
        chunk_size: int = 1024 * 1024,
        extract_tables: bool = True,
        extract_links: bool = True,
        extract_metadata: bool = True,
        preserve_structure: bool = True,
        clean_content: bool = True,
        max_content_length: int = 1000000  # 1MB max content
    ):
        """
        Initialize HTML loader.

        Args:
            timeout: Maximum time in seconds for loading operations
            max_file_size_mb: Maximum file size in MB
            max_memory_mb: Maximum memory usage in MB
            chunk_size: Chunk size for streaming reads
            extract_tables: Whether to extract tables from HTML
            extract_links: Whether to extract links
            extract_metadata: Whether to extract metadata
            preserve_structure: Whether to preserve HTML structure in text
            clean_content: Whether to clean content (remove scripts, styles, etc.)
            max_content_length: Maximum content length to extract
        """
        super().__init__(timeout, max_file_size_mb, max_memory_mb, chunk_size)
        self.extract_tables = extract_tables
        self.extract_links = extract_links
        self.extract_metadata = extract_metadata
        self.preserve_structure = preserve_structure
        self.clean_content = clean_content
        self.max_content_length = max_content_length

        # Elements to remove during cleaning
        self.remove_elements = [
            'script', 'style', 'nav', 'footer', 'header',
            'aside', 'form', 'input', 'button', 'noscript',
            'iframe', 'embed', 'object', 'applet'
        ]

        # Elements to treat as structure
        self.structure_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'section', 'article']

        logger.info(f"HTMLLoader initialized: extract_tables={extract_tables}, "
                   f"extract_links={extract_links}, preserve_structure={preserve_structure}")

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """
        Load and parse HTML document.

        Args:
            file_path: Path to HTML file

        Returns:
            LoaderResult with content and metadata
        """
        is_valid, error_code, error_msg = self.validate_file(file_path)
        if not is_valid:
            return LoaderResult(
                success=False,
                error_code=error_code,
                error_message=error_msg,
                file_path=file_path
            )

        warnings = []
        metadata = self.get_metadata(file_path)
        file_size = Path(file_path).stat().st_size

        try:
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            metadata["detected_encoding"] = encoding

            # Read HTML content
            with track_memory("HTML reading"):
                with open(file_path, 'r', encoding=encoding) as f:
                    html_content = f.read()

            # Parse HTML
            with track_memory("HTML parsing"):
                soup = BeautifulSoup(html_content, 'html.parser')

            # Extract metadata
            if self.extract_metadata:
                self._extract_metadata_from_soup(soup, metadata)

            # Clean content
            if self.clean_content:
                self._clean_soup(soup)

            # Extract content
            content_parts = []

            # Extract title
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
                if title:
                    content_parts.append(f"Title: {title}")

            # Extract main content
            main_content = self._extract_main_content(soup)
            if main_content:
                content_parts.append(main_content)

            # Extract tables
            if self.extract_tables:
                tables = self._extract_tables(soup)
                if tables:
                    content_parts.append("\n=== TABLES ===\n")
                    content_parts.extend(tables)
                    metadata["num_tables"] = len(tables)

            # Extract links
            if self.extract_links:
                links = self._extract_links(soup)
                if links:
                    content_parts.append("\n=== LINKS ===\n")
                    content_parts.extend(links)
                    metadata["num_links"] = len(links)

            if not content_parts:
                raise ParsingError("No content extracted from HTML")

            # Join content
            content = "\n\n".join(content_parts)

            # Limit content length if needed
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length]
                warnings.append(f"Content truncated to {self.max_content_length} characters")

            # Clean up soup to free memory
            soup.decompose()
            del soup
            gc.collect()

        except Exception as e:
            raise ParsingError(f"Failed to load HTML: {e}")

        return LoaderResult(
            success=True,
            content=content,
            metadata=metadata,
            file_path=file_path,
            file_size=file_size,
            warnings=warnings,
            memory_usage_mb=get_memory_usage()
        )

    def _detect_encoding(self, file_path: str) -> str:
        """Detect HTML file encoding."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                detected = result.get('encoding', 'utf-8')

                # Check for HTML meta charset
                try:
                    content = raw_data.decode(detected, errors='ignore')
                    meta_match = re.search(r'<meta[^>]*charset=["\']?([^"\' />]+)', content, re.IGNORECASE)
                    if meta_match:
                        return meta_match.group(1)
                except Exception:
                    pass

                return detected
        except Exception:
            return 'utf-8'

    def _clean_soup(self, soup: BeautifulSoup):
        """Clean HTML soup by removing unwanted elements."""
        # Remove unwanted elements
        for element in self.remove_elements:
            for tag in soup.find_all(element):
                tag.decompose()

        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove empty tags
        for tag in soup.find_all():
            if not tag.get_text(strip=True) and not tag.find_all():
                tag.decompose()

        # Remove unwanted attributes
        for tag in soup.find_all():
            if tag.attrs:
                # Keep only useful attributes
                allowed_attrs = ['class', 'id', 'href', 'src', 'alt', 'title']
                for attr in list(tag.attrs.keys()):
                    if attr not in allowed_attrs:
                        del tag[attr]

    def _extract_metadata_from_soup(self, soup: BeautifulSoup, metadata: Dict[str, Any]):
        """Extract metadata from BeautifulSoup object."""
        # Basic metadata
        if soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()

        # Meta tags
        meta_tags = {}

        # Standard meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            property_name = meta.get('property', '').lower()
            content = meta.get('content', '')

            if not content:
                continue

            if name:
                meta_tags[f"meta_{name}"] = content
            elif property_name:
                meta_tags[f"meta_{property_name}"] = content

            # Common meta tags
            if name == 'description':
                metadata["description"] = content
            elif name == 'keywords':
                metadata["keywords"] = [k.strip() for k in content.split(',') if k.strip()]
            elif name == 'author':
                metadata["author"] = content
            elif name == 'robots':
                metadata["robots"] = content

        # Open Graph metadata
        og_tags = {}
        for meta in soup.find_all('meta', property=re.compile(r'^og:')):
            property_name = meta.get('property', '')
            content = meta.get('content', '')
            if property_name and content:
                og_tags[property_name] = content

        if og_tags:
            metadata["open_graph"] = og_tags

        # JSON-LD
        json_ld = []
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                json_ld.append(data)
            except Exception:
                pass

        if json_ld:
            metadata["json_ld"] = json_ld

        # Headings
        headings = {}
        for level in range(1, 7):
            tags = soup.find_all(f'h{level}')
            if tags:
                headings[f'h{level}'] = [tag.get_text(strip=True) for tag in tags[:10]]
        if headings:
            metadata["headings"] = headings

        # Links
        links = soup.find_all('a')
        if links:
            metadata["total_links"] = len(links)
            # Get unique domains
            domains = set()
            for link in links:
                href = link.get('href', '')
                if href:
                    try:
                        parsed = urlparse(href)
                        if parsed.netloc:
                            domains.add(parsed.netloc)
                    except Exception:
                        pass
            if domains:
                metadata["linked_domains"] = list(domains)[:20]

        # Images
        images = soup.find_all('img')
        if images:
            metadata["total_images"] = len(images)
            # Count images with alt text
            alt_count = sum(1 for img in images if img.get('alt'))
            metadata["images_with_alt"] = alt_count

    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract main content from HTML.
        Uses heuristic to find main content area.
        """
        # Try to find main content area
        main_selectors = [
            'main',
            'article',
            'div[role="main"]',
            '.main-content',
            '#main-content',
            '.content',
            '#content',
            '.post-content',
            '.article-content'
        ]

        content = None
        for selector in main_selectors:
            elements = soup.select(selector)
            if elements:
                content = elements[0]
                break

        # If no main content area found, use body
        if content is None:
            content = soup.body if soup.body else soup

        # Extract text while preserving some structure
        if self.preserve_structure:
            return self._extract_structured_text(content)
        else:
            return content.get_text(separator=' ', strip=True)

    def _extract_structured_text(self, element) -> str:
        """
        Extract text while preserving some HTML structure (headings, paragraphs).
        """
        parts = []

        # Process children
        for child in element.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    parts.append(text)
            elif hasattr(child, 'name'):
                tag_name = child.name

                # Headings
                if tag_name and tag_name.startswith('h'):
                    text = child.get_text(strip=True)
                    if text:
                        level = int(tag_name[1]) if tag_name[1:].isdigit() else 1
                        parts.append(f"\n{'#' * level} {text}\n")

                # Paragraphs
                elif tag_name == 'p':
                    text = child.get_text(strip=True)
                    if text:
                        parts.append(text)

                # Lists
                elif tag_name in ['ul', 'ol']:
                    list_items = []
                    for li in child.find_all('li', recursive=False):
                        li_text = li.get_text(strip=True)
                        if li_text:
                            if tag_name == 'ul':
                                list_items.append(f"  • {li_text}")
                            else:
                                list_items.append(f"  {len(list_items) + 1}. {li_text}")
                    if list_items:
                        parts.append("\n".join(list_items))

                # Blockquotes
                elif tag_name == 'blockquote':
                    text = child.get_text(strip=True)
                    if text:
                        parts.append(f"\n> {text}\n")

                # Other block elements
                elif tag_name in ['div', 'section', 'article', 'aside']:
                    inner_text = self._extract_structured_text(child)
                    if inner_text:
                        parts.append(inner_text)

                # Inline elements
                elif tag_name in ['span', 'strong', 'em', 'b', 'i', 'a']:
                    text = child.get_text(strip=True)
                    if text:
                        parts.append(text)

        return '\n\n'.join(parts)

    def _extract_tables(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract tables from HTML.

        Returns:
            List of formatted table strings
        """
        tables = []
        table_tags = soup.find_all('table')

        for table_idx, table in enumerate(table_tags, 1):
            table_lines = []
            table_lines.append(f"Table {table_idx}:")

            # Extract rows
            rows = table.find_all('tr')
            if not rows:
                continue

            # Extract headers
            headers = []
            header_row = rows[0]
            for th in header_row.find_all(['th', 'td']):
                headers.append(th.get_text(strip=True))

            if headers:
                table_lines.append("  " + " | ".join(headers))
                table_lines.append("  " + "-" * (sum(len(h) for h in headers) + len(headers) * 3))

            # Extract data rows
            for row_idx, row in enumerate(rows[1:], 1):
                cells = []
                for td in row.find_all(['td', 'th']):
                    cells.append(td.get_text(strip=True))

                if any(cells):
                    table_lines.append(f"  Row {row_idx}: " + " | ".join(cells))

            tables.append("\n".join(table_lines))

        return tables

    def _extract_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract links from HTML.

        Returns:
            List of formatted link strings
        """
        links = []
        seen_urls = set()

        for a in soup.find_all('a', href=True):
            text = a.get_text(strip=True)
            href = a.get('href', '')

            if not href or href in seen_urls:
                continue

            seen_urls.add(href)

            # Truncate very long URLs
            if len(href) > 200:
                href = href[:197] + "..."

            # Add context
            if text:
                links.append(f"  {text}: {href}")
            else:
                links.append(f"  {href}")

            # Limit number of links
            if len(links) >= 100:
                links.append("  ... and more links")
                break

        return links

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract HTML metadata."""
        metadata = {
            "file_path": file_path,
            "file_type": "html",
            "file_size": Path(file_path).stat().st_size,
            "title": None,
            "description": None,
            "author": None,
            "keywords": [],
            "language": None,
            "encoding": None,
            "total_links": 0,
            "total_images": 0,
            "num_tables": 0,
            "headings": {},
            "open_graph": {},
            "json_ld": []
        }

        try:
            # Get basic file info
            path = Path(file_path)
            metadata["file_size"] = path.stat().st_size
            metadata["file_name"] = path.name

            # Try to extract metadata quickly
            try:
                encoding = self._detect_encoding(file_path)
                metadata["encoding"] = encoding

                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read(50000)  # Read first 50KB for metadata

                soup = BeautifulSoup(content, 'html.parser')
                self._extract_metadata_from_soup(soup, metadata)

                # Clean up
                soup.decompose()
                del soup
                gc.collect()

            except Exception as e:
                logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return metadata


# ============================================================
# Other Loaders (PDF, DOCX, Text, etc.)
# ============================================================

class PDFLoader(BaseLoader):
    """Loader for PDF documents."""
    # ... (existing PDFLoader implementation)
    pass


class DOCXLoader(BaseLoader):
    """Memory-optimized loader for DOCX documents."""
    # ... (existing DOCXLoader implementation)
    pass


class TextLoader(BaseLoader):
    """Loader for plain text files with streaming."""
    # ... (existing TextLoader implementation)
    pass


# ============================================================
# Main Document Loader
# ============================================================

class DocumentLoader:
    """Main document loader with memory management."""

    def __init__(self, timeout: int = 60, max_file_size_mb: int = 100,
                 max_memory_mb: int = 2048, extract_tables: bool = True,
                 extract_headers: bool = True, preserve_formatting: bool = True,
                 extract_lists: bool = True, max_paragraphs: int = 100000,
                 extract_links: bool = True, extract_metadata: bool = True,
                 clean_html: bool = True, preserve_html_structure: bool = True):
        """
        Initialize document loader.

        Args:
            timeout: Maximum time in seconds for loading operations
            max_file_size_mb: Maximum file size in MB
            max_memory_mb: Maximum memory usage in MB
            extract_tables: Whether to extract tables from DOCX/HTML
            extract_headers: Whether to extract headers/footers
            preserve_formatting: Whether to preserve formatting
            extract_lists: Whether to detect and format lists
            max_paragraphs: Maximum number of paragraphs to process
            extract_links: Whether to extract links from HTML
            extract_metadata: Whether to extract metadata from HTML
            clean_html: Whether to clean HTML content
            preserve_html_structure: Whether to preserve HTML structure
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
            '.html': HTMLLoader(
                timeout=timeout,
                max_file_size_mb=max_file_size_mb,
                max_memory_mb=max_memory_mb,
                extract_tables=extract_tables,
                extract_links=extract_links,
                extract_metadata=extract_metadata,
                preserve_structure=preserve_html_structure,
                clean_content=clean_html
            ),
            '.htm': HTMLLoader(
                timeout=timeout,
                max_file_size_mb=max_file_size_mb,
                max_memory_mb=max_memory_mb,
                extract_tables=extract_tables,
                extract_links=extract_links,
                extract_metadata=extract_metadata,
                preserve_structure=preserve_html_structure,
                clean_content=clean_html
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

        current_memory = get_memory_usage()
        if current_memory > self.max_memory_mb * 0.8:
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


if __name__ == "__main__":
    # Example usage with HTML
    logging.basicConfig(level=logging.INFO)

    loader = DocumentLoader(
        extract_tables=True,
        extract_links=True,
        extract_metadata=True,
        clean_html=True,
        preserve_html_structure=True
    )

    # Load an HTML file
    # result = loader.load_document("sample.html")
    # print(f"Content length: {len(result['content'])}")
    # print(f"Metadata: {result['metadata']}")

    print("HTML Loader ready with enhanced features")

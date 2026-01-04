"""
Document loader module with support for multiple file formats including PDF.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from abc import ABC, abstractmethod

import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import markdown
import csv
import json

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: str) -> str:
        """Load document and return text content."""
        pass

    @abstractmethod
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document."""
        pass


class PDFLoader(BaseLoader):
    """Loader for PDF documents."""

    def load(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text_content = []

        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            text_content.append(f"[Page {page_num}]\n{text}")
                        logger.debug(f"Extracted page {page_num} from {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            raise

        return "\n\n".join(text_content)

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = {
            "file_path": file_path,
            "file_type": "pdf",
            "file_size": os.path.getsize(file_path),
            "num_pages": 0,
            "title": None,
            "author": None,
            "subject": None,
            "creator": None
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

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return metadata


class DOCXLoader(BaseLoader):
    """Loader for DOCX documents."""

    def load(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        text_content = []

        try:
            doc = Document(file_path)

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)

            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_content.append(" | ".join(row_text))

        except Exception as e:
            logger.error(f"Failed to load DOCX {file_path}: {e}")
            raise

        return "\n\n".join(text_content)

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract DOCX metadata."""
        metadata = {
            "file_path": file_path,
            "file_type": "docx",
            "file_size": os.path.getsize(file_path),
            "num_paragraphs": 0,
            "author": None,
            "title": None
        }

        try:
            doc = Document(file_path)
            metadata["num_paragraphs"] = len(doc.paragraphs)

            if doc.core_properties:
                metadata["author"] = str(doc.core_properties.author) if doc.core_properties.author else None
                metadata["title"] = str(doc.core_properties.title) if doc.core_properties.title else None

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return metadata


class TextLoader(BaseLoader):
    """Loader for plain text files."""

    def load(self, file_path: str) -> str:
        """Load plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Fallback to different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {e}")
            raise

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract text file metadata."""
        return {
            "file_path": file_path,
            "file_type": "txt",
            "file_size": os.path.getsize(file_path),
            "encoding": "utf-8"
        }


class HTMLLoader(BaseLoader):
    """Loader for HTML documents."""

    def load(self, file_path: str) -> str:
        """Extract text from HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text
                text = soup.get_text()

                # Break into lines and remove leading/trailing space
                lines = (line.strip() for line in text.splitlines())
                # Break multi-headlines into a line each
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                # Drop blank lines
                text = '\n'.join(chunk for chunk in chunks if chunk)

                return text

        except Exception as e:
            logger.error(f"Failed to load HTML {file_path}: {e}")
            raise

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract HTML metadata."""
        metadata = {
            "file_path": file_path,
            "file_type": "html",
            "file_size": os.path.getsize(file_path),
            "title": None
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                if soup.title:
                    metadata["title"] = soup.title.string

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return metadata


class MarkdownLoader(BaseLoader):
    """Loader for Markdown files."""

    def load(self, file_path: str) -> str:
        """Convert markdown to text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                # Convert markdown to HTML then to text
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()

        except Exception as e:
            logger.error(f"Failed to load Markdown {file_path}: {e}")
            raise

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract markdown metadata."""
        return {
            "file_path": file_path,
            "file_type": "md",
            "file_size": os.path.getsize(file_path)
        }


class CSVLoader(BaseLoader):
    """Loader for CSV files."""

    def load(self, file_path: str) -> str:
        """Convert CSV to readable text format."""
        text_content = []

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)

                if rows:
                    # Add headers if present
                    headers = rows[0]
                    text_content.append("Columns: " + ", ".join(headers))

                    # Add data rows
                    for row_num, row in enumerate(rows[1:], 1):
                        row_text = f"Row {row_num}: " + ", ".join(f"{headers[i] if i < len(headers) else f'col{i}'}: {val}"
                                                                   for i, val in enumerate(row))
                        text_content.append(row_text)

        except Exception as e:
            logger.error(f"Failed to load CSV {file_path}: {e}")
            raise

        return "\n".join(text_content)

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract CSV metadata."""
        metadata = {
            "file_path": file_path,
            "file_type": "csv",
            "file_size": os.path.getsize(file_path),
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


class DocumentLoader:
    """Main document loader that routes to appropriate loader based on file extension."""

    def __init__(self):
        self.loaders = {
            '.pdf': PDFLoader(),
            '.docx': DOCXLoader(),
            '.txt': TextLoader(),
            '.html': HTMLLoader(),
            '.htm': HTMLLoader(),
            '.md': MarkdownLoader(),
            '.markdown': MarkdownLoader(),
            '.csv': CSVLoader(),
            '.json': TextLoader(),  # JSON handled as text
        }

    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load a document and return its content and metadata.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with 'content', 'metadata', and 'file_path' keys
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get loader based on extension
        extension = file_path.suffix.lower()
        loader = self.loaders.get(extension)

        if not loader:
            raise ValueError(f"Unsupported file type: {extension}. Supported types: {list(self.loaders.keys())}")

        logger.info(f"Loading document: {file_path}")

        try:
            content = loader.load(str(file_path))
            metadata = loader.get_metadata(str(file_path))

            return {
                "content": content,
                "metadata": metadata,
                "file_path": str(file_path)
            }

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise

    def load_directory(self, directory_path: str, extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Load all documents from a directory.

        Args:
            directory_path: Path to directory containing documents
            extensions: List of extensions to filter (e.g., ['.pdf', '.txt'])

        Returns:
            List of document dictionaries
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        documents = []

        # Default to all supported extensions if not specified
        if extensions is None:
            extensions = list(self.loaders.keys())

        for extension in extensions:
            for file_path in directory_path.glob(f"**/*{extension}"):
                try:
                    doc = self.load_document(str(file_path))
                    documents.append(doc)
                    logger.info(f"Successfully loaded: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    continue

        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents


# Convenience function for quick loading
def load_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Quick helper function to load multiple documents.

    Args:
        file_paths: List of file paths to load

    Returns:
        List of document dictionaries
    """
    loader = DocumentLoader()
    documents = []

    for file_path in file_paths:
        try:
            doc = loader.load_document(file_path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")

    return documents


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    loader = DocumentLoader()

    # Load a single PDF
    # doc = loader.load_document("sample.pdf")
    # print(f"Content length: {len(doc['content'])} chars")
    # print(f"Metadata: {doc['metadata']}")

    # Load all PDFs from a directory
    # docs = loader.load_directory("./data/raw/", extensions=['.pdf'])
    # print(f"Loaded {len(docs)} documents")
    pass

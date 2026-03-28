"""
Tests for document loader module.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.ingestion.loader import (
    DocumentLoader, PDFLoader, DOCXLoader, TextLoader,
    HTMLLoader, CSVLoader, LoaderResult, LoaderError,
    FileNotFoundError, UnsupportedFormatError
)


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    def test_init(self):
        """Test loader initialization."""
        loader = DocumentLoader()
        assert loader is not None
        assert loader.timeout == 60
        assert loader.max_file_size_mb == 100
        assert len(loader.loaders) > 0

    def test_load_document_file_not_found(self, temp_dir):
        """Test loading non-existent file."""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_document(temp_dir / "nonexistent.txt")

    def test_load_document_unsupported_format(self, temp_dir):
        """Test loading unsupported file format."""
        file_path = temp_dir / "test.xyz"
        file_path.write_text("test content")

        loader = DocumentLoader()
        with pytest.raises(UnsupportedFormatError):
            loader.load_document(str(file_path))

    def test_load_document_text_file(self, temp_dir):
        """Test loading a text file."""
        file_path = temp_dir / "test.txt"
        content = "This is a test document.\nIt has multiple lines."
        file_path.write_text(content)

        loader = DocumentLoader()
        result = loader.load_document(str(file_path))

        assert "content" in result
        assert "metadata" in result
        assert result["content"] == content
        assert result["metadata"]["file_type"] == "txt"

    def test_load_document_html_file(self, temp_dir):
        """Test loading an HTML file."""
        file_path = temp_dir / "test.html"
        content = """
        <html>
            <head><title>Test Page</title></head>
            <body><h1>Test</h1><p>Test content</p></body>
        </html>
        """
        file_path.write_text(content)

        loader = DocumentLoader()
        result = loader.load_document(str(file_path))

        assert "content" in result
        assert "Title: Test Page" in result["content"]
        assert result["metadata"]["file_type"] == "html"

    def test_load_document_csv_file(self, temp_dir):
        """Test loading a CSV file."""
        file_path = temp_dir / "test.csv"
        content = "name,age\nAlice,30\nBob,25\n"
        file_path.write_text(content)

        loader = DocumentLoader()
        result = loader.load_document(str(file_path))

        assert "content" in result
        assert "Alice" in result["content"]
        assert result["metadata"]["file_type"] == "csv"

    def test_get_stats(self, temp_dir):
        """Test getting loader statistics."""
        loader = DocumentLoader()

        file_path = temp_dir / "test.txt"
        file_path.write_text("test")

        try:
            loader.load_document(str(file_path))
        except Exception:
            pass

        stats = loader.get_stats()
        assert "total_attempts" in stats
        assert "successful_loads" in stats
        assert "failed_loads" in stats


class TestPDFLoader:
    """Tests for PDFLoader."""

    @patch('PyPDF2.PdfReader')
    def test_load_pdf(self, mock_pdf_reader, temp_dir):
        """Test loading a PDF file."""
        # Create mock PDF
        mock_page = Mock()
        mock_page.extract_text.return_value = "PDF content"
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {"/Title": "Test PDF"}
        mock_pdf_reader.return_value = mock_pdf

        file_path = temp_dir / "test.pdf"
        file_path.touch()

        loader = PDFLoader()
        with patch('builtins.open', MagicMock()):
            result = loader.load(str(file_path))

        assert result.success
        assert "PDF content" in result.content
        assert result.metadata["file_type"] == "pdf"


class TestTextLoader:
    """Tests for TextLoader."""

    def test_load_text_file(self, temp_dir):
        """Test loading a text file."""
        file_path = temp_dir / "test.txt"
        content = "Test content\nwith multiple lines."
        file_path.write_text(content)

        loader = TextLoader()
        result = loader.load(str(file_path))

        assert result.success
        assert result.content == content
        assert result.metadata["file_type"] == "txt"

    def test_load_text_file_encoding(self, temp_dir):
        """Test loading text file with different encoding."""
        file_path = temp_dir / "test_utf8.txt"
        content = "Unicode content: café, résumé, ñ"
        file_path.write_text(content, encoding='utf-8')

        loader = TextLoader()
        result = loader.load(str(file_path))

        assert result.success
        assert "café" in result.content

    def test_load_empty_text_file(self, temp_dir):
        """Test loading empty text file."""
        file_path = temp_dir / "empty.txt"
        file_path.touch()

        loader = TextLoader()
        with pytest.raises(Exception):
            loader.load(str(file_path))


class TestCSVLoader:
    """Tests for CSVLoader."""

    def test_load_csv(self, temp_dir):
        """Test loading a CSV file."""
        file_path = temp_dir / "test.csv"
        content = "name,age,city\nAlice,30,NYC\nBob,25,LA\n"
        file_path.write_text(content)

        loader = CSVLoader()
        result = loader.load(str(file_path))

        assert result.success
        assert "Alice" in result.content
        assert "Bob" in result.content
        assert result.metadata["num_rows"] == 2

    def test_load_csv_with_different_delimiter(self, temp_dir):
        """Test loading CSV with semicolon delimiter."""
        file_path = temp_dir / "test_semicolon.csv"
        content = "name;age;city\nAlice;30;NYC\nBob;25;LA\n"
        file_path.write_text(content)

        loader = CSVLoader(auto_detect_delimiter=True)
        result = loader.load(str(file_path))

        assert result.success
        assert "Alice" in result.content
        assert result.metadata["delimiter"] == ";"

    def test_load_csv_max_rows(self, temp_dir):
        """Test CSV loading with row limit."""
        file_path = temp_dir / "test_max_rows.csv"
        content = "name,age\n" + "\n".join([f"person{i},{i}" for i in range(50)])
        file_path.write_text(content)

        loader = CSVLoader(max_rows=10)
        result = loader.load(str(file_path))

        assert result.success
        assert result.metadata["num_rows"] <= 10


class TestHTMLLoader:
    """Tests for HTMLLoader."""

    def test_load_html(self, temp_dir):
        """Test loading HTML file."""
        file_path = temp_dir / "test.html"
        content = """
        <html>
            <head><title>Test Title</title></head>
            <body>
                <h1>Header</h1>
                <p>Paragraph content.</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
        </html>
        """
        file_path.write_text(content)

        loader = HTMLLoader()
        result = loader.load(str(file_path))

        assert result.success
        assert "Test Title" in result.content
        assert "Header" in result.content
        assert "Paragraph content" in result.content

    def test_load_html_with_tables(self, temp_dir):
        """Test loading HTML with tables."""
        file_path = temp_dir / "test_table.html"
        content = """
        <html>
            <body>
                <table>
                    <tr><th>Name</th><th>Age</th></tr>
                    <tr><td>Alice</td><td>30</td></tr>
                    <tr><td>Bob</td><td>25</td></tr>
                </table>
            </body>
        </html>
        """
        file_path.write_text(content)

        loader = HTMLLoader(extract_tables=True)
        result = loader.load(str(file_path))

        assert result.success
        assert "Table" in result.content
        assert "Alice" in result.content
        assert result.metadata["num_tables"] == 1

    def test_load_html_metadata_extraction(self, temp_dir):
        """Test HTML metadata extraction."""
        file_path = temp_dir / "test_meta.html"
        content = """
        <html>
            <head>
                <title>Test Page</title>
                <meta name="description" content="Test description">
                <meta name="keywords" content="test,html,metadata">
                <meta property="og:title" content="OG Title">
            </head>
            <body>Content</body>
        </html>
        """
        file_path.write_text(content)

        loader = HTMLLoader(extract_metadata=True)
        result = loader.load(str(file_path))

        assert result.success
        assert result.metadata["title"] == "Test Page"
        assert result.metadata["description"] == "Test description"
        assert "open_graph" in result.metadata

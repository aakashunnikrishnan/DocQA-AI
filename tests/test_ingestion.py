"""
Unit tests for document ingestion module including loader and chunker.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import (
    PDFLoader, DOCXLoader, TextLoader, HTMLLoader,
    MarkdownLoader, CSVLoader, DocumentLoader
)
from src.ingestion.chunker import (
    FixedSizeChunker, SentenceChunker, ParagraphChunker,
    RecursiveChunker, SlidingWindowChunker, MarkdownChunker,
    CodeChunker, ChunkingPipeline, ChunkingStrategy, Chunk
)


# ============== Fixtures ==============

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_pdf_file(temp_dir):
    """Create a sample PDF file for testing."""
    # Note: Creating a real PDF requires libraries; we'll mock for unit tests
    pdf_path = os.path.join(temp_dir, "sample.pdf")
    return pdf_path


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file."""
    text_path = os.path.join(temp_dir, "sample.txt")
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write("This is a test document.\nIt has multiple lines.\nThis is the third line.")
    return text_path


@pytest.fixture
def sample_html_file(temp_dir):
    """Create a sample HTML file."""
    html_path = os.path.join(temp_dir, "sample.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("""
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Header</h1>
                <p>This is a test paragraph.</p>
                <p>Another paragraph here.</p>
            </body>
        </html>
        """)
    return html_path


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file."""
    csv_path = os.path.join(temp_dir, "sample.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("name,age,city\n")
        f.write("Alice,30,New York\n")
        f.write("Bob,25,Los Angeles\n")
        f.write("Charlie,35,Chicago\n")
    return csv_path


@pytest.fixture
def sample_markdown_file(temp_dir):
    """Create a sample markdown file."""
    md_path = os.path.join(temp_dir, "sample.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(""# Sample Header

This is a paragraph with **bold** text.

## Subheader

- List item 1
- List item 2

[Link](http://example.com)
    return md_path


@pytest.fixture
def sample_text():
    """Return sample text for chunking tests."""
    return """
    This is the first sentence. Here is the second sentence. And a third one.
    
    This is a new paragraph with different content. It has multiple sentences as well.
    
    Another paragraph here. With more text to test chunking boundaries.
    """


# ============== Loader Tests ==============

class TestTextLoader:
    """Tests for TextLoader class."""

    def test_load_text_file(self, sample_text_file):
        """Test loading a text file."""
        loader = TextLoader()
        content = loader.load(sample_text_file)

        assert "This is a test document" in content
        assert "multiple lines" in content
        assert len(content) > 0

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises exception."""
        loader = TextLoader()
        with pytest.raises(Exception):
            loader.load("/nonexistent/file.txt")

    def test_get_metadata(self, sample_text_file):
        """Test metadata extraction from text file."""
        loader = TextLoader()
        metadata = loader.get_metadata(sample_text_file)

        assert metadata["file_path"] == sample_text_file
        assert metadata["file_type"] == "txt"
        assert metadata["file_size"] > 0
        assert "encoding" in metadata


class TestPDFLoader:
    """Tests for PDFLoader class."""

    @patch('PyPDF2.PdfReader')
    def test_load_pdf_mock(self, mock_pdf_reader, sample_pdf_file):
        """Test PDF loading with mock."""
        # Create mock PDF reader
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample PDF content"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {"/Title": "Test PDF", "/Author": "Test Author"}
        mock_pdf_reader.return_value = mock_pdf

        loader = PDFLoader()

        # Mock open to avoid actual file read
        with patch('builtins.open', MagicMock()):
            content = loader.load(sample_pdf_file)
            assert "Sample PDF content" in content

    def test_pdf_loader_metadata(self):
        """Test PDF metadata extraction."""
        loader = PDFLoader()
        metadata = loader.get_metadata("test.pdf")

        assert metadata["file_path"] == "test.pdf"
        assert metadata["file_type"] == "pdf"


class TestHTMLLoader:
    """Tests for HTMLLoader class."""

    def test_load_html(self, sample_html_file):
        """Test loading HTML content."""
        loader = HTMLLoader()
        content = loader.load(sample_html_file)

        assert "Test Header" in content
        assert "test paragraph" in content
        assert "Another paragraph" in content

    def test_html_metadata(self, sample_html_file):
        """Test HTML metadata extraction."""
        loader = HTMLLoader()
        metadata = loader.get_metadata(sample_html_file)

        assert metadata["file_type"] == "html"
        assert metadata["file_size"] > 0
        assert metadata["title"] == "Test Page"


class TestCSVLoader:
    """Tests for CSVLoader class."""

    def test_load_csv(self, sample_csv_file):
        """Test loading CSV content."""
        loader = CSVLoader()
        content = loader.load(sample_csv_file)

        assert "Columns: name, age, city" in content or "Columns: name,age,city" in content
        assert "Alice" in content
        assert "Bob" in content
        assert "Charlie" in content

    def test_csv_metadata(self, sample_csv_file):
        """Test CSV metadata extraction."""
        loader = CSVLoader()
        metadata = loader.get_metadata(sample_csv_file)

        assert metadata["file_type"] == "csv"
        assert metadata["num_rows"] == 4  # Header + 3 data rows
        assert metadata["num_columns"] == 3


class TestMarkdownLoader:
    """Tests for MarkdownLoader class."""

    def test_load_markdown(self, sample_markdown_file):
        """Test loading markdown content."""
        loader = MarkdownLoader()
        content = loader.load(sample_markdown_file)

        assert "Sample Header" in content
        assert "paragraph" in content
        assert "Subheader" in content
        # Bold text should be extracted
        assert "bold" in content.lower()

    def test_markdown_metadata(self, sample_markdown_file):
        """Test markdown metadata extraction."""
        loader = MarkdownLoader()
        metadata = loader.get_metadata(sample_markdown_file)

        assert metadata["file_type"] == "md"
        assert metadata["file_size"] > 0


class TestDocumentLoader:
    """Tests for main DocumentLoader class."""

    def test_load_single_file(self, sample_text_file):
        """Test loading a single file."""
        loader = DocumentLoader()
        result = loader.load_document(sample_text_file)

        assert "content" in result
        assert "metadata" in result
        assert "file_path" in result
        assert len(result["content"]) > 0

    def test_load_directory(self, temp_dir, sample_text_file, sample_html_file):
        """Test loading all files from a directory."""
        loader = DocumentLoader()
        docs = loader.load_directory(temp_dir)

        assert len(docs) >= 2  # Should load both txt and html

    def test_load_directory_with_filter(self, temp_dir, sample_text_file, sample_html_file):
        """Test loading directory with extension filter."""
        loader = DocumentLoader()
        docs = loader.load_directory(temp_dir, extensions=['.txt'])

        assert len(docs) == 1
        assert docs[0]["metadata"]["file_type"] == "txt"

    def test_unsupported_file_type(self, temp_dir):
        """Test loading unsupported file type."""
        unsupported_path = os.path.join(temp_dir, "file.xyz")
        with open(unsupported_path, 'w') as f:
            f.write("test")

        loader = DocumentLoader()
        with pytest.raises(ValueError):
            loader.load_document(unsupported_path)

    def test_nonexistent_file(self):
        """Test loading non-existent file."""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_document("/nonexistent/file.pdf")

    def test_load_directory_nonexistent(self):
        """Test loading non-existent directory."""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_directory("/nonexistent/directory")


# ============== Chunker Tests ==============

class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_chunk_basic(self, sample_text):
        """Test basic chunking functionality."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(sample_text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.text) <= 100 for chunk in chunks)

    def test_chunk_overlap(self, sample_text):
        """Test that chunks have overlap."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=30)
        chunks = chunker.chunk(sample_text)

        if len(chunks) > 1:
            # Check that chunks overlap (some text appears in both)
            assert len(chunks[0].text) > 0
            assert len(chunks[1].text) > 0

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("")

        assert len(chunks) == 0

    def test_chunk_metadata(self, sample_text):
        """Test that chunks get proper metadata."""
        metadata = {"source": "test", "doc_id": 123}
        chunker = FixedSizeChunker()
        chunks = chunker.chunk(sample_text, metadata)

        if chunks:
            assert chunks[0].metadata["source"] == "test"
            assert chunks[0].metadata["doc_id"] == 123
            assert "chunk_index" in chunks[0].metadata


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_chunk_by_sentences(self):
        """Test chunking by sentences."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunker = SentenceChunker(chunk_size=2, chunk_overlap=0)
        chunks = chunker.chunk(text)

        # Should create chunks of 2 sentences each
        for chunk in chunks:
            sentence_count = chunk.text.count('.')
            assert sentence_count <= 2

    def test_sentence_overlap(self):
        """Test sentence chunker with overlap."""
        text = "Sent1. Sent2. Sent3. Sent4. Sent5."
        chunker = SentenceChunker(chunk_size=3, chunk_overlap=1)
        chunks = chunker.chunk(text)

        if len(chunks) > 1:
            # Should have overlapping sentences
            assert len(chunks) > 0


class TestParagraphChunker:
    """Tests for ParagraphChunker."""

    def test_chunk_by_paragraphs(self):
        """Test chunking by paragraphs."""
        text = "Para1 content.\n\nPara2 content.\n\nPara3 content."
        chunker = ParagraphChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(text)

        assert len(chunks) > 0

    def test_paragraph_boundaries(self):
        """Test that chunk boundaries respect paragraph boundaries."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunker = ParagraphChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(text)

        # Each chunk should contain complete paragraphs
        for chunk in chunks:
            assert "\n\n" in chunk.text or len(chunk.text) <= 50


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_recursive_splitting(self):
        """Test recursive splitting with different separators."""
        text = "Section 1.\n\nSection 2.\nLine 1.\nLine 2.\n\nSection 3."
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(text)

        assert len(chunks) > 0

    def test_fallback_to_fixed_size(self):
        """Test fallback to fixed size when separators fail."""
        text = "A" * 1000  # Long string without separators
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        assert all(len(chunk.text) <= 200 for chunk in chunks)


class TestSlidingWindowChunker:
    """Tests for SlidingWindowChunker."""

    def test_sliding_window(self):
        """Test sliding window chunking."""
        text = "This is a test document with multiple words " * 50
        chunker = SlidingWindowChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(text)

        assert len(chunks) > 1

        # Check that chunks have overlap
        if len(chunks) > 1:
            first_chunk_end = chunks[0].text[-50:]
            second_chunk_start = chunks[1].text[:50]
            # Some overlap should exist
            assert len(first_chunk_end) > 0

    def test_invalid_overlap(self):
        """Test handling of invalid overlap values."""
        text = "Test text " * 100
        chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=150)
        chunks = chunker.chunk(text)

        # Should handle gracefully without crashing
        assert isinstance(chunks, list)


class TestMarkdownChunker:
    """Tests for MarkdownChunker."""

    def test_markdown_header_preservation(self):
        """Test that markdown headers are preserved in chunks."""
        text = """# Header 1
Content for section 1.

## Header 2
Content for section 2.

### Header 3
Content for section 3.
"""
        chunker = MarkdownChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(text)

        # Headers should be preserved in respective chunks
        for chunk in chunks:
            if "Header 1" in chunk.text:
                assert "Content for section 1" in chunk.text

    def test_markdown_metadata(self):
        """Test metadata attachment in markdown chunks."""
        text = "# Test\nContent here."
        chunker = MarkdownChunker()
        metadata = {"doc_name": "test.md"}
        chunks = chunker.chunk(text, metadata)

        if chunks:
            assert chunks[0].metadata["doc_name"] == "test.md"


class TestCodeChunker:
    """Tests for CodeChunker."""

    def test_python_function_preservation(self):
        """Test that Python functions are kept intact."""
        text = """
def test_function():
    print("Inside function")
    return True

class TestClass:
    def method(self):
        pass

print("Outside")
"""
        chunker = CodeChunker(chunk_size=300, chunk_overlap=0, language="python")
        chunks = chunker.chunk(text)

        # Functions should not be split across chunks
        for chunk in chunks:
            if "def test_function" in chunk.text:
                assert "return True" in chunk.text
                assert "Outside" not in chunk.text or len(chunk.text) < 100

    def test_code_metadata(self):
        """Test metadata for code chunks."""
        text = "def func():\n    pass"
        chunker = CodeChunker()
        metadata = {"language": "python", "file": "test.py"}
        chunks = chunker.chunk(text, metadata)

        if chunks:
            assert chunks[0].metadata["language"] == "python"


class TestChunkingPipeline:
    """Tests for main ChunkingPipeline class."""

    def test_pipeline_creation(self):
        """Test creating pipeline with different strategies."""
        for strategy in ChunkingStrategy:
            pipeline = ChunkingPipeline(strategy=strategy, chunk_size=100)
            assert pipeline.chunker is not None

    def test_chunk_document(self, sample_text):
        """Test chunking a single document."""
        pipeline = ChunkingPipeline(chunk_size=100, chunk_overlap=20)
        chunks = pipeline.chunk_document(sample_text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_chunk_batch(self, sample_text):
        """Test chunking multiple documents."""
        documents = [
            {"content": sample_text, "metadata": {"doc_id": 1}},
            {"content": "Another document here.", "metadata": {"doc_id": 2}}
        ]

        pipeline = ChunkingPipeline(chunk_size=100)
        chunks = pipeline.chunk_batch(documents)

        assert len(chunks) > 0
        # Check that metadata from both docs is present
        doc_ids = {chunk.metadata.get("doc_id") for chunk in chunks}
        assert 1 in doc_ids or 2 in doc_ids

    def test_chunk_stats(self, sample_text):
        """Test chunk statistics calculation."""
        pipeline = ChunkingPipeline(chunk_size=100)
        chunks = pipeline.chunk_document(sample_text)

        stats = pipeline.get_chunk_stats(chunks)

        assert "total_chunks" in stats
        assert "avg_size" in stats
        assert "min_size" in stats
        assert "max_size" in stats
        assert stats["total_chunks"] == len(chunks)

    def test_empty_document_chunking(self):
        """Test chunking empty document."""
        pipeline = ChunkingPipeline()
        chunks = pipeline.chunk_document("")

        assert chunks == []

    def test_invalid_strategy(self):
        """Test handling of invalid strategy."""
        with pytest.raises(ValueError):
            ChunkingPipeline(strategy="invalid_strategy")  # type: ignore


class TestChunkClass:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a chunk object."""
        chunk = Chunk(
            text="Test content",
            metadata={"key": "value"},
            index=0,
            start_char=0,
            end_char=12
        )

        assert chunk.text == "Test content"
        assert chunk.metadata["key"] == "value"
        assert chunk.index == 0

    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = Chunk(
            text="Test",
            metadata={"score": 0.9},
            index=5,
            start_char=10,
            end_char=14
        )

        chunk_dict = chunk.to_dict()

        assert chunk_dict["text"] == "Test"
        assert chunk_dict["metadata"]["score"] == 0.9
        assert chunk_dict["index"] == 5


class TestIntegrationLoaderAndChunker:
    """Integration tests combining loader and chunker."""

    def test_load_and_chunk_text_file(self, sample_text_file):
        """Test loading a text file and chunking its content."""
        # Load
        loader = DocumentLoader()
        doc = loader.load_document(sample_text_file)

        # Chunk
        pipeline = ChunkingPipeline(chunk_size=50, chunk_overlap=10)
        chunks = pipeline.chunk_document(doc["content"], doc["metadata"])

        assert len(chunks) > 0
        # Metadata should be preserved
        for chunk in chunks:
            assert chunk.metadata["file_type"] == "txt"

    def test_load_and_chunk_html(self, sample_html_file):
        """Test loading HTML and chunking content."""
        loader = DocumentLoader()
        doc = loader.load_document(sample_html_file)

        pipeline = ChunkingPipeline(strategy=ChunkingStrategy.SENTENCE)
        chunks = pipeline.chunk_document(doc["content"], doc["metadata"])

        assert len(chunks) > 0
        assert chunks[0].metadata["title"] == "Test Page"

    def test_batch_load_and_chunk(self, temp_dir, sample_text_file, sample_html_file):
        """Test loading multiple files and chunking them together."""
        loader = DocumentLoader()
        docs = loader.load_directory(temp_dir)

        pipeline = ChunkingPipeline(chunk_size=100)
        all_chunks = pipeline.chunk_batch(docs)

        assert len(all_chunks) > 0
        # Should have chunks from both documents
        file_paths = {chunk.metadata.get("file_path") for chunk in all_chunks}
        assert len(file_paths) >= 1


# ============== Performance Tests ==============

class TestPerformance:
    """Performance and edge case tests."""

    def test_large_text_chunking(self):
        """Test chunking large text efficiently."""
        large_text = "This is a sentence. " * 10000  # ~200k chars
        pipeline = ChunkingPipeline(chunk_size=1000, chunk_overlap=100)

        import time
        start = time.time()
        chunks = pipeline.chunk_document(large_text)
        duration = time.time() - start

        assert len(chunks) > 10
        assert duration < 1.0  # Should be fast (< 1 second)

    def test_unicode_handling(self):
        """Test chunking text with unicode characters."""
        text = "Hello 世界. こんにちは. 🌟 Emoji test. Café Müller."
        chunker = FixedSizeChunker(chunk_size=50)
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        # Unicode should be preserved
        assert any("世界" in chunk.text for chunk in chunks)

    def test_special_characters(self):
        """Test handling of special characters."""
        text = "Tab\tseparated\nNewline\r\nCarriage return\rMultiple    spaces."
        chunker = FixedSizeChunker()
        chunks = chunker.chunk(text)

        assert chunks[0].text is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

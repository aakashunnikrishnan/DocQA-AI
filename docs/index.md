# Welcome to DocQA AI

**Production-ready RAG system for document question answering**

## What is DocQA AI?

DocQA AI is a comprehensive Retrieval-Augmented Generation (RAG) system that enables you to ask questions about your documents and get accurate, context-aware answers using state-of-the-art LLMs.

## Key Features

### 🔍 Hybrid Search
Combines vector similarity with BM25 keyword search for optimal retrieval performance.

### 🤖 Multiple LLM Support
Seamless integration with OpenAI GPT-4, Anthropic Claude, Google Gemini, and local models.

### 📚 Multi-Format Support
Process PDF, DOCX, HTML, Markdown, TXT, CSV, and JSON files.

### ⚡ High Performance
Async processing, multi-level caching, batch embeddings, and optimized vector search.

### 📊 Evaluation Suite
Comprehensive metrics for retrieval and generation quality.

### 🐳 Easy Deployment
Docker, Kubernetes, and cloud-ready configurations.

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/docqa-ai.git
cd docqa-ai
make install-dev

# Ingest documents
make ingest-sample

# Start the server
make run-dev

# Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'

# 📄 DocQA AI - Intelligent Document Question Answering System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A production-ready RAG (Retrieval-Augmented Generation) system that answers questions about your documents using state-of-the-art LLMs. Support for PDF, DOCX, HTML, TXT, and more.

## ✨ Features

- 🔍 **Hybrid Search** - Combines vector similarity with keyword search for optimal retrieval
- 🤖 **Multiple LLM Support** - OpenAI GPT-4, GPT-3.5, Llama 2, Mistral, and more
- 📚 **Multi-Format Support** - PDF, DOCX, HTML, Markdown, TXT, CSV, JSON
- ⚡ **High Performance** - Async processing, caching, batch embeddings
- 📊 **Evaluation Suite** - ROUGE, BLEU, faithfulness, relevance metrics
- 🐳 **Easy Deployment** - Docker, Kubernetes, AWS/Azure/GCP ready
- 🔐 **Enterprise Ready** - Authentication, rate limiting, audit logs
- 📈 **Monitoring** - OpenTelemetry, Prometheus metrics, Grafana dashboards

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI API key (or other LLM provider)
- 8GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/doc-qa-ai.git
cd doc-qa-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Ingest your documents
python scripts/ingest_documents.py --data-dir ./data/raw/

# Start the API server
python scripts/start_api.py

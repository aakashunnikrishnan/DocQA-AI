# 📄 DocQA AI - Document Question Answering System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI/CD Pipeline](https://github.com/yourusername/docqa-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/docqa-ai/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yourusername/docqa-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/docqa-ai)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Docker Pulls](https://img.shields.io/docker/pulls/yourusername/docqa-ai.svg)](https://hub.docker.com/r/yourusername/docqa-ai)

A production-ready RAG (Retrieval-Augmented Generation) system that answers questions about your documents using state-of-the-art LLMs. Support for PDF, DOCX, HTML, TXT, Markdown, CSV, and more.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/docqa-ai.git
cd docqa-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install-dev

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Ingest sample documents
make ingest-sample

# Start the API server
make run-dev

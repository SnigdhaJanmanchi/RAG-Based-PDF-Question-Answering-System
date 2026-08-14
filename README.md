# RAG Based PDF Question Answering System


> **An end-to-end Retrieval-Augmented Generation (RAG) application for interacting with PDF documents through natural language.**

AI PDF Document Assistant converts PDF documents into an intelligent, searchable knowledge base. The application extracts document content, generates semantic embeddings, indexes them in a FAISS vector database, retrieves the most relevant context, and leverages a Large Language Model (LLM) to generate accurate, context-aware responses.

Designed with a modular and production-oriented architecture, the project demonstrates the complete RAG pipeline while remaining scalable and easy to extend with additional document formats, embedding models, vector databases, or LLM providers.

---

## Core Concepts

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Embeddings
- Vector Databases (FAISS)
- Large Language Models (LLMs)
- Context-Aware Question Answering
- REST API Development
- Modular Backend Architecture

---

## Current Implementation

- PDF upload and validation
- Text extraction using PyMuPDF
- Recursive text chunking
- Semantic embedding generation with Sentence Transformers
- Vector indexing using FAISS
- Semantic similarity search
- Context retrieval for user queries
- FastAPI REST API with interactive Swagger documentation
- Modular architecture for parsing, embeddings, retrieval, vector storage, and API routing

---

## Technology Stack

| Category | Technologies |
|----------|--------------|
| **Backend** | Python, FastAPI, Uvicorn |
| **AI / Machine Learning** | Sentence Transformers, OpenAI API |
| **Vector Database** | FAISS |
| **Document Processing** | PyMuPDF, LangChain Text Splitters |
| **Frontend** | Streamlit *(In Progress)* |

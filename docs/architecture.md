# Architecture

This project is currently in the scaffolding and schema-design phase.

The intended architecture is a typed FastAPI backend with separate layers for:

- API routing;
- application configuration;
- document parsing interfaces;
- normalization schemas;
- persistence models;
- processing services.

Docling is expected to become the primary parser for PDFs. PyMuPDF and pdfplumber are included as fallback and diagnostic dependencies. DOCX support is reserved for a later phase through python-docx.

No semantic search, embeddings, RAG, chatbot flow or LLM extraction logic is implemented in this milestone.

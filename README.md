# 📚 RAG Document Chat

## Project Overview
RAG Document Chat is an AI-powered application that allows users to upload PDF documents and ask questions based on their content.

It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate answers using an AI language model.

## Features
- Upload PDF documents
- Extract text from PDF files
- Split documents into smaller chunks
- Generate text embeddings
- Store and retrieve information using ChromaDB
- Generate answers using FLAN-T5
- Display chat history

## Technologies Used
- Python
- Streamlit
- PyPDF
- Sentence Transformers
- ChromaDB
- Hugging Face Transformers
- FLAN-T5

## RAG Workflow
PDF Upload → Text Extraction → Chunking → Embeddings → Vector Database → Retrieval → LLM → Answer

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Objective
To build a document-based question-answering system using Retrieval-Augmented Generation.

## Author
Indhu

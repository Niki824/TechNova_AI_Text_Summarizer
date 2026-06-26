# TechNova RAG-Based Q&A System

## Overview

TechNova RAG-Based Q&A System is a Generative AI application that uses Retrieval-Augmented Generation (RAG) to answer user questions based on uploaded documents. The system retrieves relevant document sections using vector embeddings and generates answers using a language model.

## Features

* Upload PDF or TXT documents.
* Automatically split documents into chunks.
* Generate vector embeddings using Sentence Transformers.
* Store embeddings using FAISS vector database.
* Retrieve the most relevant document sections.
* Generate answers using a language model.
* Display retrieved context for transparency.

## Technologies Used

* Python
* Streamlit
* Sentence Transformers
* FAISS
* Hugging Face Transformers
* PyTorch
* PyPDF

## Project Workflow

1. Upload document.
2. Extract text from document.
3. Split text into chunks.
4. Generate embeddings.
5. Store embeddings in FAISS.
6. Retrieve relevant chunks based on the user's question.
7. Generate answer using retrieved context.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Example Questions

* What is RAG?
* What are vector embeddings?
* What are the applications of RAG systems?

## Future Improvements

* Support multiple document uploads.
* Add conversation history.
* Support web-based knowledge sources.
* Add ChromaDB support.
* Integrate Gemini API or OpenAI API for higher quality responses.

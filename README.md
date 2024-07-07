# Langchain RAG Project

This project demonstrates a Retrieval-Augmented Generation (RAG) system using Langchain. The system is designed to load documents from PDF files, split them into chunks, generate embeddings, store them in a vector store, and perform question answering using a large language model (LLM) from Hugging Face.

## Key Features

- Uses PyPDFLoader to load PDF documents
- Splits documents into manageable chunks with RecursiveCharacterTextSplitter
- Generates embeddings using jinaai/jina-embeddings-v2-base-es model
- Stores embeddings in a Chroma vector database
- Utilizes Mistral-7B-Instruct-v0.2 for question answering
- Supports Spanish language queries

## System Components

1. **Document Loading**: Reads PDF documents from a specified directory
2. **Text Splitting**: Splits documents into chunks for better processing
3. **Embedding Generation**: Uses Hugging Face embeddings for creating document embeddings
4. **Vector Storage**: Stores embeddings in a Chroma vector database
5. **Language Model**: Uses Mistral-7B-Instruct-v0.2 for generating answers to queries
6. **RetrievalQA Chain**: Combines retrieval and generation for effective question answering

## Setup

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/langchain-rag-project.git
cd langchain-rag-project

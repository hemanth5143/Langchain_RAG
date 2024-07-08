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
```

2. **Install the required packages**:

```bash
pip install langchain langchain_community huggingface_hub jinaai[jina-embeddings-v2] chromadb
```

3. **Place your PDF documents in the `data` directory**.

4. **Run the main script**:

```bash
python app.py
```

## Usage

1. **Load Documents**: The script loads PDF documents from the specified directory.
2. **Generate Embeddings and Store**: Creates embeddings for document chunks and stores them in a Chroma vector database.
3. **Ask Questions**: Use the `ask_question` function to get answers based on the documents.

Example usage:

```python
from app import ask_question

question = "LO QUE SE DIJO DE Israel EN LA Biblia - Traducción Reina Valera en Español"
print(ask_question(question))
```

## Configuration

- **PDF Directory**: Modify the `data` directory path if your documents are located elsewhere.
- **Vector Store Directory**: Change the `chroma_db` directory path if needed.
- **Model**: The current setup uses the `mistralai/Mistral-7B-Instruct-v0.2` model from Hugging Face.

## Authentication

Ensure you have a Hugging Face API token. You can set the token in your environment or directly in the script.

```python
from huggingface_hub import login

token = "your_huggingface_api_token"
login(token)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



## Acknowledgements

- [Langchain](https://github.com/hwchase17/langchain)
- [Hugging Face](https://huggingface.co)
- [Jina AI](https://jina.ai)
- [Chroma](https://www.trychroma.com)
```


##RAG System uisng Langchain
Here's a `README.md` file you can use for your Langchain RAG project on GitHub:

```markdown
# Langchain RAG Project

This project demonstrates a Retrieval-Augmented Generation (RAG) system using Langchain. The system is designed to load documents from PDF files, split them into chunks, generate embeddings, store them in a vector store, and perform question answering using a large language model (LLM) from Hugging Face.

## Project Structure

- **data/**: Directory containing PDF files to be processed.
- **chroma_db/**: Directory to store the Chroma vector database.
- **main.py**: Main script to run the project.
- **README.md**: This file.

## Prerequisites

- Python 3.8 or higher
- Install the required packages:

```bash
pip install langchain langchain_community huggingface_hub jinaai[jina-embeddings-v2] chromadb
```

## Setup

1. **Load Documents**: Load PDF documents from the `data` directory.
2. **Split Documents**: Split the documents into chunks for better processing.
3. **Generate Embeddings**: Use the `jinaai/jina-embeddings-v2-base-es` model to generate embeddings for the document chunks.
4. **Store in Vector Database**: Store the embeddings in a Chroma vector database.
5. **Question Answering**: Use a Hugging Face model to perform question answering based on the retrieved context.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/langchain-rag-project.git
cd langchain-rag-project
```

2. Place your PDF documents in the `data` directory.

3. Run the main script:

```bash
python main.py
```

4. Ask a question using the `ask_question` function.

```python
from main import ask_question

question = "LO QUE SE DIJO DE Israel EN LA Biblia - Traducción Reina Valera en Español"
print(ask_question(question))
```

## Example

```python
question = "LO QUE SE DIJO DE Israel EN LA Biblia - Traducción Reina Valera en Español"
result = ask_question(question)
print(result)
```

## Configuration

- **PDF Directory**: Modify the `data` directory path if your documents are located elsewhere.
- **Vector Store Directory**: Change the `chroma_db` directory path if needed.
- **Model**: The current setup uses the `mistralai/Mistral-7B-Instruct-v0.2` model from Hugging Face.

## Authentication

Ensure you have a Hugging Face API token. You can set the token in your environment or directly in the script.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Acknowledgements

- [Langchain](https://github.com/hwchase17/langchain)
- [Hugging Face](https://huggingface.co)
- [Jina AI](https://jina.ai)

```

Feel free to modify the content to better suit your project's specifics and needs.

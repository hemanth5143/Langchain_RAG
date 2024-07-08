from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="jinaai/jina-embeddings-v2-base-es",      
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)


# Define the path to your Chroma vector store directory
chroma_db_directory = r"C:\Users\hsai5\OneDrive\Documents\LLM projects\Langchain_RAG\chroma_db"

# Create the Chroma vector store
chroma_vector_store = Chroma(persist_directory=chroma_db_directory, embedding_function=huggingface_embeddings)
chroma_retriever = chroma_vector_store.as_retriever()

prompt_template = """
Eres un asistente de preguntas y respuestas. Tu objetivo es responder preguntas
con la mayor precisión posible según las instrucciones y el contexto proporcionados.

Contexto: {context}

Pregunta: {question}

Respuesta:
"""

from huggingface_hub import login

token = "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"
login(token)

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceHub(
    repo_id=repo_id,
    huggingfacehub_api_token=token,
    model_kwargs={"temperature": 0.5, "max_length": 2048}
)

# Create a document compressor
compressor = LLMChainExtractor.from_llm(llm)

# Create the contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=chroma_retriever  # Use Chroma retriever here
)

# Use the compression retriever in your QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Example usage
question = "LO QUE SE DIJO DE Israel EN LA Biblia - Traducción Reina Valera en Español"
result = qa_chain({"query": question})
print(result['result'])

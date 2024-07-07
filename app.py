from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma


loader = PyPDFDirectoryLoader(r"C:\Users\hsai5\OneDrive\Documents\LLM projects\Langchain_RAG\data")
documents = loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_documents=text_splitter.split_documents(documents)

huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="jinaai/jina-embeddings-v2-base-es",      
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)


vectorstore = Chroma.from_documents(
    documents=final_documents,
    embedding=huggingface_embeddings,
    persist_directory=r"C:\Users\hsai5\OneDrive\Documents\LLM projects\Langchain_RAG\chroma_db"
)

vectorstore.persist()

from huggingface_hub import login

token = "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"
login(token)

prompt_template = """
Eres un asistente de preguntas y respuestas. Tu objetivo es responder preguntas
con la mayor precisión posible según las instrucciones y el contexto proporcionados.

Contexto: {context}

Pregunta: {question}

Respuesta:
"""

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

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Function to ask questions
def ask_question(question):
    result = qa_chain({"query": question})
    return result['result']

# Example usage
question = "LO QUE SE DIJO DE Israel EN LA Biblia - Traducción Reina Valera en Español"
print(ask_question(question))
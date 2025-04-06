from fastapi import APIRouter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob

from pydantic import BaseModel


# Create a router instance to define API endpoints for this module
router = APIRouter()

# Define the input data model for the POST request
class UserQuery(BaseModel):
    prompt: str

# Define the output data model for the response
class BotResponse(BaseModel):
    response: str

# Define the path to the folder containing the source `.txt` documents
folder_path = os.path.expanduser("~/Desktop/AI_Text_Folder")

# Function to extract the full text content from a given .txt file
def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# List to store documents in LangChain's Document format
documents = []

# Load and wrap each .txt file as a Document with metadata (filename)
for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
    text = extract_text_from_txt(file_path)
    documents.append(Document(page_content=text, metadata={"filename": os.path.basename(file_path)}))

# Create a text splitter to divide documents into overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Split the full documents into smaller chunks for embedding
split_docs = text_splitter.split_documents(documents)

# Create an embedding model using a local Ollama model
embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

# Generate vector embeddings for the split documents and store them in FAISS
vector_store = FAISS.from_documents(split_docs, embedding_model)

# Load the language model used for generating responses (also via Ollama)
llm = Ollama(model="gemma2:9b")

# Create a RetrievalQA chain:
# - Retrieves relevant document chunks from FAISS
# - Sends them along with the prompt to the LLM for a contextual answer
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_store.as_retriever())

# Wrapper function to run the QA chain and return the generated response
def response_func(prompt: str) -> str:
    response = qa_chain.run(prompt)
    return response.content if hasattr(response, "content") else str(response)

# Define a POST API endpoint to accept user prompts and return LLM responses
@router.post("/Prompt", response_model=BotResponse)
async def prompt_endpoint(user: UserQuery):
    response = response_func(user.prompt)
    return {"response": response}

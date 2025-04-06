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

router = APIRouter()

class UserQuery(BaseModel):
    prompt: str

class BotResponse(BaseModel):
    response: str

#Folder path with the used txt documents
folder_path = os.path.expanduser("~/Desktop/AI_Text_Folder")

#Function to get text from a file
def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

documents = []

# Iterate through every file in our folder and extract the test for each text file
for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
    text = extract_text_from_txt(file_path)

    # Adds file content to the documents list
    documents.append(Document(page_content=text, metadata={"filename": os.path.basename(file_path)}))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


split_docs = text_splitter.split_documents(documents)

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

vector_store = FAISS.from_documents(split_docs, embedding_model)

llm = Ollama(model="gemma2:9b")

qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_store.as_retriever())

def response_func(prompt: str) -> str:
    response = qa_chain.run(prompt)
    return response.content if hasattr(response, "content") else str(response)

#Post endpoint
@router.post("/Prompt", response_model=BotResponse)
async def prompt_endpoint (user: UserQuery):
    response = response_func(user.prompt)
    return {"response": response}

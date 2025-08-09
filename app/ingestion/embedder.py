from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

def create_chunks(text: str, chunk_size=800, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def upsert_documents(docs: list, metadatas: list, collection_name="default"):
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vect = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb, collection_name=collection_name)
    vect.add_texts(texts=docs, metadatas=metadatas)
    vect.persist()
    return vect

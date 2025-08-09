from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.ingestion.embedder import CHROMA_DIR
from langchain_community.vectorstores import Chroma

def get_retriever(collection_name="default"):
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb, collection_name=collection_name)
    return db.as_retriever(search_kwargs={"k": 4})

def answer_question(question: str, collection_name: str = "default"):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro",temperature=0.1)
    retriever = get_retriever(collection_name=collection_name)
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    return qa.run(question)

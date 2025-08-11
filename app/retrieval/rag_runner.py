from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.ingestion.embedder import CHROMA_DIR
from langchain_community.vectorstores import Chroma
import re


def _strip_markdown(md: str) -> str:
    if not md:
        return md
    # Remove fenced code blocks and inline code
    text = re.sub(r"```.*?```", "", md, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Convert links/images to plain text
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Strip headings, lists, blockquotes
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.M)
    # Bold/italic markers
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove markdown tables
    text = re.sub(r"^\s*\|.*\|\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*[-:]{2,}\s*(\|\s*[-:]{2,}\s*)+$", "", text, flags=re.M)

    # Normalize to simple paragraphs
    lines = [ln.strip() for ln in text.splitlines()]
    paras = []
    cur = []
    for ln in lines:
        if ln:
            cur.append(ln)
        else:
            if cur:
                paras.append(" ".join(cur))
                cur = []
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras).strip()


def get_retriever(collection_name="default"):
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb, collection_name=collection_name)
    return db.as_retriever(search_kwargs={"k": 4})


def answer_question(question: str, collection_name: str = "default"):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)
    retriever = get_retriever(collection_name=collection_name)
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    raw = qa.run(question)
    return _strip_markdown(raw)

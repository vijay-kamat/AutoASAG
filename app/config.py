import os
from dotenv import load_dotenv

load_dotenv()

# ==== API KEYS ====
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# ==== PATHS ====

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/embeddings/chroma_db")

MODEL_DIR = os.getenv("MODEL_DIR", "./app/models")

MOHLER_DATASET_PATH = os.getenv("MOHLER_DATASET_PATH", "./data/mohler/")

# ==== RAG CONFIG ====

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 4))

# ==== GRADING CONFIG ====
BERTSCORE_MODEL = os.getenv("BERTSCORE_MODEL", "roberta-large")
NLI_MODEL = os.getenv("NLI_MODEL", "roberta-large-mnli")

# ==== STREAMLIT CONFIG ====
STREAMLIT_SECRET_KEY = os.getenv("STREAMLIT_SECRET_KEY", "change_me")

# ==== TRAINING CONFIG ====
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 300))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", 4))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.05))

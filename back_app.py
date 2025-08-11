# back_app.py (CLI backend tester)
import sys
import os
import argparse
import json
import logging
from typing import List
import pandas as pd

# ensure project package imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# app modules
from app import config
from app.scrapers.web_scraper import fetch_text_from_url
from app.ingestion.embedder import create_chunks, upsert_documents, CHROMA_DIR
from app.retrieval.rag_runner import answer_question
from app.grading.semantic_scoring import compute_bert_scores, compute_nli_probs
import joblib
import torch

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backend_tester")

# runtime device for torch/bert-score
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")


def process_url_to_chroma(url: str, collection_name: str = "test_collection"):
    """
    Scrape text from url, chunk it, upsert to Chroma collection.
    Returns number of chunks created.
    """
    logger.info(f"Scraping URL: {url}")
    text = fetch_text_from_url(url)
    if not text or len(text.strip()) < 50:
        raise ValueError("Scraped text is empty or too short to create meaningful chunks.")
    logger.info(f"Scraped {len(text)} characters.")

    # chunk
    chunks = create_chunks(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    logger.info(f"Created {len(chunks)} chunks (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    # metadata for tracing
    metadatas = [{"source": url, "chunk_id": i} for i in range(len(chunks))]

    # upsert into Chroma
    vect = upsert_documents(chunks, metadatas, collection_name=collection_name)
    # Log the actual persist directory used by embedder
    logger.info(f"Upserted {len(chunks)} docs into Chroma at {CHROMA_DIR}/{collection_name}")
    return len(chunks)


def generate_rag_answer(question: str, collection_name: str = "test_collection"):
    """
    Use your existing RAG pipeline to answer the question using the chosen collection.
    Returns the generated answer string.
    """
    logger.info(f"Generating RAG answer for question: {question}")
    answer = answer_question(question, collection_name=collection_name)
    logger.info("RAG generation done.")
    return answer


def build_features_and_predict(student_answer: str, reference_answer: str, model_path: str):
    """
    Compute BERTScore (P, R, F1) and NLI probs (contradiction, neutral, entailment),
    build a feature vector matching the trained model, and predict using a pre-trained XGBoost model (joblib).
    Returns dict with features and predicted grade.
    """
    # 1) BERTScore - accepts lists
    logger.info("Computing BERTScore...")
    P_list, R_list, F1_list = compute_bert_scores([student_answer], [reference_answer], lang="en")
    p, r, f1 = P_list[0], R_list[0], F1_list[0]

    logger.info("Computing NLI probabilities...")
    # premise=reference, hypothesis=student
    c_prob, n_prob, e_prob = compute_nli_probs(reference_answer, student_answer)

    # Build features exactly as the trained model expects (6 features, in this order)
    feature_names = [
        "bert_precision", "bert_recall", "bert_f1",
        "nli_contradiction", "nli_neutral", "nli_entailment",
    ]
    feature_vector = [
        p, r, f1,
        c_prob, n_prob, e_prob,
    ]

    # Prepare DataFrame with the same feature names as training to satisfy XGBoost validation
    X_infer = pd.DataFrame([feature_vector], columns=feature_names)

    # load model
    logger.info(f"Loading grading model from {model_path}")
    clf = joblib.load(model_path)

    # Predict
    pred = clf.predict(X_infer)[0]

    result = {
        "features": dict(zip(feature_names, feature_vector)),
        "predicted_grade": round(float(pred))
    }
    return result


def save_run_output(outpath: str, payload: dict):
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved run output to {outpath}")


def parse_args():
    parser = argparse.ArgumentParser(description="Backend tester: scrape -> embed -> RAG -> grade.")
    parser.add_argument("--url", help="URL to scrape", required=True)
    parser.add_argument("--question", help="Question to ask RAG", required=True)
    parser.add_argument("--student_answer", help="Student answer to grade", required=True)
    parser.add_argument("--collection", help="Chroma collection name", default="test_collection")
    parser.add_argument("--model", help="Path to trained grading model (joblib)", default=os.path.join(config.MODEL_DIR, "xgb_grade_predictor.joblib"))
    parser.add_argument("--out", help="Path to save JSON output", default="last_run_output.json")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Process url -> Chroma
    try:
        num_chunks = process_url_to_chroma(args.url, collection_name=args.collection)
    except Exception:
        logger.exception("Failed during scraping/upsert step.")
        return

    # 2. Generate RAG answer
    try:
        rag_answer = generate_rag_answer(args.question, collection_name=args.collection)
    except Exception:
        logger.exception("RAG generation failed.")
        return

    # 3. Grade the student answer by comparing to rag_answer
    try:
        result = build_features_and_predict(args.student_answer, rag_answer, model_path=args.model)
    except Exception:
        logger.exception("Grading step failed.")
        return

    # 4. Consolidate and save
    out = {
        "url": args.url,
        "question": args.question,
        "num_chunks": num_chunks,
        "rag_answer": rag_answer,
        "student_answer": args.student_answer,
        "grading": result
    }

    save_run_output(args.out, out)
    logger.info("Run completed successfully.")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

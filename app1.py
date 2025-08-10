import pandas as pd
import os
from app.grading.semantic_scoring import compute_bert_scores, compute_nli_probs
from app.trainer.train_model import train

# Read input dataset (kept as-is)
df = pd.read_csv("mohler_dataset_filtered.csv")

P, R, F1 = compute_bert_scores(df["student_answer"].tolist(), df["desired_answer"].tolist())
df["bert_precision"] = P
df["bert_recall"] = R
df["bert_f1"] = F1

contradictions, neutrals, entailments = [], [], []
for student, desired in zip(df["student_answer"], df["desired_answer"]):
    c_prob, n_prob, e_prob = compute_nli_probs(desired, student)
    contradictions.append(c_prob)
    neutrals.append(n_prob)
    entailments.append(e_prob)

df["nli_contradiction"] = contradictions
df["nli_neutral"] = neutrals
df["nli_entailment"] = entailments

feature_cols = ["bert_precision", "bert_recall", "bert_f1", "nli_contradiction", "nli_neutral", "nli_entailment"]
train_df = df[feature_cols + ["score_avg"]]

features_dir = os.path.join("app", "features")
os.makedirs(features_dir, exist_ok=True)
out_path = os.path.join(features_dir, "mohler_dataset_with_features.csv")
train_df.to_csv(out_path, index=False)
print("Saved feature enriched dataset.")

feature_cols = ["bert_precision", "bert_recall", "bert_f1", "nli_contradiction", "nli_neutral", "nli_entailment"]
model = train(df[feature_cols + ["score_avg"]])
print("Model trained and saved.")

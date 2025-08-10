from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load NLI model once and move to device
NLI_MODEL = "roberta-large-mnli"
tokenizer_nli = AutoTokenizer.from_pretrained(NLI_MODEL)
model_nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()

def compute_bert_scores(cands, refs, lang="en"):

    P, R, F1 = bert_score(cands, refs, lang=lang, model_type="roberta-large", device=str(device), rescale_with_baseline=True)
    return P.tolist(), R.tolist(), F1.tolist()

def compute_nli_probs(premise: str, hypothesis: str):
    inputs = tokenizer_nli(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model_nli(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    contradiction_prob = float(probs[0])
    neutral_prob = float(probs[1])
    entailment_prob = float(probs[2])
    return contradiction_prob, neutral_prob, entailment_prob

import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.pkl")

tokenizer = None
indobert_model = None
svm_model = None


def load_indobert():
    global tokenizer, indobert_model
    if tokenizer is None or indobert_model is None:
        tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        indobert_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
        indobert_model.eval()
    return tokenizer, indobert_model


def load_svm():
    global svm_model
    if svm_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model SVM tidak ditemukan di {MODEL_PATH}")
        svm_model = joblib.load(MODEL_PATH)
    return svm_model


try:
    load_indobert()
except Exception as e:
    print(f"Initial load IndoBERT deferred: {e}")

try:
    load_svm()
except Exception as e:
    print(f"Initial load SVM deferred: {e}")


def get_indobert_embedding(text: str) -> np.ndarray:
    """
    Mean Pooling -- SAMA dengan get_embeddings() saat training
    (bukan lagi CLS token).
    """
    tok, model = load_indobert()
    inputs = tok(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    mask = (
        inputs["attention_mask"]
        .unsqueeze(-1)
        .expand(outputs.last_hidden_state.size())
        .float()
    )
    sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = (sum_embeddings / sum_mask).detach().numpy()

    return mean_pooled


def predict_hoax(text: str):
    model_svm = load_svm()
    embedding = get_indobert_embedding(text)
    prediction = model_svm.predict(embedding)

    if hasattr(model_svm, "predict_proba"):
        proba = model_svm.predict_proba(embedding)[0]
        confidence = (
            float(proba[1] * 100) if prediction[0] == 1 else float(proba[0] * 100)
        )
    else:
        confidence = 100.0

    return ("FAKTA", confidence) if prediction[0] == 1 else ("HOAKS", confidence)

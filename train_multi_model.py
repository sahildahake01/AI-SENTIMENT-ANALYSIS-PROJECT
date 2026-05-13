# train_multi_model.py
# ============================================================
# MULTI-MODEL TRAINING EXTENSION
# ============================================================
# Run AFTER your existing sentiment_pipeline.py, OR import
# train_extra_models() from it and call it at the end of
# run_pipeline() in sentiment_pipeline.py.
#
# What this script does:
#   1. Loads the existing TF-IDF vectorizer + label-encoder
#      saved by sentiment_pipeline.py
#   2. Loads sentiment_results.csv (already produced)
#   3. Re-vectorizes the cleaned text
#   4. Trains Naive Bayes  → model_nb.pkl
#   5. Trains MLP/NN       → model_nn.pkl
#   6. Appends new accuracy keys into metrics.json so the
#      analytics dashboard shows all three scores
#
# The existing model.pkl / vectorizer.pkl / label_encoder.pkl
# are NEVER touched.
# ============================================================

import json
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Shared config (must match sentiment_pipeline.py) ──────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
RESULTS_CSV    = "sentiment_results.csv"
METRICS_JSON   = "metrics.json"

STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your",
    "yours","yourself","yourselves","he","him","his","himself","she",
    "her","hers","herself","it","its","itself","they","them","their",
    "theirs","themselves","what","which","who","whom","this","that",
    "these","those","am","is","are","was","were","be","been","being",
    "have","has","had","having","do","does","did","doing","a","an",
    "the","and","but","if","or","because","as","until","while","of",
    "at","by","for","with","about","against","between","into","through",
    "during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then",
    "once","here","there","when","where","why","how","all","both",
    "each","few","more","most","other","some","such","no","nor",
    "not","only","own","same","so","than","too","very","s","t",
    "can","will","just","don","should","now","d","ll","m","o",
    "re","ve","y","ain","aren","couldn","didn","doesn","hadn",
    "hasn","haven","isn","ma","mightn","mustn","needn","shan",
    "shouldn","wasn","weren","won","wouldn",
}

def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(t for t in text.split() if t not in STOPWORDS and len(t) > 1)


# ── Main training function ─────────────────────────────────────────────
def train_extra_models():
    print("\n" + "="*60)
    print("MULTI-MODEL EXTENSION TRAINING")
    print("="*60)

    # 1. Load shared artefacts produced by sentiment_pipeline.py
    print("\n[Load] Loading existing vectorizer and label encoder …")
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    le         = pickle.load(open("label_encoder.pkl", "rb"))

    # 2. Load training data
    print(f"[Load] Reading '{RESULTS_CSV}' …")
    df = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=["review_text", "actual_sentiment"])

    # Clean text (same preprocessing as pipeline)
    df["cleaned"] = df["review_text"].apply(_clean)
    df = df[df["cleaned"].str.strip() != ""]

    print(f"  Usable rows: {len(df):,}")
    print(f"  Class distribution:\n{df['actual_sentiment'].value_counts().to_string()}")

    # 3. Vectorize & encode
    X = vectorizer.transform(df["cleaned"])
    y = le.transform(df["actual_sentiment"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # ── 4. Naive Bayes ─────────────────────────────────────────────────
    print("\n--- Naive Bayes ---")
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_acc  = accuracy_score(y_test, nb_pred)
    print(f"  Accuracy : {nb_acc:.4f}")
    print(classification_report(y_test, nb_pred, target_names=le.classes_, zero_division=0))

    pickle.dump(nb, open("model_nb.pkl", "wb"))
    print("  Saved → model_nb.pkl ✅")

    # ── 5. Neural Network (MLP) ────────────────────────────────────────
    print("\n--- Neural Network (MLPClassifier) ---")
    nn = MLPClassifier(
        hidden_layer_sizes=(256, 128),   # two hidden layers
        activation="relu",
        solver="adam",
        alpha=1e-4,                      # L2 regularisation
        learning_rate_init=0.001,
        max_iter=30,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    nn.fit(X_train, y_train)
    nn_pred = nn.predict(X_test)
    nn_acc  = accuracy_score(y_test, nn_pred)
    print(f"  Accuracy : {nn_acc:.4f}")
    print(classification_report(y_test, nn_pred, target_names=le.classes_, zero_division=0))

    pickle.dump(nn, open("model_nn.pkl", "wb"))
    print("  Saved → model_nn.pkl ✅")

    # ── 6. Patch metrics.json ──────────────────────────────────────────
    print("\n[Metrics] Updating metrics.json …")
    try:
        with open(METRICS_JSON) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}

    metrics["naive_bayes_accuracy"]    = round(nb_acc, 4)
    metrics["neural_network_accuracy"] = round(nn_acc, 4)

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print("  Updated metrics.json ✅")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"  Naive Bayes  accuracy: {nb_acc:.4f}")
    print(f"  Neural Net   accuracy: {nn_acc:.4f}")
    print("="*60 + "\n")

    return {"naive_bayes": nb_acc, "neural_network": nn_acc}


if __name__ == "__main__":
    train_extra_models()

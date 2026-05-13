import pickle
import numpy as np
from collections import Counter

import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text

class MultiModelEngine:
    """
    Wraps three trained models (LR, NB, MLP/NN) and a shared TF-IDF
    vectorizer. Exposes a single .predict_all(text) method that returns
    per-model labels + confidences plus an ensemble final prediction.

    All models are loaded from .pkl files saved by the training pipeline.
    The existing LR model/vectorizer/label-encoder are reused unchanged.
    """

    # ── File paths (relative to project root) ──────────────────────────
    MODEL_LR  = "model.pkl"
    MODEL_NB  = "model_nb.pkl"
    MODEL_NN  = "model_nn.pkl"
    VEC_PATH  = "vectorizer.pkl"
    LE_PATH   = "label_encoder.pkl"

    def __init__(self):
        # Load shared assets
        self.vectorizer = pickle.load(open(self.VEC_PATH, "rb"))
        self.le         = pickle.load(open(self.LE_PATH,  "rb"))

        # Load individual models
        self.lr_model = pickle.load(open(self.MODEL_LR, "rb"))
        self.nb_model = pickle.load(open(self.MODEL_NB, "rb"))
        self.nn_model = pickle.load(open(self.MODEL_NN, "rb"))

    # ── Core prediction helper ─────────────────────────────────────────
    def _predict_single(self, model, X_vec) -> dict:
        """
        Run one model on an already-vectorized sample.
        Returns {"label": str, "confidence": float}.
        """
        proba      = model.predict_proba(X_vec)[0]          # shape (n_classes,)
        pred_class = np.argmax(proba)
        confidence = round(float(proba[pred_class]), 4)
        label      = self.le.inverse_transform([pred_class])[0]
        return {"label": label, "confidence": confidence}

    # ── Ensemble logic ─────────────────────────────────────────────────
    @staticmethod
    def _ensemble(lr_result: dict, nb_result: dict, nn_result: dict) -> str:
        """
        Majority voting across the three models.
        On a tie (all three different — impossible with 3 binary choices
        but possible with 3 classes each differing), pick the prediction
        that has the highest confidence score.
        """
        labels = [lr_result["label"], nb_result["label"], nn_result["label"]]
        counts = Counter(labels)
        top_count = counts.most_common(1)[0][1]

        # Clear majority
        if top_count >= 2:
            return counts.most_common(1)[0][0]

        # Three-way tie: pick the model with highest confidence
        candidates = [lr_result, nb_result, nn_result]
        return max(candidates, key=lambda m: m["confidence"])["label"]


    # ── Public API ─────────────────────────────────────────────────────
    def predict_all(self, text: str) -> dict:
        """
        Run all three models on raw text.

        Returns:
        {
            "lr":               {"label": "Positive", "confidence": 0.89},
            "nb":               {"label": "Neutral",  "confidence": 0.72},
            "nn":               {"label": "Positive", "confidence": 0.93},
            "final_prediction": "Positive"
        }
        """
        cleaned = clean_text(text)
        X_vec = self.vectorizer.transform([cleaned])    # sparse matrix

        lr_result = self._predict_single(self.lr_model, X_vec)
        nb_result = self._predict_single(self.nb_model, X_vec)
        nn_result = self._predict_single(self.nn_model, X_vec)

        final = self._ensemble(lr_result, nb_result, nn_result)

        return {
            "lr": lr_result,
            "nb": nb_result,
            "nn": nn_result,
            "final_prediction": final,
        }

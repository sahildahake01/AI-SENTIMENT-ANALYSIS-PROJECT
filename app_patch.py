# app_patch.py
# ============================================================
# FLASK PATCH — NEW /predict_all_models ENDPOINT
# ============================================================
# INTEGRATION STEPS (3 lines in your existing app.py):
#
#   engine = None  # declared at module level
#   register_multi_model_routes(app)
#
# That's it. All existing routes remain 100% unchanged.
# ============================================================

from flask import request, jsonify
from multi_model_engine import MultiModelEngine

# Lazy-loaded singleton — only initialises once on first request
_engine_instance = None


def _get_engine() -> MultiModelEngine:
    """Return a cached MultiModelEngine, initialising on first call."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = MultiModelEngine()
    return _engine_instance


def register_multi_model_routes(app):
    """
    Call this function once after creating your Flask `app` object.
    It attaches the new endpoint without modifying any existing route.

    Usage in app.py:
        from app_patch import register_multi_model_routes
        register_multi_model_routes(app)
    """

    # ── NEW endpoint: /predict_all_models ──────────────────────────────
    @app.route("/predict_all_models", methods=["POST"])
    def predict_all_models():
        """
        POST  /predict_all_models
        Body: { "text": "<review string>" }

        Response:
        {
            "lr":               {"label": "Positive", "confidence": 0.89},
            "nb":               {"label": "Neutral",  "confidence": 0.72},
            "nn":               {"label": "Positive", "confidence": 0.93},
            "final_prediction": "Positive"
        }
        """
        data = request.get_json(silent=True)

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in JSON body."}), 400

        text = str(data["text"]).strip()
        if not text:
            return jsonify({"error": "'text' field is empty."}), 400

        try:
            engine = _get_engine()
            result = engine.predict_all(text)
            return jsonify(result), 200

        except FileNotFoundError as exc:
            return jsonify({
                "error": f"Model file not found: {exc}. "
                         "Run the training pipeline first."
            }), 503

        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ── Health-check for the new engine ────────────────────────────────
    @app.route("/multi_model_status", methods=["GET"])
    def multi_model_status():
        """Quick liveness check — returns which models are loaded."""
        try:
            engine = _get_engine()
            return jsonify({
                "status":  "ok",
                "models":  ["logistic_regression", "naive_bayes", "neural_network"],
                "classes": engine.le.classes_.tolist(),
            }), 200
        except Exception as exc:
            return jsonify({"status": "error", "detail": str(exc)}), 503

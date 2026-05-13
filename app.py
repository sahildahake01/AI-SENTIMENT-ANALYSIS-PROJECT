import os
import csv
import json
import datetime
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import confusion_matrix
from app_patch import register_multi_model_routes

app = Flask(__name__)

CSV_FILE = 'sentiment_results.csv'
METRICS_FILE = 'metrics.json'

# ---------- LOAD TRAINED MODEL ----------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# ---------- PREDICTION ----------
def predict_sentiment(text):
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    pred_class = model.predict(X)[0]
    confidence = round(max(proba) * 100, 2)
    sentiment = le.inverse_transform([pred_class])[0]
    return sentiment, confidence

# ---------- CSV HANDLING ----------
def ensure_csv():
    if not os.path.exists(CSV_FILE):
        import random

        sentiments = ["Positive", "Negative", "Neutral"]

        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["review", "actual_sentiment", "predicted_lr"])

            for _ in range(30000):
                actual = random.choice(sentiments)
                predicted = random.choice(sentiments)
                writer.writerow(["sample review", actual, predicted])

def append_prediction(review, sentiment):
    ensure_csv()
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([review, "", sentiment])  # actual blank (user input)

# ---------- DASHBOARD STATS ----------
def get_dashboard_stats():
    ensure_csv()
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    df = pd.read_csv(CSV_FILE)

    if 'predicted_lr' in df.columns:
        for val in df['predicted_lr'].dropna():
            if val in counts:
                counts[val] += 1

    total = sum(counts.values())
    most = max(counts, key=counts.get) if total > 0 else "N/A"

    return {
        "positive": counts["Positive"],
        "negative": counts["Negative"],
        "neutral": counts["Neutral"],
        "total": total,
        "most_frequent": most
    }

# ---------- METRICS (REAL DATA) ----------
def generate_metrics():
    if not os.path.exists(CSV_FILE):
        return None

    df = pd.read_csv(CSV_FILE)

    if 'actual_sentiment' not in df.columns or 'predicted_lr' not in df.columns:
        return None

    df = df.dropna(subset=['actual_sentiment', 'predicted_lr'])

    if len(df) == 0:
        return None

    labels = ["Negative", "Neutral", "Positive"]

    cm = confusion_matrix(df['actual_sentiment'], df['predicted_lr'], labels=labels)

    metrics = {
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "logistic_regression_accuracy": round((df['actual_sentiment'] == df['predicted_lr']).mean(), 4)
    }

    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_metrics():
    if not os.path.exists(METRICS_FILE):
        return generate_metrics()
    with open(METRICS_FILE) as f:
        return json.load(f)

# ---------- ROUTES ----------
@app.route('/')
def dashboard():
    stats = get_dashboard_stats()

    chart_data = {
        "labels": ["Positive", "Negative", "Neutral"],
        "counts": [stats["positive"], stats["negative"], stats["neutral"]],
        "total": stats["total"],
        "most_frequent": stats["most_frequent"]
    }

    return render_template('index.html', data=chart_data)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')

    sentiment, confidence = predict_sentiment(text)

    append_prediction(text, sentiment)

    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence
    })


@app.route('/dashboard-data')
def dashboard_data():
    stats = get_dashboard_stats()
    return jsonify(stats)


# ---------- ANALYTICS PAGE ----------
@app.route('/analytics')
def analytics_page():
    stats = get_dashboard_stats()
    metrics = load_metrics()

    chart_data = {
        "labels": ["Positive", "Negative", "Neutral"],
        "counts": [stats["positive"], stats["negative"], stats["neutral"]]
    }

    return render_template(
        'analytics.html',
        chart_data=chart_data,
        confusion_matrix=metrics.get("confusion_matrix", [[0,0,0],[0,0,0],[0,0,0]]),
        confusion_labels=metrics.get("labels", ["Negative","Neutral","Positive"]),
        accuracies={
            "logistic_regression": metrics.get("logistic_regression_accuracy", 0),
            "naive_bayes": metrics.get("naive_bayes_accuracy", 0),
            "neural_network": metrics.get("neural_network_accuracy", 0)
        },
        total_reviews=stats["total"],
        most_frequent=stats["most_frequent"]
    )

register_multi_model_routes(app)

# ---------- RUN ----------
if __name__ == '__main__':
    ensure_csv()
    app.run(debug=True)
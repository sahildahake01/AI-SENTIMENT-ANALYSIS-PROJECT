# ============================================================
# FIXED SENTIMENT ANALYSIS PIPELINE
# Handles: Amazon (rating + text) + Flipkart (review + rating)
# Memory-efficient with sampling
# ============================================================

import json
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Optional PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not installed. Custom neural network will be skipped.")

# ============================================================
# CONFIGURATION
# ============================================================
AMAZON_FILE = "AMEZON_DATASET.jsonl"
FLIPKART_FILE = "FLIPCART_DATASET.csv"
SAMPLE_SIZE = 150000         # Use 150k total samples
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000

# ============================================================
# 1. DATA LOADING (Memory-efficient with sampling)
# ============================================================
def load_amazon_jsonl_sample(filepath, sample_size=SAMPLE_SIZE):
    """
    Load a random sample of Amazon reviews from JSONL file.
    Assumes columns: 'rating' and 'text'
    """
    print(f"[Amazon] Reading file: {filepath}")
    records = []
    
    # First, count total lines to sample efficiently
    total_lines = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    print(f"  Total records in file: {total_lines:,}")
    
    # Sample line indices
    if total_lines > sample_size:
        indices = np.random.choice(total_lines, sample_size, replace=False)
        indices_set = set(indices)
    else:
        indices_set = None
    
    # Read only sampled lines
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if indices_set is None or i in indices_set:
                try:
                    data = json.loads(line.strip())
                    rating = data.get('rating')
                    text = data.get('text')
                    if rating is not None and text and isinstance(rating, (int, float)):
                        records.append({
                            'review_text': text,
                            'rating': float(rating)
                        })
                except:
                    continue
    
    df = pd.DataFrame(records)
    print(f"[Amazon] Loaded {len(df):,} sampled records")
    return df

def load_flipkart_csv_sample(filepath, sample_size=SAMPLE_SIZE):
    """Load Flipkart reviews, sample if needed."""
    print(f"[Flipkart] Reading file: {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"  Total records: {len(df):,}")
    
    # Identify review text column
    text_col = None
    rating_col = None
    for col in df.columns:
        if col.lower() in ['review', 'review_text', 'reviewtext', 'review description']:
            text_col = col
        if col.lower() in ['rating', 'rate', 'stars']:
            rating_col = col
    
    if text_col is None or rating_col is None:
        raise ValueError(f"Flipkart: Cannot find text/rating columns. Found: {df.columns.tolist()}")
    
    df = df.rename(columns={text_col: 'review_text', rating_col: 'rating'})
    df = df[['review_text', 'rating']].dropna()
    
    # Sample if needed
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        print(f"[Flipkart] Sampled down to {len(df):,} records")
    else:
        print(f"[Flipkart] Using all {len(df):,} records")
    
    return df

# ============================================================
# 2. TEXT PREPROCESSING (same as before)
# ============================================================
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn",
}

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

# ============================================================
# 3. SENTIMENT LABELING
# ============================================================
def assign_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

# ============================================================
# 4. CUSTOM NEURAL NETWORK (PyTorch)
# ============================================================
class CustomSentimentNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_neural_network(X_train, y_train, X_test, y_test, input_dim, epochs=30, batch_size=128):
    print("\n[Custom NN] Training neural network...")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = CustomSentimentNN(input_dim=input_dim, hidden_dim=128, output_dim=3)
    criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 2.5, 1.0]))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_t)
                val_loss = criterion(val_outputs, y_test_t)
                _, val_preds = torch.max(val_outputs, 1)
                val_acc = (val_preds == y_test_t).float().mean()
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f} - Val Acc: {val_acc:.4f}")
            scheduler.step(val_loss)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predictions = torch.max(outputs, 1)
    return predictions.numpy()

# ============================================================
# 5. MAIN PIPELINE
# ============================================================
def run_pipeline():
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS PIPELINE (Memory Efficient)")
    print("="*60 + "\n")
    
    # Load data (with sampling)
    print("--- Loading Data ---")
    amazon_df = load_amazon_jsonl_sample(AMAZON_FILE, sample_size=100000)
    flipkart_df = load_flipkart_csv_sample(FLIPKART_FILE, sample_size=50000)

    # Combine datasets
    df = pd.concat([amazon_df, flipkart_df], ignore_index=True)
    df.drop_duplicates(subset=['review_text'], inplace=True)
    print(f"\n[Merge] Total records: {len(df):,}")
    
    # Preprocess text
    print("\n--- Preprocessing Text ---")
    df['cleaned_text'] = df['review_text'].apply(preprocess_text)
    initial_len = len(df)
    df = df[df['cleaned_text'].str.strip() != ""]
    print(f"  Removed {initial_len - len(df)} empty reviews")
    
    # Add sentiment labels
    print("\n--- Adding Sentiment Labels ---")
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')##_____________________##
    df = df.dropna(subset=['rating'])
    df['sentiment'] = df['rating'].apply(assign_sentiment)
    print("  Distribution:")
    print(df['sentiment'].value_counts().to_string())
    
    # TF-IDF Vectorization
    print("\n--- TF-IDF Vectorization ---")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_text'])
    print(f"  Feature matrix shape: {X.shape}")
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Convert to dense for neural network (optional)
    X_train_dense = X_train[:30000].toarray()
    y_train_nn = y_train[:30000]

    X_test_dense = X_test[:10000].toarray()
    y_test_nn = y_test[:10000]

    # Model training
    print("\n" + "="*60)
    print("MODEL TRAINING & EVALUATION")
    print("="*60)
    
    results = {}
    
    # Logistic Regression
    print("\n--- Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    results['Logistic Regression'] = lr_acc
    print(f"  Accuracy: {lr_acc:.4f}")
    print(classification_report(y_test, lr_pred, target_names=le.classes_))
    
    # Naïve Bayes
    print("\n--- Naïve Bayes ---")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    results['Naïve Bayes'] = nb_acc
    print(f"  Accuracy: {nb_acc:.4f}")
    print(classification_report(y_test, nb_pred, target_names=le.classes_))
    
    # Custom Neural Network
    if PYTORCH_AVAILABLE:
        print("\n--- Custom Neural Network ---")
        nn_pred = train_neural_network(X_train_dense, y_train_nn, X_test_dense, y_test_nn, input_dim=MAX_FEATURES,epochs=30)
        nn_acc = accuracy_score(y_test_nn, nn_pred)
        results['Custom Neural Network'] = nn_acc
        print(f"  Accuracy: {nn_acc:.4f}")
        print(classification_report(y_test_nn, nn_pred, target_names=le.classes_))
    else:
        print("\n[Skip] Custom Neural Network - PyTorch not installed")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':<10}")
    print("-"*35)
    for name, acc in results.items():
        print(f"{name:<25} {acc:.4f}")
    
    # Save output
    output_df = pd.DataFrame({
        'review_text': df['review_text'],
        'rating': df['rating'],
        'actual_sentiment': df['sentiment'],
        'predicted_lr': le.inverse_transform(lr_model.predict(X)),
        'predicted_nb': le.inverse_transform(nb_model.predict(X)),
    })
    output_df.to_csv('sentiment_results.csv', index=False, encoding='utf-8-sig')
    print("\n[Output] Results saved to 'sentiment_results.csv'")
    
    # ============================================================
    # ==== ADDED FOR FLASK INTEGRATION ====
    # ============================================================

    import json

    # 1. Confusion Matrix (TEST DATA)
    cm = confusion_matrix(y_test, lr_pred)

    # 2. Metrics JSON
    metrics = {
    "confusion_matrix": cm.tolist(),
    "labels": le.classes_.tolist(),
    "logistic_regression_accuracy": float(lr_acc),
    "naive_bayes_accuracy": float(nb_acc),
    "neural_network_accuracy": float(nn_acc) if 'nn_acc' in locals() else 0.0
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[Flask] metrics.json saved ✅")

    # 3. Save trained model
    pickle.dump(lr_model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
    pickle.dump(le, open("label_encoder.pkl", "wb"))

    print("[Flask] model.pkl, vectorizer.pkl, label_encoder.pkl saved ✅")

    pickle.dump(lr_model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
    pickle.dump(le, open("label_encoder.pkl", "wb"))

    return results

if __name__ == "__main__":
    run_pipeline()


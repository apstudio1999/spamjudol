#!/usr/bin/env python3
"""
TRAIN LOKAL - Script untuk training GNN model lokal
Untuk training dari dataset baru atau retrain model

Penggunaan:
    python train_local.py --dataset <path_csv> [--epochs 30] [--train]
    
Contoh:
    python train_local.py --dataset dataset_youtube_5000_realistic.csv --epochs 30 --train
"""

import os
import sys
import json
import pickle
import re
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import Sastrawi untuk text processing
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ============================================================================
# KONFIGURASI
# ============================================================================
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VECTORIZER_FILE = "tfidf_vectorizer.pkl"
    MODEL_FILE = "gnn_spam_model.pt"
    EPOCHS_DEFAULT = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    KNN_NEIGHBORS = 5
    SIM_THRESHOLD = 0.15

print(f"Device: {Config.DEVICE}")

# ============================================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================================

# Setup Sastrawi
stop_factory = StopWordRemoverFactory()
stop_remover = stop_factory.create_stop_word_remover()
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def clean_text_step(text):
    """Remove URLs, special chars, extra spaces"""
    t = re.sub(r"http\S+", "", str(text))
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    t = t.lower().strip()
    return t

def regex_tokenize(text):
    """Tokenize text into words"""
    if not isinstance(text, str):
        return []
    return re.findall(r'\b\w+\b', text.lower())

def stopword_remove_string(text):
    """Remove Indonesian stopwords"""
    try:
        return stop_remover.remove(text)
    except Exception:
        return text

def normalize_text(text):
    """Normalize text: slang correction, duplicate char removal"""
    t = re.sub(r'(.)\1{2,}', r'\1\1', text)
    t = re.sub(r'\s+', ' ', t)
    slang_map = {
        "yg": "yang", "dg": "dengan", "gk": "gak", "ga": "gak",
        "tdk": "tidak", "klo": "kalau"
    }
    tokens = t.split()
    tokens = [slang_map.get(tok, tok) for tok in tokens]
    return " ".join(tokens)

def stem_string(text):
    """Stem text"""
    try:
        return stemmer.stem(text)
    except Exception:
        return text

def preprocess_text(text):
    """Full preprocessing pipeline"""
    text = clean_text_step(text)
    text = stopword_remove_string(text)
    text = normalize_text(text)
    text = stem_string(text)
    return text

# ============================================================================
# GRAPH BUILDING
# ============================================================================

def build_reply_graph(df):
    """Build graph dari reply relationships (comment_id -> parent_id)"""
    print("Building reply-based graph...")
    G = nx.DiGraph()
    n = len(df)
    G.add_nodes_from(range(n))
    
    if "comment_id" in df.columns and "parent_id" in df.columns:
        id2idx = {str(r["comment_id"]): idx for idx, r in df.reset_index().iterrows()}
        edge_ct = 0
        for idx, r in df.reset_index().iterrows():
            pid = r.get("parent_id", None)
            if pd.notna(pid) and str(pid) in id2idx:
                G.add_edge(id2idx[str(pid)], idx)
                edge_ct += 1
        print(f"  Reply graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, "reply_graph"
    else:
        print("  No comment_id/parent_id columns found, fallback to similarity graph")
        return None, None

def build_similarity_graph(df, text_col="stemmed"):
    """Build k-NN graph based on TF-IDF similarity"""
    print("Building TF-IDF similarity k-NN graph...")
    texts = df[text_col].astype(str).tolist()
    vec = TfidfVectorizer(max_features=2000)
    X = vec.fit_transform(texts).toarray()
    sim = cosine_similarity(X)
    
    G = nx.DiGraph()
    n = len(texts)
    for i in range(n):
        G.add_node(i)
        neighbors = np.argsort(sim[i])[::-1][1:Config.KNN_NEIGHBORS+1]
        for j in neighbors:
            if sim[i, j] > Config.SIM_THRESHOLD:
                G.add_edge(i, int(j))
    
    print(f"  Similarity graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, "similarity_knn"

# ============================================================================
# MODEL
# ============================================================================

class SimpleGNN(torch.nn.Module):
    """Graph Neural Network for spam detection"""
    def __init__(self, in_channels):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, 2)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, data, train_idx, optimizer, loss_fn, epochs):
    """Train GNN model"""
    print(f"\nTraining {epochs} epochs...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(data.x.to(Config.DEVICE), data.edge_index.to(Config.DEVICE))
        loss = loss_fn(out[train_idx.to(Config.DEVICE)], data.y[train_idx].to(Config.DEVICE))
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs} loss: {loss.item():.4f}")
    
    print("Training selesai!")

# ============================================================================
# EVALUATION & METRICS
# ============================================================================

def evaluate_model(model, data, test_idx, device):
    """Evaluate model pada test set"""
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        probs = torch.exp(out)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
    
    return preds, probs

def compute_metrics(y_true, y_pred, y_score):
    """Compute all evaluation metrics"""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score))
    }
    return metrics

def plot_confusion_matrix(cm, save_path="confusion_matrix_gnn.png"):
    """Plot dan save confusion matrix"""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Spam", "Spam"],
                yticklabels=["Non-Spam", "Spam"])
    plt.title("Confusion Matrix — GNN")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved: {save_path}")

# ============================================================================
# BASELINE COMPARISON
# ============================================================================

def baseline_comparison(X, y, y_pred_gnn, metrics_gnn, test_idx):
    """Compare GNN dengan Logistic Regression dan SVM"""
    print("\nBaseline comparison (LR, SVM vs GNN)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    lr = LogisticRegression(max_iter=200).fit(X_train, y_train)
    svm = SVC(kernel="linear", probability=True).fit(X_train, y_train)
    
    lr_pred = lr.predict(X_test)
    svm_pred = svm.predict(X_test)
    
    table = pd.DataFrame({
        "Model": ["LogisticRegression", "SVM", "GNN"],
        "Accuracy": [
            accuracy_score(y_test, lr_pred),
            accuracy_score(y_test, svm_pred),
            metrics_gnn["accuracy"]
        ],
        "Precision": [
            precision_score(y_test, lr_pred, zero_division=0),
            precision_score(y_test, svm_pred, zero_division=0),
            metrics_gnn["precision"]
        ],
        "Recall": [
            recall_score(y_test, lr_pred, zero_division=0),
            recall_score(y_test, svm_pred, zero_division=0),
            metrics_gnn["recall"]
        ],
        "F1": [
            f1_score(y_test, lr_pred, zero_division=0),
            f1_score(y_test, svm_pred, zero_division=0),
            metrics_gnn["f1"]
        ]
    })
    
    return table

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train GNN model untuk YouTube Spam Detection"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_youtube_5000_realistic.csv",
        help="Path ke dataset CSV"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=Config.EPOCHS_DEFAULT,
        help=f"Jumlah epochs training (default: {Config.EPOCHS_DEFAULT})"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Set untuk melakukan training baru (default: hanya preprocess)"
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline comparison"
    )
    
    args = parser.parse_args()
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  YouTube Spam Detector - Training Lokal                     ║
╚════════════════════════════════════════════════════════════╝

Konfigurasi:
  Dataset: {args.dataset}
  Training: {args.train}
  Epochs: {args.epochs}
  Device: {Config.DEVICE}
    """)
    
    # ========== STEP 0: Load Dataset ==========
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset tidak ditemukan: {args.dataset}")
        return False
    
    print(f"\n[STEP 1/5] Load dataset...")
    df_orig = pd.read_csv(args.dataset)
    print(f"  ✓ Loaded {len(df_orig)} rows, columns: {df_orig.columns.tolist()}")
    
    # Validasi label column
    if "label" not in df_orig.columns:
        print("❌ Dataset harus memiliki kolom 'label'")
        return False
    
    # Cari text column
    text_col_candidates = [c for c in df_orig.columns 
                          if c.lower() in ("comment_text", "text", "content", "comment")]
    if not text_col_candidates:
        print("❌ Tidak menemukan kolom text. Rename kolom ke 'comment_text' atau 'text'")
        return False
    text_col = text_col_candidates[0]
    print(f"  ✓ Text column: {text_col}")
    print(f"  ✓ Label distribution: {df_orig['label'].value_counts().to_dict()}")
    
    # ========== STEP 1: Preprocess Dataset ==========
    print(f"\n[STEP 2/5] Preprocess text (stemming, stopword removal)...")
    df = df_orig.copy()
    df["clean_text"] = df[text_col].astype(str).apply(clean_text_step)
    df["tokens"] = df["clean_text"].apply(regex_tokenize)
    df["no_stop"] = df["clean_text"].apply(stopword_remove_string)
    df["normalized"] = df["no_stop"].apply(normalize_text)
    df["stemmed"] = df["normalized"].apply(stem_string)
    
    # Save preprocessed dataset
    df.to_csv("dataset_stemmed_with_label.csv", index=False)
    print(f"  ✓ Saved: dataset_stemmed_with_label.csv")
    
    # ========== STEP 2: Build Graph ==========
    print(f"\n[STEP 3/5] Build graph structure...")
    G, graph_method = build_reply_graph(df)
    
    if G is None:
        G, graph_method = build_similarity_graph(df, text_col="stemmed")
    
    # Save graph
    with open("graph_structure_generated.pkl", "wb") as f:
        pickle.dump(G, f)
    edges = list(G.edges())
    edge_arr = np.array(edges).T if len(edges) > 0 else np.zeros((2, 0), dtype=np.int64)
    np.save("edge_index.npy", edge_arr)
    with open("graph_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": graph_method,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges()
        }, f, indent=2)
    print(f"  ✓ Graph saved (method: {graph_method})")
    
    # ========== STEP 3: Extract Features ==========
    print(f"\n[STEP 4/5] Extract TF-IDF features...")
    texts = df["stemmed"].astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(texts).toarray()
    joblib.dump(vectorizer, Config.VECTORIZER_FILE)
    print(f"  ✓ Saved vectorizer: {Config.VECTORIZER_FILE}")
    print(f"  ✓ Feature matrix shape: {X.shape}")
    
    # ========== STEP 4: Prepare Data ==========
    print(f"\n[STEP 5/5] Prepare PyTorch data...")
    y = df["label"].astype(int).values
    
    # Load edges
    if os.path.exists("edge_index.npy"):
        edge_arr = np.load("edge_index.npy", allow_pickle=True)
        edges_list = list(map(tuple, edge_arr.T.tolist())) if edge_arr.size else []
    else:
        edges_list = list(G.edges())
    
    # Create tensors
    edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous() \
                 if len(edges_list) > 0 else torch.empty((2, 0), dtype=torch.long)
    
    x_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Pad if needed
    if edge_index.numel() > 0:
        max_idx = int(edge_index.max().item())
        if max_idx + 1 > x_tensor.shape[0]:
            pad_rows = max_idx + 1 - x_tensor.shape[0]
            pad = torch.zeros((pad_rows, x_tensor.shape[1]), dtype=x_tensor.dtype)
            x_tensor = torch.cat([x_tensor, pad], dim=0)
            y_tensor = torch.cat([y_tensor, torch.full((pad_rows,), -1, dtype=y_tensor.dtype)], dim=0)
    
    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    print(f"  ✓ Data prepared: {data.x.shape}, edges: {tuple(edge_index.shape)}")
    
    # Train/test split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=Config.TEST_SIZE, 
                                          random_state=Config.RANDOM_STATE)
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)
    print(f"  ✓ Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    
    # ========== STEP 5: Model Setup & Training ==========
    print(f"\n[MODEL] Setup GNN model...")
    model = SimpleGNN(x_tensor.shape[1]).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=Config.LEARNING_RATE,
                                weight_decay=Config.WEIGHT_DECAY)
    loss_fn = torch.nn.NLLLoss()
    print(f"  ✓ Model ready on {Config.DEVICE}")
    
    if args.train:
        train_model(model, data, train_idx, optimizer, loss_fn, args.epochs)
        model_file = f"gnn_spam_model_trained_{int(time.time())}.pt"
        torch.save(model.state_dict(), model_file)
        print(f"  ✓ Saved trained model: {model_file}")
        # Also save as default
        torch.save(model.state_dict(), Config.MODEL_FILE)
        print(f"  ✓ Also saved as: {Config.MODEL_FILE}")
    else:
        if os.path.exists(Config.MODEL_FILE):
            state = torch.load(Config.MODEL_FILE, map_location=Config.DEVICE)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            print(f"  ✓ Loaded pretrained model: {Config.MODEL_FILE}")
        else:
            print(f"⚠ Model {Config.MODEL_FILE} tidak ditemukan, menggunakan model yang baru dibuat")
    
    # ========== STEP 6: Evaluation ==========
    print(f"\n[EVALUATION] Testing model...")
    y_pred, y_score = evaluate_model(model, data, test_idx, Config.DEVICE)
    
    y_true = y[test_idx.numpy()]
    y_pred = y_pred[test_idx.numpy()]
    y_score = y_score[test_idx.numpy()]
    
    metrics = compute_metrics(y_true, y_pred, y_score)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Save metrics
    with open("metrics_gnn.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Saved: metrics_gnn.json")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm)
    
    # Save predictions
    df_out = df.copy()
    df_out["pred_label"] = y_pred[:len(df_out)]
    df_out["spam_score"] = y_score[:len(df_out)]
    df_out.to_csv("predictions_gnn_full.csv", index=False)
    print(f"  ✓ Saved: predictions_gnn_full.csv")
    
    # Baseline comparison
    if not args.no_baseline:
        baseline_table = baseline_comparison(X, y, y_pred, metrics, test_idx)
        baseline_table.to_csv("baseline_comparison.csv", index=False)
        print(f"  ✓ Saved: baseline_comparison.csv")
        print("\nBaseline Comparison:")
        print(baseline_table.to_string(index=False))
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  Training Selesai!                                         ║
╚════════════════════════════════════════════════════════════╝

Output files:
  ✓ gnn_spam_model.pt (trained model)
  ✓ tfidf_vectorizer.pkl (feature extractor)
  ✓ dataset_stemmed_with_label.csv (preprocessed data)
  ✓ graph_structure_generated.pkl (graph structure)
  ✓ edge_index.npy (edge indices)
  ✓ graph_summary.json (graph info)
  ✓ metrics_gnn.json (evaluation metrics)
  ✓ predictions_gnn_full.csv (predictions)
  ✓ confusion_matrix_gnn.png (confusion matrix plot)
  ✓ baseline_comparison.csv (model comparison)

Selanjutnya:
  1. Jalankan inference: python inference_local.py
  2. Jalankan Streamlit UI: streamlit run streamlit_app.py
    """)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

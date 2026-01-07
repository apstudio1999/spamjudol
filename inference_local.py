#!/usr/bin/env python3
"""
INFERENCE LOKAL - Prediksi spam pada komentar baru
Gunakan model yang sudah terlatih untuk membuat prediksi

Penggunaan:
    python inference_local.py <path_to_csv> [--output results.csv]
    python inference_local.py --text "ini adalah komentar yang ingin diperiksa"
    
Contoh:
    python inference_local.py labels_to_fill.csv --output predictions.csv
    python inference_local.py --text "subscribe channel kami gratis"
"""

import os
import sys
import re
import pickle
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import joblib
import warnings
warnings.filterwarnings('ignore')

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ============================================================================
# KONFIGURASI
# ============================================================================
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_FILE = "gnn_spam_model.pt"
    VECTORIZER_FILE = "tfidf_vectorizer.pkl"

# Setup Sastrawi
stop_factory = StopWordRemoverFactory()
stop_remover = stop_factory.create_stop_word_remover()
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

# ============================================================================
# TEXT PROCESSING (sama seperti training)
# ============================================================================

def clean_text_step(text):
    t = re.sub(r"http\S+", "", str(text))
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.lower().strip()

def stopword_remove_string(text):
    try:
        return stop_remover.remove(text)
    except:
        return text

def normalize_text(text):
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
    try:
        return stemmer.stem(text)
    except:
        return text

def preprocess_text(text):
    """Full preprocessing pipeline"""
    text = clean_text_step(text)
    text = stopword_remove_string(text)
    text = normalize_text(text)
    text = stem_string(text)
    return text

# ============================================================================
# MODEL (same as training)
# ============================================================================

class SimpleGNN(torch.nn.Module):
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
# INFERENCE ENGINE
# ============================================================================

class SpamDetector:
    def __init__(self, model_path=None, vectorizer_path=None):
        """Initialize spam detector dengan model dan vectorizer"""
        self.model_path = model_path or Config.MODEL_FILE
        self.vectorizer_path = vectorizer_path or Config.VECTORIZER_FILE
        self.device = Config.DEVICE
        
        # Load vectorizer
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer tidak ditemukan: {self.vectorizer_path}")
        
        self.vectorizer = joblib.load(self.vectorizer_path)
        print(f"✓ Loaded vectorizer from {self.vectorizer_path}")
        
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model tidak ditemukan: {self.model_path}")
        
        in_channels = self.vectorizer.get_feature_names_out().shape[0]
        self.model = SimpleGNN(in_channels).to(self.device)
        
        state = torch.load(self.model_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"✓ Loaded model from {self.model_path}")
    
    def predict(self, texts):
        """
        Predict pada batch texts
        
        Args:
            texts: str atau list of str
            
        Returns:
            list of dict dengan keys: 'text', 'prediction', 'confidence', 'label'
        """
        # Convert to list jika string
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess
        processed = [preprocess_text(t) for t in texts]
        
        # Vectorize
        X = self.vectorizer.transform(processed).toarray()
        
        # Predict
        x_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        
        # Dummy edge_index (no graph untuk inference)
        edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            out = self.model(x_tensor, edge_index)
            probs = torch.exp(out).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
        
        results = []
        for i, orig_text in enumerate(texts):
            is_spam = preds[i]
            confidence = probs[i][is_spam]
            results.append({
                "text": orig_text,
                "prediction": int(is_spam),
                "confidence": float(confidence),
                "label": "Spam" if is_spam else "Non-Spam",
                "spam_score": float(probs[i][1])  # Probability of being spam
            })
        
        return results

# ============================================================================
# CLI & FILE PROCESSING
# ============================================================================

def predict_csv(csv_path, detector, output_path=None):
    """Predict pada CSV file"""
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Find text column
    text_cols = [c for c in df.columns 
                if c.lower() in ("comment_text", "text", "content", "comment")]
    
    if not text_cols:
        print(f"❌ Tidak menemukan kolom text dalam CSV")
        print(f"   Kolom yang tersedia: {df.columns.tolist()}")
        return False
    
    text_col = text_cols[0]
    print(f"✓ Menggunakan kolom: {text_col}")
    
    # Predict
    print(f"Predicting {len(df)} rows...")
    texts = df[text_col].astype(str).tolist()
    results = detector.predict(texts)
    
    # Add to dataframe
    df["pred_label"] = [r["prediction"] for r in results]
    df["pred_confidence"] = [r["confidence"] for r in results]
    df["spam_score"] = [r["spam_score"] for r in results]
    
    # Save
    if output_path is None:
        output_path = csv_path.replace(".csv", "_predicted.csv")
    
    df.to_csv(output_path, index=False)
    print(f"✓ Saved predictions to: {output_path}")
    
    # Statistics
    print(f"\nPrediction Statistics:")
    print(f"  Total predictions: {len(df)}")
    print(f"  Spam detected: {(df['pred_label'] == 1).sum()}")
    print(f"  Non-spam: {(df['pred_label'] == 0).sum()}")
    print(f"  Average confidence: {df['pred_confidence'].mean():.4f}")
    
    return True

def predict_text(text, detector):
    """Predict single text"""
    print(f"\nAnalyzing: {text}")
    result = detector.predict(text)[0]
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  PREDICTION RESULT                                         ║
╚════════════════════════════════════════════════════════════╝

Text:       {result['text']}
Label:      {result['label']}
Confidence: {result['confidence']:.4f}
Spam Score: {result['spam_score']:.4f}
    """)
    
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inference lokal untuk YouTube Spam Detection"
    )
    
    # Arguments
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path ke CSV file untuk batch prediction"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single text untuk prediksi (gunakan ini atau input file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path untuk output CSV (default: input_predicted.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=Config.MODEL_FILE,
        help=f"Path ke model file (default: {Config.MODEL_FILE})"
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        default=Config.VECTORIZER_FILE,
        help=f"Path ke vectorizer file (default: {Config.VECTORIZER_FILE})"
    )
    
    args = parser.parse_args()
    
    # Validate
    if not args.input and not args.text:
        parser.print_help()
        print(f"""
Contoh penggunaan:

  # Batch prediction dari CSV
  python inference_local.py labels_to_fill.csv --output predictions.csv
  
  # Single text prediction
  python inference_local.py --text "subscribe channel kami gratis"
        """)
        return False
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  YouTube Spam Detector - Inference Lokal                    ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Load detector
    print("\n[INIT] Loading model...")
    try:
        detector = SpamDetector(
            model_path=args.model,
            vectorizer_path=args.vectorizer
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False
    
    # Predict
    print(f"\n[PREDICT]")
    
    if args.text:
        return predict_text(args.text, detector)
    else:
        return predict_csv(args.input, detector, args.output)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""
Streamlit App untuk YouTube Spam Detector (Local Version)
=========================================================

Aplikasi web interaktif untuk prediksi spam komentar YouTube.
Dioptimalkan untuk berjalan lokal (tanpa Google Colab).

Usage:
    streamlit run streamlit_app.py

Features:
    ✓ Single text prediction
    ✓ Batch CSV upload & prediction
    ✓ Real-time visualization
    ✓ Download results
    ✓ Model statistics
"""

import os
import re
import io
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ============================================================================
# CONFIG & SETUP
# ============================================================================

st.set_page_config(
    page_title="YouTube Spam Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .spam-badge {
        color: #d62728;
        font-weight: bold;
    }
    .non-spam-badge {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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
# TEXT PROCESSING
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
    text = clean_text_step(text)
    text = stopword_remove_string(text)
    text = normalize_text(text)
    text = stem_string(text)
    return text

# ============================================================================
# MODEL
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

@st.cache_resource
def load_model_and_vectorizer():
    """Load model dan vectorizer (cached)"""
    # Load vectorizer
    if not os.path.exists(Config.VECTORIZER_FILE):
        st.error(f"❌ Vectorizer file not found: {Config.VECTORIZER_FILE}")
        st.stop()
    
    vectorizer = joblib.load(Config.VECTORIZER_FILE)
    
    # Load model
    if not os.path.exists(Config.MODEL_FILE):
        st.error(f"❌ Model file not found: {Config.MODEL_FILE}")
        st.stop()
    
    in_channels = vectorizer.get_feature_names_out().shape[0]
    model = SimpleGNN(in_channels).to(Config.DEVICE)
    
    state = torch.load(Config.MODEL_FILE, map_location=Config.DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    
    return model, vectorizer

def predict_texts(texts, model, vectorizer):
    """Predict pada batch texts"""
    if isinstance(texts, str):
        texts = [texts]
    
    # Preprocess
    processed = [preprocess_text(t) for t in texts]
    
    # Vectorize
    X = vectorizer.transform(processed).toarray()
    
    # Predict
    x_tensor = torch.tensor(X, dtype=torch.float).to(Config.DEVICE)
    edge_index = torch.empty((2, 0), dtype=torch.long).to(Config.DEVICE)
    
    with torch.no_grad():
        out = model(x_tensor, edge_index)
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
            "spam_score": float(probs[i][1])
        })
    
    return results

# ============================================================================
# UI - MAIN PAGE
# ============================================================================

def main():
    # Header
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("🔍 YouTube Spam Detector")
        st.markdown("Deteksi spam dalam komentar YouTube menggunakan Graph Neural Network")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Load model
    try:
        model, vectorizer = load_model_and_vectorizer()
        st.sidebar.success("✓ Model loaded")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    mode = st.sidebar.radio(
        "Select Mode",
        ["📝 Single Prediction", "📊 Batch Upload", "📈 Statistics"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.markdown("""
    **Model:** Graph Neural Network (GNN)
    
    **Input:** TF-IDF vectors
    
    **Features:** 2000 dimensions
    
    **Classes:** Spam / Non-Spam
    
    **Device:** GPU (if available)
    """)
    
    # ========== MODE 1: Single Prediction ==========
    if mode == "📝 Single Prediction":
        st.header("Single Comment Prediction")
        
        col1, col2 = st.columns([0.7, 0.3])
        
        with col1:
            text_input = st.text_area(
                "Paste comment to analyze:",
                placeholder="e.g., 'Subscribe to my channel for more videos'",
                height=120
            )
        
        with col2:
            st.markdown("### Result")
            predict_button = st.button("🔍 Predict", use_container_width=True)
        
        if predict_button and text_input:
            with st.spinner("Analyzing..."):
                results = predict_texts(text_input, model, vectorizer)
                result = results[0]
            
            # Display result
            if result["prediction"] == 1:
                st.error(f"### 🚨 **SPAM**")
                badge_class = "spam-badge"
            else:
                st.success(f"### ✅ **NON-SPAM**")
                badge_class = "non-spam-badge"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Label", result["label"])
            with col2:
                st.metric("Confidence", f"{result['confidence']:.2%}")
            with col3:
                st.metric("Spam Score", f"{result['spam_score']:.4f}")
            
            # Processed text
            with st.expander("📖 Processed Text (after preprocessing)"):
                processed = preprocess_text(text_input)
                st.code(processed)
        
        elif predict_button and not text_input:
            st.warning("Please enter a comment to analyze")
    
    # ========== MODE 2: Batch Upload ==========
    elif mode == "📊 Batch Upload":
        st.header("Batch Prediction from CSV")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="CSV harus memiliki kolom: comment_text, text, content, atau comment"
        )
        
        if uploaded_file is not None:
            # Load CSV
            df = pd.read_csv(uploaded_file)
            st.info(f"✓ Loaded {len(df)} rows")
            
            # Find text column
            text_cols = [c for c in df.columns 
                        if c.lower() in ("comment_text", "text", "content", "comment")]
            
            if not text_cols:
                st.error("❌ CSV harus memiliki kolom: comment_text, text, content, atau comment")
                st.dataframe(df.head())
                st.stop()
            
            text_col = text_cols[0]
            st.info(f"✓ Using column: **{text_col}**")
            
            # Predict
            if st.button("🔍 Predict All", use_container_width=True):
                with st.spinner(f"Predicting {len(df)} rows..."):
                    texts = df[text_col].astype(str).tolist()
                    results = predict_texts(texts, model, vectorizer)
                
                # Add results to dataframe
                df["pred_label"] = [r["prediction"] for r in results]
                df["pred_confidence"] = [r["confidence"] for r in results]
                df["spam_score"] = [r["spam_score"] for r in results]
                df["label"] = [r["label"] for r in results]
                
                st.success("✓ Prediction complete!")
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", len(df))
                with col2:
                    spam_count = (df["pred_label"] == 1).sum()
                    st.metric("Spam Detected", spam_count)
                with col3:
                    non_spam_count = (df["pred_label"] == 0).sum()
                    st.metric("Non-Spam", non_spam_count)
                with col4:
                    avg_conf = df["pred_confidence"].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.2%}")
                
                # Show results
                st.dataframe(
                    df[[text_col, "label", "pred_confidence", "spam_score"]].head(20),
                    use_container_width=True
                )
                
                # Download
                csv_download = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv_download,
                    file_name="spam_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Visualization
                st.subheader("📊 Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    labels_count = df["label"].value_counts()
                    ax.pie(labels_count.values, labels=labels_count.index, 
                          autopct='%1.1f%%', colors=['#d62728', '#2ca02c'],
                          startangle=90)
                    ax.set_title("Spam vs Non-Spam Distribution")
                    st.pyplot(fig, use_container_width=True)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(df[df["pred_label"] == 0]["pred_confidence"], 
                           bins=20, alpha=0.6, label="Non-Spam", color='green')
                    ax.hist(df[df["pred_label"] == 1]["pred_confidence"], 
                           bins=20, alpha=0.6, label="Spam", color='red')
                    ax.set_xlabel("Confidence Score")
                    ax.set_ylabel("Count")
                    ax.set_title("Confidence Distribution by Label")
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)
    
    # ========== MODE 3: Statistics ==========
    elif mode == "📈 Statistics":
        st.header("Model Statistics & Info")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 Model Info")
            st.markdown(f"""
            **Architecture:** Graph Neural Network (GCN)
            
            **Layers:**
            - GCNConv (2000 → 128)
            - ReLU Activation
            - Dropout (0.4)
            - GCNConv (128 → 2)
            
            **Optimizer:** Adam
            - Learning rate: 0.001
            - Weight decay: 5e-4
            
            **Loss:** Negative Log Likelihood
            
            **Device:** {Config.DEVICE}
            """)
        
        with col2:
            st.subheader("📊 Model Performance")
            
            # Try to load metrics
            metrics_file = "metrics_gnn.json"
            if os.path.exists(metrics_file):
                with open(metrics_file) as f:
                    metrics = json.load(f)
                
                st.markdown(f"""
                **Test Set Metrics:**
                
                - **Accuracy:** {metrics.get('accuracy', 0):.4f}
                - **Precision:** {metrics.get('precision', 0):.4f}
                - **Recall:** {metrics.get('recall', 0):.4f}
                - **F1-Score:** {metrics.get('f1', 0):.4f}
                - **ROC-AUC:** {metrics.get('roc_auc', 0):.4f}
                """)
            else:
                st.info("Metrics file not found. Run training to generate metrics.")
        
        st.markdown("---")
        
        # Baseline comparison
        baseline_file = "baseline_comparison.csv"
        if os.path.exists(baseline_file):
            st.subheader("📈 Model Comparison")
            baseline_df = pd.read_csv(baseline_file)
            st.dataframe(baseline_df, use_container_width=True)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            baseline_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(
                kind="bar", ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )
            ax.set_title("Model Performance Comparison")
            ax.set_ylabel("Score")
            ax.set_xlabel("Model")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        # Confusion matrix
        cm_file = "confusion_matrix_gnn.png"
        if os.path.exists(cm_file):
            st.subheader("🔲 Confusion Matrix")
            from PIL import Image
            img = Image.open(cm_file)
            st.image(img, use_container_width=False, width=400)
        
        st.markdown("---")
        
        st.subheader("📚 Preprocessing Pipeline")
        st.markdown("""
        Setiap teks melalui preprocessing berikut:
        
        1. **URL Removal** — Hapus semua links
        2. **Character Cleaning** — Hapus special characters, keep alphanumeric & space
        3. **Lowercasing** — Convert semua ke lowercase
        4. **Stopword Removal** — Hapus kata umum (Indonesian/English)
        5. **Text Normalization** — 
           - Remove duplicate characters (e.g., "ooooo" → "oo")
           - Slang correction (e.g., "yg" → "yang")
        6. **Stemming** — Reduce ke root word (Sastrawi library)
        7. **Vectorization** — TF-IDF (2000 features)
        """)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()

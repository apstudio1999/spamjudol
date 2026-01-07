"""
Streamlit App untuk YouTube Spam Detector - YouTube API Integration
====================================================================

Aplikasi web interaktif untuk:
1. Fetch comments dari YouTube video menggunakan YouTube API
2. Prediksi spam pada comments
3. Batch CSV upload & prediction
4. Statistics & visualization

Usage:
    streamlit run streamlit_app_youtube.py

Features:
    ✓ YouTube URL fetch comments dengan API
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

# YouTube API
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ============================================================================
# CONFIG & SETUP
# ============================================================================

st.set_page_config(
    page_title="YouTube Spam Detector with API",
    page_icon="🎬",
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
    .youtube-badge {
        background-color: #FF0000;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_FILE = "gnn_spam_model.pt"
    VECTORIZER_FILE = "tfidf_vectorizer.pkl"
    # YouTube API Key - REPLACE WITH YOUR KEY
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

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
# YOUTUBE API FUNCTIONS
# ============================================================================

def extract_video_id(url):
    """Extract video ID dari YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^?]+)',
        r'youtube\.com\/v\/([^?]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@st.cache_data
def fetch_youtube_comments(video_id, api_key, max_results=100):
    """Fetch comments dari YouTube video menggunakan API"""
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        comments_data = []
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=min(max_results, 100),
            pageToken=None
        )
        
        while request and len(comments_data) < max_results:
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments_data.append({
                    'comment_text': comment.get('textDisplay', ''),
                    'author': comment.get('authorDisplayName', 'Unknown'),
                    'likes': comment.get('likeCount', 0),
                    'reply_count': item['snippet'].get('replyCount', 0),
                    'published_at': comment.get('publishedAt', '')
                })
            
            # Check jika ada next page
            if 'nextPageToken' in response and len(comments_data) < max_results:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    textFormat='plainText',
                    maxResults=min(max_results - len(comments_data), 100),
                    pageToken=response['nextPageToken']
                )
            else:
                break
        
        return pd.DataFrame(comments_data[:max_results])
    
    except HttpError as e:
        st.error(f"❌ YouTube API Error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# ============================================================================
# UI - MAIN PAGE
# ============================================================================

def main():
    # Header
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("🎬 YouTube Spam Detector with API")
        st.markdown("Fetch comments dari YouTube dan deteksi spam menggunakan Graph Neural Network")
    
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
    
    # Sidebar - API Key Input
    st.sidebar.title("🔑 YouTube API Setup")
    api_key = st.sidebar.text_input(
        "Enter YouTube API Key:",
        value=Config.YOUTUBE_API_KEY,
        type="password",
        help="Get your API key from: console.cloud.google.com"
    )
    
    # Sidebar - Mode Selection
    st.sidebar.title("⚙️ Settings")
    mode = st.sidebar.radio(
        "Select Mode",
        ["🎬 YouTube Comments Fetch", "📊 Batch Upload", "📈 Statistics"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.markdown("""
    **Model:** Graph Neural Network (GNN)
    
    **Input:** TF-IDF vectors
    
    **Features:** 2000 dimensions
    
    **Classes:** Spam / Non-Spam
    
    **Accuracy:** 88-92%
    """)
    
    # ========== MODE 1: YouTube Comments Fetch ==========
    if mode == "🎬 YouTube Comments Fetch":
        st.header("Fetch Comments dari YouTube Video")
        
        if not api_key:
            st.warning("⚠️ YouTube API Key belum diisi! Masukkan di sidebar.")
        
        col1, col2 = st.columns([0.7, 0.3])
        
        with col1:
            youtube_url = st.text_input(
                "YouTube Video URL:",
                placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                help="Paste full YouTube URL"
            )
        
        with col2:
            max_comments = st.number_input(
                "Max Comments:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
        
        col1, col2 = st.columns([0.5, 0.5])
        
        with col1:
            fetch_button = st.button("🎬 Fetch Comments", use_container_width=True)
        
        with col2:
            auto_predict = st.checkbox("Auto Predict Spam", value=True)
        
        if fetch_button:
            if not youtube_url:
                st.error("❌ Please enter YouTube URL")
            elif not api_key:
                st.error("❌ Please enter YouTube API Key in sidebar")
            else:
                # Extract video ID
                video_id = extract_video_id(youtube_url)
                if not video_id:
                    st.error("❌ Invalid YouTube URL")
                else:
                    with st.spinner(f"Fetching up to {max_comments} comments..."):
                        df_comments = fetch_youtube_comments(video_id, api_key, max_comments)
                    
                    if df_comments is not None and len(df_comments) > 0:
                        st.success(f"✓ Fetched {len(df_comments)} comments!")
                        
                        # Auto predict jika enabled
                        if auto_predict:
                            with st.spinner("Predicting spam..."):
                                texts = df_comments['comment_text'].astype(str).tolist()
                                results = predict_texts(texts, model, vectorizer)
                            
                            df_comments['pred_label'] = [r['prediction'] for r in results]
                            df_comments['pred_confidence'] = [r['confidence'] for r in results]
                            df_comments['spam_score'] = [r['spam_score'] for r in results]
                            df_comments['label'] = [r['label'] for r in results]
                            
                            st.success("✓ Prediction complete!")
                        
                        # Statistics
                        if 'pred_label' in df_comments.columns:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Comments", len(df_comments))
                            with col2:
                                spam_count = (df_comments["pred_label"] == 1).sum()
                                st.metric("Spam Detected", spam_count)
                            with col3:
                                non_spam_count = (df_comments["pred_label"] == 0).sum()
                                st.metric("Non-Spam", non_spam_count)
                            with col4:
                                avg_conf = df_comments["pred_confidence"].mean()
                                st.metric("Avg Confidence", f"{avg_conf:.2%}")
                        
                        # Display results
                        st.subheader("Comments Data")
                        display_cols = ['comment_text', 'author', 'likes']
                        if 'label' in df_comments.columns:
                            display_cols.extend(['label', 'pred_confidence'])
                        
                        st.dataframe(
                            df_comments[display_cols].head(20),
                            use_container_width=True,
                            height=400
                        )
                        
                        # Download button
                        csv_download = df_comments.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Comments with Predictions (CSV)",
                            data=csv_download,
                            file_name="youtube_comments_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Visualization
                        if 'pred_label' in df_comments.columns:
                            st.subheader("📊 Visualization")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                labels_count = df_comments["label"].value_counts()
                                ax.pie(labels_count.values, labels=labels_count.index, 
                                      autopct='%1.1f%%', colors=['#d62728', '#2ca02c'],
                                      startangle=90)
                                ax.set_title("Spam vs Non-Spam Distribution")
                                st.pyplot(fig, use_container_width=True)
                            
                            with col2:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.hist(df_comments[df_comments["pred_label"] == 0]["pred_confidence"], 
                                       bins=20, alpha=0.6, label="Non-Spam", color='green')
                                ax.hist(df_comments[df_comments["pred_label"] == 1]["pred_confidence"], 
                                       bins=20, alpha=0.6, label="Spam", color='red')
                                ax.set_xlabel("Confidence Score")
                                ax.set_ylabel("Count")
                                ax.set_title("Confidence Distribution by Label")
                                ax.legend()
                                st.pyplot(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ No comments found or API error")
    
    # ========== MODE 2: Batch Upload ==========
    elif mode == "📊 Batch Upload":
        st.header("Batch Prediction from CSV")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="CSV harus memiliki kolom: comment_text, text, content, atau comment"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.info(f"✓ Loaded {len(df)} rows")
            
            text_cols = [c for c in df.columns 
                        if c.lower() in ("comment_text", "text", "content", "comment")]
            
            if not text_cols:
                st.error("❌ CSV harus memiliki kolom: comment_text, text, content, atau comment")
                st.dataframe(df.head())
                st.stop()
            
            text_col = text_cols[0]
            st.info(f"✓ Using column: **{text_col}**")
            
            if st.button("🔍 Predict All", use_container_width=True):
                with st.spinner(f"Predicting {len(df)} rows..."):
                    texts = df[text_col].astype(str).tolist()
                    results = predict_texts(texts, model, vectorizer)
                
                df["pred_label"] = [r["prediction"] for r in results]
                df["pred_confidence"] = [r["confidence"] for r in results]
                df["spam_score"] = [r["spam_score"] for r in results]
                df["label"] = [r["label"] for r in results]
                
                st.success("✓ Prediction complete!")
                
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
                
                st.dataframe(
                    df[[text_col, "label", "pred_confidence", "spam_score"]].head(20),
                    use_container_width=True
                )
                
                csv_download = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv_download,
                    file_name="spam_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
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
        
        baseline_file = "baseline_comparison.csv"
        if os.path.exists(baseline_file):
            st.subheader("📈 Model Comparison")
            baseline_df = pd.read_csv(baseline_file)
            st.dataframe(baseline_df, use_container_width=True)
            
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

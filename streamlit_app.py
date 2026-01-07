# streamlit_app.py
# Streamlit app untuk mendeteksi komentar spam (YouTube) menggunakan model GNN.
# Tempatkan file ini di /content/drive/MyDrive/youtube_spam_detector/
# Requirements (install sekali): streamlit, google-api-python-client, Sastrawi, joblib, networkx, pyvis, torch, torch_geometric, scikit-learn, pandas
#
# Cara jalankan (di Colab setelah mount Drive) :
# !streamlit run /content/drive/MyDrive/youtube_spam_detector/streamlit_app.py

import os
import re
import io
import time
import pickle
import math
import tempfile
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# network visualization
from pyvis.network import Network
import streamlit.components.v1 as components

# ML utils
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# optional: YouTube API
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# preprocessing
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# torch & pyg
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ----------------------------
# Config / paths
# ----------------------------
PROJECT_PATH_DEFAULT = "/content/drive/MyDrive/youtube_spam_detector"
TFIDF_FILENAME = "tfidf_vectorizer.pkl"
GRAPH_PICKLE = "graph_structure.pkl"   # optional: not required for single-video run
MODEL_FILENAME = "gnn_spam_model.pt"

# ----------------------------
# SimpleGNN (must match saved model)
# ----------------------------
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

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_vectorizer(path):
    return joblib.load(path)

@st.cache_resource
def load_model(path, in_dim):
    model = SimpleGNN(in_dim)
    state = torch.load(path, map_location="cpu")
    # handle wrappers
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model

def minimal_clean(text: str, stop_remover=None) -> str:
    t = re.sub(r"http\S+", "", str(text))
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = t.lower().strip()
    if stop_remover is not None:
        try:
            t = stop_remover.remove(t)
        except Exception:
            pass
    return t

def fetch_comments_from_youtube(video_id: str, api_key: str, max_results: int = 500) -> pd.DataFrame:
    """
    Mengambil komentar publik dari YouTube Data API v3 (comments + replies).
    Kembalikan DataFrame dengan kolom: comment_id, author, text, published_at, parent_id (None jika top-level).
    """
    comments = []
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.commentThreads().list(
        part="snippet,replies",
        maxResults=100,
        videoId=video_id,
        textFormat="plainText"
    )
    while request and len(comments) < max_results:
        try:
            resp = request.execute()
        except HttpError as e:
            st.error(f"Error fetching comments from YouTube API: {e}")
            break
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            cid = item["snippet"]["topLevelComment"]["id"]
            comments.append({
                "comment_id": cid,
                "author": top.get("authorDisplayName"),
                "text": top.get("textDisplay"),
                "published_at": top.get("publishedAt"),
                "parent_id": None
            })
            # replies
            replies = item.get("replies", {}).get("comments", [])
            for r in replies:
                r_snip = r["snippet"]
                comments.append({
                    "comment_id": r["id"],
                    "author": r_snip.get("authorDisplayName"),
                    "text": r_snip.get("textDisplay"),
                    "published_at": r_snip.get("publishedAt"),
                    "parent_id": cid
                })
        request = youtube.commentThreads().list_next(request, resp)
        # safeguard
        if len(comments) >= max_results:
            break
    df = pd.DataFrame(comments)
    return df

def build_similarity_graph(X_tfidf: np.ndarray, top_k: int = 5, threshold: float = 0.25) -> Tuple[np.ndarray, list]:
    """
    Build edge list by nearest-neighbors on TF-IDF vectors.
    Returns edge_index (2, E) as torch.LongTensor and list of (u,v) edges.
    """
    sim = cosine_similarity(X_tfidf)
    n = sim.shape[0]
    edges = set()
    for i in range(n):
        # top_k neighbors excluding self
        idx = np.argsort(sim[i])[::-1]
        cnt = 0
        for j in idx:
            if j == i: 
                continue
            if sim[i, j] < threshold:
                break
            edges.add((i, j))
            cnt += 1
            if cnt >= top_k:
                break
    edges_list = list(edges)
    if len(edges_list) == 0:
        # fallback: fully connect small graphs
        for i in range(n):
            for j in range(i+1, n):
                edges_list.append((i, j)); edges_list.append((j, i))
    edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    return edge_index, edges_list

def run_gnn_inference(X_tfidf: np.ndarray, model, edge_index):
    x_tensor = torch.tensor(X_tfidf, dtype=torch.float)
    data = Data(x=x_tensor, edge_index=edge_index)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.exp(out)[:, 1].cpu().numpy()  # spam prob
        preds = out.argmax(dim=1).cpu().numpy()
    return preds, probs

def visualize_pyvis(nodes_df: pd.DataFrame, edges: List[Tuple[int,int]], score_col: str = "spam_score"):
    net = Network(height="600px", width="100%", notebook=False)
    net.toggle_physics(True)
    # add nodes with size by spam score
    for idx, row in nodes_df.iterrows():
        label = str(idx)
        title = f"{row['clean_text'][:200]}<br>score={row.get(score_col, 0):.3f}"
        size = 8 + (row.get(score_col, 0) * 25)
        color = "#ff6666" if row.get(score_col,0) >= 0.5 else "#66b3ff"
        net.add_node(int(idx), label=label, title=title, size=size, color=color)
    for u,v in edges:
        net.add_edge(int(u), int(v))
    # save to temporary html and return html
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.show(tmpfile.name)
    html = open(tmpfile.name, "r", encoding="utf-8").read()
    return html

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(layout="wide", page_title="GNN YouTube Spam Detector")

st.title("GNN — YouTube Spam Detector (Streamlit)")
st.markdown("Masukkan link YouTube atau unggah CSV komentar. Sistem akan memproses dan menampilkan tabel, diagram batang, dan graf interaktif.")

# Sidebar: paths & options
st.sidebar.header("Settings & Paths")
project_path = st.sidebar.text_input("Project path (Drive)", PROJECT_PATH_DEFAULT)
tfidf_path = os.path.join(project_path, TFIDF_FILENAME)
model_path = os.path.join(project_path, MODEL_FILENAME)

st.sidebar.markdown("**Model & Vectorizer**")
st.sidebar.write(tfidf_path)
st.sidebar.write(model_path)

# Load resources (with caching)
if not os.path.exists(tfidf_path) or not os.path.exists(model_path):
    st.error("TF-IDF vectorizer or model file tidak ditemukan di project path. Pastikan file berada di: " + str(project_path))
    st.stop()

try:
    vectorizer = load_vectorizer(tfidf_path)
except Exception as e:
    st.error(f"Gagal memuat TF-IDF vectorizer: {e}")
    st.stop()

# load model (needs in_dim)
dummy_in_dim = vectorizer.transform(["test"]).shape[1]
try:
    model = load_model(model_path, dummy_in_dim)
except Exception as e:
    st.error(f"Gagal memuat model GNN: {e}")
    st.stop()

# Sastrawi stopword
stop_factory = StopWordRemoverFactory()
stop_remover = stop_factory.create_stop_word_remover()

# Input area
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Ambil komentar dari YouTube")
    youtube_url = st.text_input("YouTube video URL (atau ID video)", "")
    api_key = st.text_input("YouTube API key (opsional, jika kosong gunakan upload CSV)", type="password")
    max_comments = st.number_input("Max comments to fetch (YouTube API)", min_value=50, max_value=2000, value=500, step=50)
    threshold = st.slider("Similarity threshold (graph edge)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    top_k = st.slider("Top-K neighbors per node", min_value=1, max_value=20, value=5, step=1)
    run_btn = st.button("Proses komentar")

with col2:
    st.subheader("Atau unggah CSV komentar")
    uploaded_file = st.file_uploader("Upload CSV berisi kolom 'comment_text' (opsional)", type=["csv"])
    st.markdown("Jika API key tidak tersedia, silakan unggah CSV komentar sebagai alternatif.")

# Action: get comments either by API or uploaded file
df_comments = None
if run_btn:
    if youtube_url.strip() != "" and api_key.strip() != "":
        # parse video id from URL or accept direct id
        vid = youtube_url.strip()
        if "watch?v=" in vid or "youtu.be" in vid:
            # extract id
            m = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", vid)
            if m:
                vid = m.group(1)
        st.info(f"Fetching comments for video id: {vid} (max {max_comments}) ...")
        try:
            df_comments = fetch_comments_from_youtube(vid, api_key, max_results=int(max_comments))
            st.success(f"Fetched {len(df_comments)} comments")
        except Exception as e:
            st.error(f"Error fetching comments: {e}")
    elif uploaded_file is not None:
        try:
            df_comments = pd.read_csv(uploaded_file)
            st.success(f"Loaded uploaded CSV with {len(df_comments)} rows")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    else:
        st.warning("Masukkan API key & video URL, atau unggah file CSV sebelum menekan Proses.")

# If we have comments, preprocess & infer
if df_comments is not None and len(df_comments) > 0:
    # ensure column names
    if "comment_text" not in df_comments.columns:
        # try common alternatives
        if "text" in df_comments.columns:
            df_comments = df_comments.rename(columns={"text":"comment_text"})
        elif "comment" in df_comments.columns:
            df_comments = df_comments.rename(columns={"comment":"comment_text"})
        else:
            st.error("Kolom komentar tidak ditemukan (harus bernama 'comment_text' atau 'text').")
            st.stop()

    # cleaning
    st.info("Preprocessing text (cleaning & stopword removal)...")
    df_comments["clean_text"] = df_comments["comment_text"].astype(str).apply(lambda t: minimal_clean(t, stop_remover))
    # vectorize using pre-trained vectorizer: transform (note: if new words unseen, vectorizer handles it)
    X_tfidf = vectorizer.transform(df_comments["clean_text"].tolist()).toarray()
    st.write("TF-IDF shape:", X_tfidf.shape)

    # build similarity graph
    st.info("Membangun graf (similarity-based)...")
    edge_index, edges_list = build_similarity_graph(X_tfidf, top_k=int(top_k), threshold=float(threshold))
    st.write("Edges:", len(edges_list))

    # run model inference
    st.info("Menjalankan inference GNN...")
    try:
        preds, probs = run_gnn_inference(X_tfidf, model, edge_index)
    except Exception as e:
        st.error(f"Error saat inference model: {e}")
        st.stop()

    df_comments["pred_label"] = preds[:len(df_comments)]
    df_comments["spam_score"] = probs[:len(df_comments)]

    # display top table
    st.subheader("Hasil Deteksi — Tabel Komentar")
    st.write("Urut berdasarkan spam_score tertinggi")
    st.dataframe(df_comments[["comment_text", "clean_text", "pred_label", "spam_score"]].sort_values("spam_score", ascending=False).reset_index(drop=True), height=300)

    # summary bar chart
    st.subheader("Ringkasan Jumlah Spam vs Non-Spam")
    counts = df_comments["pred_label"].value_counts().rename({0:"Non-Spam",1:"Spam"})
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    ax.set_ylabel("Jumlah komentar")
    ax.set_xlabel("Label (0=Non-Spam,1=Spam)")
    st.pyplot(fig)

    # plot distribution of spam scores
    st.subheader("Distribusi Spam Score")
    fig2, ax2 = plt.subplots()
    sns.histplot(df_comments["spam_score"], bins=30, kde=True, ax=ax2)
    ax2.set_xlabel("Spam score (0..1)")
    st.pyplot(fig2)

    # network visualization (PyVis)
    st.subheader("Visualisasi Graf (interaktif)")
    with st.spinner("Membuat visualisasi graf..."):
        html = visualize_pyvis(df_comments.reset_index(drop=True), edges_list, score_col="spam_score")
        components.html(html, height=650)

    # Save outputs to project folder
    out_csv = os.path.join(project_path, f"predictions_streamlit_{int(time.time())}.csv")
    df_comments.to_csv(out_csv, index=False)
    st.success(f"Hasil prediksi disimpan ke: {out_csv}")

    st.markdown("**Catatan**: jika dataset besar atau model/pustaka heavy (torch_geometric) belum terpasang, jalankan SEL A untuk meng-install dependencies di Colab/VM sebelum menjalankan Streamlit.")

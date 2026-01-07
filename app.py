# app.py
# Streamlit app — ambil komentar YouTube (dengan API key di UI atau dari environment),
# caching & rate-limit, preprocess konsisten dengan pipeline, inference GNN (opsional) atau fallback LR,
# tampil tabel hasil, bar chart, dan visualisasi graf.
#
# Requirements (pip): streamlit google-api-python-client joblib scikit-learn pandas networkx matplotlib seaborn Sastrawi
# Optional (GNN): torch torchvision torch_geometric

import os, time, json, re, io
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Try PyG (optional)
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    PYG_OK = True
except Exception:
    PYG_OK = False

# ----------------------
# Configuration
# ----------------------
PROJECT_PATH = os.environ.get("PROJECT_PATH", "/content/drive/MyDrive/youtube_spam_detector")
Path(PROJECT_PATH).mkdir(parents=True, exist_ok=True)
CACHE_DIR = os.path.join(PROJECT_PATH, "cache_comments")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
FETCH_LOG_FP = os.path.join(PROJECT_PATH, "fetch_log.json")
TFIDF_FP = os.path.join(PROJECT_PATH, "tfidf_vectorizer.pkl")
GNN_FP = os.path.join(PROJECT_PATH, "gnn_spam_model.pt")
ORIG_DATA_FP = os.path.join(PROJECT_PATH, "dataset_youtube_5000_realistic.csv")

# Cache / rate-lim settings
CACHE_TTL_SEC = int(os.environ.get("CACHE_TTL_SEC", 6*3600))  # 6 hours
MIN_FETCH_INTERVAL_SEC = int(os.environ.get("MIN_FETCH_INTERVAL_SEC", 60))  # per video minimal interval
MAX_FETCHS_PER_DAY = int(os.environ.get("MAX_FETCHS_PER_DAY", 500))

# Preprocessing setup
stop_remover = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

# GNN model class (if PyG)
if PYG_OK:
    class SimpleGNN(torch.nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, 128)
            self.conv2 = GCNConv(128, 2)
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.4, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

# ----------------------
# Utility functions
# ----------------------
def now_ts():
    return int(time.time())

def load_fetch_log():
    if os.path.exists(FETCH_LOG_FP):
        try:
            with open(FETCH_LOG_FP, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_fetch_log(log):
    with open(FETCH_LOG_FP, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

def is_rate_limited(video_id):
    log = load_fetch_log()
    rec = log.get(video_id, {"last_fetch": 0, "daily_count": {}})
    last = rec.get("last_fetch", 0)
    if now_ts() - last < MIN_FETCH_INTERVAL_SEC:
        return True, f"Please wait {MIN_FETCH_INTERVAL_SEC - (now_ts()-last)}s before refetching this video."
    # daily count
    today = time.strftime("%Y-%m-%d")
    daily = rec.get("daily_count", {}).get(today, 0)
    if daily >= MAX_FETCHS_PER_DAY:
        return True, f"Daily fetch limit for this video reached ({MAX_FETCHS_PER_DAY})."
    return False, None

def record_fetch(video_id):
    log = load_fetch_log()
    rec = log.get(video_id, {"last_fetch": 0, "daily_count": {}})
    rec["last_fetch"] = now_ts()
    today = time.strftime("%Y-%m-%d")
    dc = rec.get("daily_count", {})
    dc[today] = dc.get(today, 0) + 1
    rec["daily_count"] = dc
    log[video_id] = rec
    save_fetch_log(log)

def cache_path_for(video_id):
    return os.path.join(CACHE_DIR, f"{video_id}.json")

def is_cache_valid(video_id):
    p = cache_path_for(video_id)
    if not os.path.exists(p):
        return False
    mtime = int(os.path.getmtime(p))
    return (now_ts() - mtime) <= CACHE_TTL_SEC

def load_cache(video_id):
    p = cache_path_for(video_id)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cache(video_id, data):
    p = cache_path_for(video_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def parse_video_id(url: str):
    if not url:
        return None
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return None

def fetch_comments_from_api(video_id, api_key, max_pages=5):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    req = youtube.commentThreads().list(part="snippet,replies",
                                       videoId=video_id,
                                       maxResults=100,
                                       textFormat="plainText")
    pages = 0
    while req and pages < max_pages:
        resp = req.execute()
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            cid = item["id"]
            comments.append({
                "comment_id": cid,
                "parent_id": None,
                "comment_text": top.get("textDisplay",""),
                "author": top.get("authorDisplayName",""),
                "published": top.get("publishedAt","")
            })
            if "replies" in item:
                for r in item["replies"].get("comments", []):
                    rs = r["snippet"]
                    comments.append({
                        "comment_id": r["id"],
                        "parent_id": cid,
                        "comment_text": rs.get("textDisplay",""),
                        "author": rs.get("authorDisplayName",""),
                        "published": rs.get("publishedAt","")
                    })
        req = youtube.commentThreads().list_next(req, resp)
        pages += 1
    return comments

# preprocessing (same pipeline)
def clean_text(t):
    t = re.sub(r"http\S+", "", str(t))
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.lower().strip()

def normalize_text(t):
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    t = re.sub(r'\s+', ' ', t)
    slang_map = {"yg":"yang","dg":"dengan","gk":"gak","gak":"gak","ga":"gak","tdk":"tidak","klo":"kalau"}
    tokens = t.split()
    tokens = [slang_map.get(tok, tok) for tok in tokens]
    return " ".join(tokens)

def stem_text(t):
    try:
        return stemmer.stem(t)
    except:
        return t

def preprocess_series(series):
    s = series.astype(str).apply(clean_text)
    s = s.apply(lambda x: stop_remover.remove(x))
    s = s.apply(normalize_text)
    s = s.apply(stem_text)
    return s

# ---------- Streamlit UI ----------
st.set_page_config(page_title="YouTube Spam Detector", layout="wide")
st.title("YouTube Spam Detector — ambil komentar (dengan API key)")

# Sidebar: API key visible + mode selection
st.sidebar.header("Input")
st.sidebar.markdown("Masukkan YouTube Data API key (opsional). Jika kosong, app akan mencoba membaca YOUTUBE_API_KEY dari environment.")
api_key_input = st.sidebar.text_input("YouTube Data API key", type="password")
# if user leaves empty, try environment
API_KEY_ENV = os.environ.get("YOUTUBE_API_KEY")
API_KEY = api_key_input.strip() if api_key_input and api_key_input.strip() else API_KEY_ENV

mode = st.sidebar.selectbox("Mode input", ["Fetch by URL (use API key above)", "Upload CSV", "Paste comments"])
st.sidebar.markdown("Jika Anda deploy publik dan ingin server fetch otomatis, set `YOUTUBE_API_KEY` sebagai env var di server.")

# show model file status
st.sidebar.markdown("---")
st.sidebar.write("Model & assets (project folder):")
st.sidebar.write(f"- tfidf vectorizer: {TFIDF_FP}")
st.sidebar.write(f"- gnn model: {GNN_FP} (optional)")
st.sidebar.write(f"- original labeled dataset: {ORIG_DATA_FP}")

df_input = None

if mode == "Fetch by URL (use API key above)":
    video_url = st.sidebar.text_input("YouTube video URL (paste)")
    if st.sidebar.button("Fetch comments and analyze"):
        if not API_KEY:
            st.error("Tidak ada API key. Masukkan API key di sidebar atau set YOUTUBE_API_KEY di environment.")
        else:
            vid = parse_video_id(video_url)
            if not vid:
                st.error("Tidak dapat mem-parse video id. Periksa URL.")
            else:
                # check cache & rate-limit
                if is_cache_valid(vid):
                    st.info("Menggunakan cache untuk komentar.")
                    cached = load_cache(vid)
                    df_input = pd.DataFrame(cached)
                else:
                    rl, msg = is_rate_limited(vid)
                    if rl:
                        st.error("Rate limited: " + msg)
                    else:
                        st.info("Fetching from YouTube API (this may take a moment)...")
                        try:
                            comments = fetch_comments_from_api(vid, API_KEY, max_pages=5)
                            if not comments:
                                st.warning("No comments fetched.")
                                df_input = pd.DataFrame(comments)
                            else:
                                save_cache(vid, comments)
                                record_fetch(vid)
                                df_input = pd.DataFrame(comments)
                                st.success(f"Fetched {len(df_input)} comments and saved to cache.")
                        except Exception as e:
                            st.exception(e)

elif mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV (columns: comment_text ...)", type=["csv"])
    if uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Gagal membaca CSV: " + str(e))
elif mode == "Paste comments":
    pasted = st.sidebar.text_area("Paste comments (one comment per line)")
    if st.sidebar.button("Process pasted comments"):
        lines = [ln.strip() for ln in pasted.splitlines() if ln.strip()]
        df_input = pd.DataFrame({"comment_text": lines})

# If input present: run pipeline & predictions
if df_input is not None:
    st.header("Preview input (first 200 rows)")
    st.dataframe(df_input.head(200))

    # ensure comment_text column
    if "comment_text" not in df_input.columns:
        if "text" in df_input.columns:
            df_input = df_input.rename(columns={"text":"comment_text"})
        else:
            st.error("No 'comment_text' column. For upload/paste ensure this column exists or mode produces it.")
            st.stop()

    # Preprocessing (show sample)
    st.subheader("Pra-pemrosesan — contoh tiap langkah")
    df_input["clean_text"] = df_input["comment_text"].astype(str).apply(clean_text)
    df_input["no_stop"] = df_input["clean_text"].apply(lambda x: stop_remover.remove(x))
    df_input["normalized"] = df_input["no_stop"].apply(normalize_text)
    df_input["stemmed"] = df_input["normalized"].apply(stem_text)
    st.dataframe(df_input[["comment_text","stemmed"]].head(10))

    # Load vectorizer
    if not os.path.exists(TFIDF_FP):
        st.error(f"tfidf vectorizer not found: {TFIDF_FP}. Place tfidf_vectorizer.pkl in project folder.")
        st.stop()
    vect = joblib.load(TFIDF_FP)
    X = vect.transform(df_input["stemmed"].astype(str).tolist()).toarray()

    # Try GNN inference if available
    preds = None; probs = None
    if PYG_OK and os.path.exists(GNN_FP):
        st.info("Running GNN inference (torch_geometric available).")
        try:
            device = torch.device("cpu")
            model = SimpleGNN(X.shape[1]).to(device)
            state = torch.load(GNN_FP, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.",""): v for k,v in state.items()}
            model.load_state_dict(state)
            model.eval()
            # build graph: use parent_id info if present; else similarity fallback
            G = nx.DiGraph()
            n = len(df_input)
            G.add_nodes_from(range(n))
            if "comment_id" in df_input.columns and "parent_id" in df_input.columns:
                id2idx = {str(r): i for i,r in enumerate(df_input["comment_id"].astype(str).tolist())}
                for i, row in df_input.reset_index().iterrows():
                    pid = row.get("parent_id", None)
                    if pid and str(pid) in id2idx:
                        G.add_edge(id2idx[str(pid)], i)
            if G.number_of_edges() == 0:
                sim = cosine_similarity(X)
                for i in range(len(sim)):
                    neighbors = np.argsort(sim[i])[::-1][1:6]
                    for j in neighbors:
                        if sim[i,j] > 0.15:
                            G.add_edge(i,int(j))
            edgelist = list(G.edges())
            import torch as _torch
            if len(edgelist)>0:
                edge_index = _torch.tensor(edgelist, dtype=_torch.long).t().contiguous()
            else:
                edge_index = _torch.empty((2,0), dtype=_torch.long)
            x_tensor = _torch.tensor(X, dtype=_torch.float)
            with _torch.no_grad():
                out = model(x_tensor, edge_index)
                preds = out.argmax(dim=1).cpu().numpy()
                probs = _torch.exp(out)[:,1].cpu().numpy()
            st.success("GNN inference done.")
        except Exception as e:
            st.warning("GNN inference failed, will fallback to LogisticRegression. Error: " + str(e))
            preds = None

    # Fallback LR if GNN not used
    if preds is None:
        st.info("Fallback: training LogisticRegression quickly using original labeled dataset (Drive).")
        if not os.path.exists(ORIG_DATA_FP):
            st.error(f"Original labeled dataset not found: {ORIG_DATA_FP}. Cannot fallback to LR.")
            st.stop()
        df_orig = pd.read_csv(ORIG_DATA_FP)
        if "stemmed" not in df_orig.columns:
            df_orig["clean_text"] = df_orig["comment_text"].astype(str).apply(clean_text)
            df_orig["no_stop"] = df_orig["clean_text"].apply(lambda x: stop_remover.remove(x))
            df_orig["normalized"] = df_orig["no_stop"].apply(normalize_text)
            df_orig["stemmed"] = df_orig["normalized"].apply(stem_text)
        X_train = vect.transform(df_orig["stemmed"].astype(str).tolist()).toarray()
        y_train = df_orig["label"].astype(int).values
        lr = LogisticRegression(max_iter=400).fit(X_train, y_train)
        preds = lr.predict(X)
        try:
            probs = lr.predict_proba(X)[:,1]
        except:
            probs = np.zeros(len(preds))

    df_input["pred_label"] = preds.astype(int)
    df_input["spam_score"] = probs.astype(float)

    # Display results
    st.subheader("Hasil Prediksi - contoh")
    st.dataframe(df_input[["comment_text","pred_label","spam_score"]].head(200))

    st.subheader("Jumlah Spam vs Non-Spam")
    counts = df_input["pred_label"].value_counts().rename(index={0:"Non-Spam",1:"Spam"})
    st.bar_chart(counts)

    # graph visualization
    st.subheader("Graph (reply graph or similarity fallback)")
    try:
        # rebuild small graph for visualization
        Gvis = nx.DiGraph()
        Gvis.add_nodes_from(range(len(df_input)))
        if "comment_id" in df_input.columns and "parent_id" in df_input.columns:
            id2idx = {str(r): i for i,r in enumerate(df_input["comment_id"].astype(str).tolist())}
            for i, row in df_input.reset_index().iterrows():
                pid = row.get("parent_id", None)
                if pid and str(pid) in id2idx:
                    Gvis.add_edge(id2idx[str(pid)], i)
        if Gvis.number_of_edges() == 0:
            sim = cosine_similarity(joblib.load(TFIDF_FP).transform(df_input["stemmed"].astype(str)).toarray())
            for i in range(len(sim)):
                neighbors = np.argsort(sim[i])[::-1][1:6]
                for j in neighbors:
                    if sim[i,j] > 0.15:
                        Gvis.add_edge(i,int(j))
        sample_nodes = list(range(min(300, Gvis.number_of_nodes())))
        SG = Gvis.subgraph(sample_nodes).copy()
        plt.figure(figsize=(10,6))
        pos = nx.spring_layout(SG, seed=42)
        nx.draw_networkx_nodes(SG, pos, node_size=30)
        nx.draw_networkx_edges(SG, pos, alpha=0.4)
        plt.axis("off")
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.warning("Graph visualization failed: " + str(e))

    # Download button
    csv = df_input.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", csv, "predictions_youtube.csv", "text/csv")

else:
    st.info("Pilih mode input (sidebar). Untuk publik: Upload CSV atau Paste comments juga tersedia.")

# End of app.py

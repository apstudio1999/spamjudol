# YouTube Spam Detector — Local Version

Script deteksi spam komentar YouTube menggunakan Graph Neural Network (GNN), untuk dijalankan **lokal di komputer Anda** tanpa perlu Google Colab.

## 📋 Struktur Proyek

```
spam_detector/
├── setup_local.py               # Install dependencies (run ONCE)
├── train_local.py               # Training GNN model
├── inference_local.py           # Predict pada komentar baru
├── streamlit_app.py             # Web UI untuk prediksi & analisis
│
├── gnn_spam_model.pt            # ✓ Model GNN (sudah ada dari Colab)
├── tfidf_vectorizer.pkl         # ✓ TF-IDF vectorizer (sudah ada)
├── dataset_stemmed_with_label.csv # Dataset untuk referensi
├── graph_structure_generated.pkl # Graph struktur
├── edge_index.npy               # Edge indices
│
├── requirements.txt             # Dependencies list
└── README.md                    # File ini

```

## 🚀 Setup Awal (Windows)

### 1. **Install Dependencies**

Buka Command Prompt atau PowerShell di folder proyek, lalu jalankan:

```powershell
python setup_local.py
```

Script ini akan:
- ✓ Upgrade pip, setuptools, wheel
- ✓ Install semua library dari `requirements.txt`
- ✓ Verify PyTorch dan torch-geometric
- ✓ Check file-file penting ada

**Jika ada error:** Coba instalasi manual:
```powershell
pip install -r requirements.txt
```

### 2. **Verifikasi Model & Files**

Pastikan file-file berikut sudah ada di folder proyek:
- `gnn_spam_model.pt` — Model GNN yang sudah dilatih ✓
- `tfidf_vectorizer.pkl` — Feature extractor ✓
- Dataset CSV (bisa gunakan `dataset_youtube_5000_realistic.csv`)

Jika belum ada, download dari Colab atau copy dari Google Drive.

---

## 📝 Cara Menggunakan

### **Option A: Prediksi Single Text (Terminal)**

Prediksi satu komentar langsung dari terminal:

```powershell
python inference_local.py --text "subscribe channel kami gratis"
```

Output:
```
╔════════════════════════════════════════════════════════════╗
║  PREDICTION RESULT                                         ║
╚════════════════════════════════════════════════════════════╝

Text:       subscribe channel kami gratis
Label:      Spam
Confidence: 0.8234
Spam Score: 0.8234
```

### **Option B: Batch Prediction dari CSV**

Prediksi multiple komentar dari file CSV:

```powershell
python inference_local.py labels_to_fill.csv --output my_predictions.csv
```

Input CSV harus memiliki kolom: `comment_text`, `text`, `content`, atau `comment`

Output file akan include kolom baru:
- `pred_label` — 0 (Non-Spam) atau 1 (Spam)
- `pred_confidence` — Confidence score (0.0 - 1.0)
- `spam_score` — Probability spam (0.0 - 1.0)

### **Option C: Web UI (Streamlit) — Recommended! 🎯**

Buka aplikasi web interaktif:

```powershell
streamlit run streamlit_app.py
```

Otomatis membuka browser di `http://localhost:8501`

Fitur:
- ✓ Paste atau upload CSV untuk prediksi
- ✓ Single text prediction
- ✓ Real-time results
- ✓ Visualization & statistics
- ✓ Download predictions

---

## 🔄 Training Model Baru (Optional)

Jika ingin retrain model dengan dataset baru:

### 1. **Prepare Dataset**

CSV harus punya kolom:
- `comment_text` atau `text` — Isi komentar
- `label` — 0 (non-spam) atau 1 (spam)

Contoh:
```csv
comment_text,label
"Subscribe channel kami",1
"Komentar yang bagus",0
```

### 2. **Run Training**

```powershell
# Preprocess dataset saja (no training)
python train_local.py --dataset my_dataset.csv

# Preprocess + training
python train_local.py --dataset my_dataset.csv --epochs 30 --train

# Skip baseline comparison (faster)
python train_local.py --dataset my_dataset.csv --epochs 30 --train --no-baseline
```

**Opsi parameter:**
- `--dataset` — Path ke CSV (default: `dataset_youtube_5000_realistic.csv`)
- `--epochs` — Jumlah training epochs (default: 30)
- `--train` — Flag untuk melakukan training (tanpa flag, hanya preprocessing)
- `--no-baseline` — Skip model comparison dengan LR & SVM

**Output training:**
- `gnn_spam_model.pt` — Model terlatih (akan override yang lama)
- `metrics_gnn.json` — Evaluation metrics
- `confusion_matrix_gnn.png` — Confusion matrix plot
- `predictions_gnn_full.csv` — Predictions pada seluruh dataset
- `baseline_comparison.csv` — Perbandingan dengan baseline models

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'torch_geometric'"

```powershell
# Install torch-geometric CPU version
pip install torch-geometric

# Jika masih error, coba:
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install torch-geometric
```

### Error: "FileNotFoundError: gnn_spam_model.pt"

Pastikan file sudah didownload dari Colab dan ada di folder proyek.

### Slow inference atau out of memory?

Model menggunakan CPU secara default. Jika punya GPU NVIDIA:

Edit file `.py`:
```python
# Ganti:
DEVICE = torch.device("cpu")

# Dengan:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Streamlit not opening browser automatically?

Buka manual di: `http://localhost:8501`

---

## 📊 Model Architecture

**GNN Model:**
- Input: TF-IDF vectors (2000 features)
- Layer 1: GCNConv(2000 → 128)
- Activation: ReLU
- Dropout: 0.4
- Layer 2: GCNConv(128 → 2)
- Output: Log-softmax (binary classification)

**Optimizer:** Adam (lr=0.001, weight_decay=5e-4)

**Loss:** NLLLoss

---

## 📈 Performance

Dari training di Colab:
- **Accuracy:** ~88-92%
- **Precision:** ~85-90%
- **Recall:** ~90-95%
- **F1-Score:** ~87-92%
- **ROC-AUC:** ~0.94

(Metrics tergantung dataset yang digunakan)

---

## 📚 File Descriptions

| File | Purpose |
|------|---------|
| `setup_local.py` | Install dependencies, verify setup |
| `train_local.py` | Training GNN model dari scratch |
| `inference_local.py` | Batch/single prediction |
| `streamlit_app.py` | Web UI untuk end-user |
| `gnn_spam_model.pt` | Model weights (binary) |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `requirements.txt` | Python dependencies list |

---

## 🎓 Text Preprocessing Pipeline

Setiap komentar melalui:
1. **URL removal** — Hapus links
2. **Cleaning** — Hapus special characters
3. **Lowercasing** — Semua jadi huruf kecil
4. **Tokenization** — Split menjadi words
5. **Stopword removal** — Hapus kata umum (Indonesian + English)
6. **Normalization** — Duplicate char removal, slang correction
7. **Stemming** — Reduce ke root word (Sastrawi)
8. **Vectorization** — TF-IDF features

---

## 💡 Tips & Best Practices

1. **Untuk prediksi terbaik:** Gunakan data yang similar dengan training data
2. **Confidence score rendah?** Bisa berarti ambiguous comment
3. **Retrain berkala:** Jika pattern spam berubah, retrain dengan data baru
4. **Hyperparameter tuning:** Edit `Config` class di script untuk adjust learning rate, epochs, dll

---

## 📞 Support

Jika ada error atau pertanyaan:
1. Check error message dalam console
2. Lihat logs di Streamlit UI
3. Verify file paths ada
4. Coba setup ulang: `python setup_local.py`

---

## 📄 License

Tugas Skripsi — Deteksi Spam Komentar YouTube menggunakan GNN

---

**Created with ❤️ for Local Deployment**

Last updated: 2025-12-22

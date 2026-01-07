# 🚀 QUICK START GUIDE - YouTube Spam Detector (Local)

**Ingin langsung jalankan? Ikuti langkah-langkah di bawah!**

## Step 1: Setup (Run Once)

Buka PowerShell/Command Prompt di folder proyek, jalankan:

```powershell
python setup_local.py
```

Tunggu sampai selesai. Akan install semua dependencies yang dibutuhkan.

---

## Step 2: Pilih Mode Penggunaan

### 🎯 **OPTION A: Web UI (Recommended! Paling Mudah)**

Jalankan:
```powershell
streamlit run streamlit_app_local.py
```

Browser otomatis terbuka di `http://localhost:8501`

**Fitur:**
- ✓ Paste text atau upload CSV
- ✓ Lihat hasil real-time
- ✓ Download predictions
- ✓ Lihat statistics & metrics

---

### 🖥️ **OPTION B: Command Line (Terminal)**

Prediksi single text:
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

Prediksi batch dari CSV:
```powershell
python inference_local.py labels_to_fill.csv --output my_predictions.csv
```

---

### 🏋️ **OPTION C: Training Model Baru**

Jika ingin retrain dengan dataset baru:

```powershell
# Hanya preprocess
python train_local.py --dataset my_dataset.csv

# Preprocess + training (30 epochs)
python train_local.py --dataset my_dataset.csv --epochs 30 --train
```

Dataset harus punya kolom: `comment_text` atau `text`, dan `label` (0/1)

---

## File Penting yang Harus Ada

Pastikan folder proyek memiliki:

✓ `gnn_spam_model.pt` — Model GNN (dari Colab)  
✓ `tfidf_vectorizer.pkl` — Vectorizer (dari Colab)  
✓ `dataset_youtube_5000_realistic.csv` — Dataset (opsional)  

Jika belum punya, download dari Google Drive/Colab folder.

---

## Contoh Workflow

### Skenario 1: Cek apakah komentar spam

```powershell
# Terminal
python inference_local.py --text "Hi! Check my youtube channel"
```

### Skenario 2: Batch check 1000 komentar

```powershell
# Upload CSV dengan kolom 'comment_text'
streamlit run streamlit_app_local.py
# → UI akan muncul, upload CSV, klik Predict
# → Download hasil CSV
```

### Skenario 3: Retrain model dengan data baru

```powershell
# CSV baru dengan kolom 'comment_text' dan 'label'
python train_local.py --dataset comments_2024.csv --epochs 20 --train

# Akan generate:
# - gnn_spam_model.pt (new model)
# - metrics_gnn.json
# - confusion_matrix_gnn.png
# - predictions_gnn_full.csv
```

---

## Troubleshooting

**Q: "ModuleNotFoundError: No module named 'torch_geometric'"**

A: Jalankan:
```powershell
pip install torch-geometric
```

**Q: "FileNotFoundError: gnn_spam_model.pt"**

A: Download file dari Colab atau Google Drive, copy ke folder proyek

**Q: Streamlit crash atau error**

A: Coba:
```powershell
# Stop (Ctrl+C)
# Cek error di console
# Jalankan setup ulang
python setup_local.py
```

**Q: Prediksi lamaaaaa sekali**

A: Model sedang berjalan di CPU. Jika punya GPU NVIDIA, edit file:

Buka `inference_local.py` atau `streamlit_app_local.py`, cari:
```python
DEVICE = torch.device("cpu")
```

Ganti dengan:
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## File Documentation

| File | Fungsi |
|------|--------|
| `setup_local.py` | Install dependencies (run once) |
| `inference_local.py` | CLI untuk prediksi |
| `streamlit_app_local.py` | Web UI (recommended) |
| `train_local.py` | Training model baru |
| `README.md` | Dokumentasi lengkap |

---

## Next Steps

1. ✓ Jalankan `setup_local.py`
2. ✓ Coba `streamlit run streamlit_app_local.py`
3. ✓ Upload CSV atau paste text untuk test
4. ✓ Download predictions jika butuh
5. ✓ Integrasikan ke aplikasi Anda!

---

**Need help?** Baca `README.md` untuk dokumentasi lengkap.

**Created:** December 2025  
**Model:** Graph Neural Network (GCN)

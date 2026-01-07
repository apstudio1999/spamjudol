# 📋 PRE-DEPLOYMENT CHECKLIST

Gunakan checklist ini untuk memastikan semua file siap sebelum di-deploy lokal.

## ✓ File Structure Check

Pastikan folder proyek memiliki struktur ini:

```
spam_detector/
├── Python Scripts (BARU)
│   ├── [ ] setup_local.py              ✓ Install dependencies
│   ├── [ ] train_local.py              ✓ Training lokal
│   ├── [ ] inference_local.py          ✓ Prediction CLI
│   └── [ ] streamlit_app_local.py      ✓ Web UI
│
├── Model & Data (DARI COLAB)
│   ├── [ ] gnn_spam_model.pt           ✓ Model weights
│   ├── [ ] tfidf_vectorizer.pkl        ✓ Vectorizer
│   └── [ ] dataset_youtube_5000_realistic.csv  ✓ Dataset
│
├── Documentation (BARU)
│   ├── [ ] README.md                   ✓ Dokumentasi lengkap
│   ├── [ ] QUICKSTART.md               ✓ Quick start guide
│   ├── [ ] INTEGRATION_GUIDE.md        ✓ Integrasi ke apps lain
│   └── [ ] CHECKLIST.md                ✓ File ini
│
├── Configuration
│   └── [ ] requirements.txt             ✓ Dependencies list
│
└── Optional Output Files
    ├── [ ] dataset_stemmed_with_label.csv
    ├── [ ] graph_structure_generated.pkl
    ├── [ ] edge_index.npy
    ├── [ ] metrics_gnn.json
    ├── [ ] confusion_matrix_gnn.png
    └── [ ] baseline_comparison.csv
```

---

## ✓ Dependencies Check

Pastikan Python 3.8+ dan packages berikut:

```
[ ] Python >= 3.8
[ ] pip
[ ] pandas >= 1.3.0
[ ] numpy >= 1.21.0
[ ] scikit-learn >= 0.24.0
[ ] torch >= 2.0.0
[ ] torch-geometric >= 2.2.0
[ ] networkx >= 2.6.0
[ ] Sastrawi >= 1.0.1
[ ] joblib >= 1.0.0
[ ] streamlit >= 1.20.0
[ ] matplotlib >= 3.4.0
[ ] seaborn >= 0.11.0
```

**Verify dengan:**
```powershell
python -c "import torch; import torch_geometric; print('✓ All good')"
```

---

## ✓ Model & Vectorizer Files

Pastikan file-file ini ada dan valid:

```
[ ] gnn_spam_model.pt
    - Size: ~4-5 MB
    - Format: PyTorch binary
    
[ ] tfidf_vectorizer.pkl
    - Size: ~1-2 MB
    - Format: Joblib pickle
```

**Verify:**
```powershell
python -c "import torch; torch.load('gnn_spam_model.pt')"
python -c "import joblib; joblib.load('tfidf_vectorizer.pkl')"
```

---

## ✓ Configuration Check

```
[ ] setup_local.py dikonfigurasi
[ ] paths sudah benar (tidak hardcoded Colab paths)
[ ] requirements.txt lengkap
[ ] CUDA/GPU settings (if using GPU)
```

---

## ✓ Run Tests

### Test 1: Setup
```powershell
[ ] python setup_local.py
    Expected: Setup berhasil, semua packages terinstall
```

### Test 2: Single Prediction
```powershell
[ ] python inference_local.py --text "test spam message"
    Expected: Result dengan label, confidence, spam_score
```

### Test 3: Batch Prediction
```powershell
[ ] python inference_local.py labels_to_fill.csv
    Expected: CSV dengan predictions dihasilkan
```

### Test 4: Streamlit App
```powershell
[ ] streamlit run streamlit_app_local.py
    Expected: Browser terbuka, dapat paste text dan upload CSV
```

### Test 5: Training (Optional)
```powershell
[ ] python train_local.py --dataset dataset.csv
    Expected: Preprocessing selesai, metrics dihasilkan
```

---

## ✓ Documentation Check

Pastikan dokumentasi lengkap:

```
[ ] README.md
    - [ ] Setup instructions
    - [ ] Usage examples
    - [ ] Troubleshooting
    
[ ] QUICKSTART.md
    - [ ] Step-by-step quick start
    - [ ] Examples
    
[ ] INTEGRATION_GUIDE.md
    - [ ] Integration examples
    - [ ] Best practices
```

---

## ✓ Deployment Readiness

### For Local Machine
```
[ ] All Python scripts tested
[ ] Model & vectorizer files present & verified
[ ] requirements.txt installable
[ ] Documentation complete
```

### For Production Server
```
[ ] Choose deployment method:
    [ ] Streamlit Cloud (simplest)
    [ ] Docker container
    [ ] VM/Server with systemd
    [ ] API server (Flask/FastAPI)
    
[ ] Security considerations:
    [ ] Model file permissions
    [ ] Input validation
    [ ] Rate limiting (if API)
    [ ] Logging & monitoring
```

### For Team Sharing
```
[ ] README is clear for non-technical users
[ ] QUICKSTART.md has copy-paste commands
[ ] Error messages are helpful
[ ] Logging enabled for debugging
```

---

## ✓ Performance Metrics

Before deployment, verify performance:

```
[ ] Single prediction latency: < 1 second
[ ] Batch prediction (100 items): < 10 seconds
[ ] Memory usage: < 2 GB
[ ] GPU usage (if applicable): reasonable
```

**Test:**
```powershell
python -c "
import time
from inference_local import SpamDetector
detector = SpamDetector()

# Single
start = time.time()
result = detector.predict('test')
print(f'Single: {time.time()-start:.3f}s')

# Batch 100
start = time.time()
result = detector.predict(['test'] * 100)
print(f'Batch 100: {time.time()-start:.3f}s')
"
```

---

## ✓ Common Issues Pre-flight

```
[ ] ModuleNotFoundError: Run setup_local.py
[ ] FileNotFoundError: Check model/vectorizer files
[ ] CUDA errors: Check GPU/PyTorch installation
[ ] Streamlit port conflicts: Kill existing process, retry
[ ] Memory issues: Reduce batch size or use CPU
```

---

## ✓ Final Checklist

Before saying "READY FOR PRODUCTION":

```
☐ All files structure correct
☐ All dependencies installed
☐ Single prediction test: PASS
☐ Batch prediction test: PASS
☐ Streamlit UI test: PASS
☐ Documentation complete
☐ Performance acceptable
☐ Error handling implemented
☐ Logging enabled
☐ Security review done (if applicable)
```

---

## 📝 Sign-off

**Prepared by:** [Your Name]  
**Date:** [Date]  
**Status:** ☐ READY / ☐ IN PROGRESS / ☐ BLOCKED

**Notes:**
```
[Add any notes here]
```

---

## 📞 Support Contact

Jika ada issues:
1. Check error logs
2. Run setup_local.py again
3. Verify file paths
4. Check requirements.txt
5. See troubleshooting di README.md

---

**Created:** December 2025  
**Version:** 1.0  
**Last Updated:** [Date]

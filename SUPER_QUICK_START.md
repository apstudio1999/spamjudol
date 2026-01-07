# ⚡ SUPER QUICK START - 3 LANGKAH MUDAH

## Langkah 1: Install Packages (Cukup 1x)

### Cara Termudah: Double-Click `install.bat`
1. Buka File Explorer
2. Navigate ke: `d:\Projek\web\spam_detector`
3. **Double-click file: `install.bat`**
4. Tunggu 5-10 menit sampai selesai

✓ Otomatis install semua dependencies!

---

### Cara Manual: Terminal
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn joblib Sastrawi networkx torch torchvision streamlit -q
```

---

## Langkah 2: Jalankan Web App

### Cara Termudah: Double-Click `run_streamlit.bat`
1. Double-click file: `run_streamlit.bat`
2. Browser otomatis terbuka
3. DONE! 🎉

---

### Cara Manual: Terminal
```powershell
python -m streamlit run streamlit_app_local.py
```

---

## Langkah 3: Gunakan!

Browser sudah terbuka di: `http://localhost:8501`

1. Paste text: **"subscribe channel kami gratis"**
2. Click button: **"Predict"**
3. Lihat result: **"SPAM"** atau **"NON-SPAM"** ✨

---

## Itu Saja!

Sudah bisa gunakan! 

Fitur Web App:
- ✓ Paste text atau upload CSV
- ✓ Lihat prediction real-time
- ✓ Download hasil CSV
- ✓ Lihat charts & statistics

---

## Masalah?

### Streamlit command not recognized
```powershell
# Gunakan cara ini:
python -m streamlit run streamlit_app_local.py
```

### Install sangat lambat?
1. Matikan antivirus sementara
2. Gunakan WiFi 5GHz atau Ethernet
3. Tutup aplikasi berat lain (Chrome, VS Code)
4. Baca: `INSTALL_CEPAT.txt` untuk tips

### Port 8501 already in use
```powershell
python -m streamlit run streamlit_app_local.py --server.port 8502
```

---

## File Shortcut Yang Tersedia

| File | Fungsi |
|------|--------|
| `install.bat` | Install packages (double-click) |
| `run_streamlit.bat` | Jalankan web app (double-click) |
| `predict.bat` | CLI prediction |

---

**That's it! Enjoy! 🚀**

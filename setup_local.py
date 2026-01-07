#!/usr/bin/env python3
"""
SETUP LOKAL - Install dependencies untuk YouTube Spam Detector
Jalankan: python setup_local.py
"""
import os
import sys
import subprocess

def run_command(cmd, description):
    """Run shell command dengan error handling"""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} — OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} — FAILED")
        print(f"Error: {e}")
        return False

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║  YouTube Spam Detector - Setup Lokal (Windows/Local)       ║
║  Instalasi dependencies dan konfigurasi awal               ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Pastikan pip updated
    print("\n[1/4] Update pip, setuptools, wheel...")
    run_command(
        f"{sys.executable} -m pip install --upgrade pip setuptools wheel",
        "Upgrade pip/setuptools/wheel"
    )
    
    # Install requirements
    print("\n[2/4] Install dependencies dari requirements.txt...")
    if os.path.exists("requirements.txt"):
        run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Install requirements.txt"
        )
    else:
        print("❌ requirements.txt tidak ditemukan!")
        return False
    
    # Verifikasi PyTorch dan torch-geometric
    print("\n[3/4] Verifikasi PyTorch dan torch-geometric...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} terinstal")
        print(f"  CUDA tersedia: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch belum terinstal")
        return False
    
    try:
        import torch_geometric
        print(f"✓ torch-geometric terinstal")
    except ImportError:
        print("⚠ torch-geometric mungkin belum terinstal sepenuhnya")
        print("  (Bisa dilanjutkan, akan coba install CPU wheels)")
    
    # Verifikasi file model dan vectorizer
    print("\n[4/4] Verifikasi file-file penting...")
    required_files = {
        "gnn_spam_model.pt": "Model GNN yang sudah terlatih",
        "tfidf_vectorizer.pkl": "TF-IDF Vectorizer untuk ekstraksi fitur",
        "dataset_stemmed_with_label.csv": "Dataset untuk referensi",
    }
    
    missing = []
    for fname, desc in required_files.items():
        if os.path.exists(fname):
            print(f"✓ {fname:.<40} OK")
        else:
            print(f"✗ {fname:.<40} MISSING")
            missing.append(fname)
    
    if missing:
        print(f"\n⚠ File yang hilang: {', '.join(missing)}")
        print("  (Pastikan sudah didownload dari Colab atau upload ke folder ini)")
    
    print("""
╔════════════════════════════════════════════════════════════╗
║  Setup Selesai!                                            ║
╚════════════════════════════════════════════════════════════╝

Langkah berikutnya:
  1. Jalankan inference: python inference_local.py
  2. Jalankan Streamlit UI: streamlit run streamlit_app.py
  3. Dataset baru? Jalankan: python train_local.py

Dokumentasi lengkap: baca README.md
    """)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

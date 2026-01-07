#!/usr/bin/env python3
"""
FAST INSTALL - Instalasi cepat tanpa banyak tunggu
Untuk Windows lokal - menghindari virtual env yang slow

Jalankan: python fast_install.py
"""
import subprocess
import sys

packages_quick = [
    "streamlit>=1.20.0",
    "torch>=2.0.0",
    "torch-geometric>=2.2.0",
]

print("""
╔════════════════════════════════════════════════════════════╗
║  FAST INSTALL - YouTube Spam Detector                      ║
║  Optimized untuk install cepat tanpa Virtual Env           ║
╚════════════════════════════════════════════════════════════╝
""")

print("\n[STEP 1] Upgrade pip...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"])

print("[STEP 2] Install essensial packages...")
essential = [
    "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn",
    "joblib", "Sastrawi", "networkx"
]
subprocess.run([sys.executable, "-m", "pip", "install"] + essential + ["-q"])

print("[STEP 3] Install PyTorch (besar, tunggu ~2-3 min)...")
subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "-q"])

print("[STEP 4] Install Streamlit...")
subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "-q"])

print("[STEP 5] Install torch-geometric (best-effort)...")
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "torch-geometric", "-q"])
    print("  ✓ torch-geometric installed")
except:
    print("  ⚠ torch-geometric install issue (OK, bisa continue)")

print("""
╔════════════════════════════════════════════════════════════╗
║  ✓ INSTALL COMPLETE!                                      ║
╚════════════════════════════════════════════════════════════╝

Next:
  streamlit run streamlit_app_local.py
""")

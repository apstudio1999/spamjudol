# 🎬 YouTube API Integration Guide

Script Anda sudah dimodifikasi untuk **fetch comments langsung dari YouTube video** menggunakan YouTube API!

## ✨ Fitur Baru

✓ **Fetch Comments dari YouTube**
  - Paste YouTube URL
  - Auto-fetch all comments dari video
  - Configurable max comments

✓ **Auto Prediction**
  - Comments otomatis diprediksi setelah fetch
  - Lihat mana yang spam dan mana yang bukan

✓ **Statistics & Visualization**
  - Spam vs non-spam pie chart
  - Confidence distribution
  - Download results as CSV

---

## 🔑 Setup YouTube API Key

### Step 1: Get API Key
1. Go to: https://console.cloud.google.com/
2. Create new project (or select existing)
3. Enable "YouTube Data API v3"
4. Create OAuth 2.0 credentials (API key)
5. Copy API key

### Step 2: Setup di App
**CARA A: Masukkan di Sidebar (Recommended)**
```
1. Jalankan app: python -m streamlit run streamlit_app_youtube.py
2. Di sidebar, masukkan YouTube API Key di text input
3. Paste YouTube URL
4. Click "Fetch Comments"
```

**CARA B: Set Environment Variable (Secure)**
```powershell
# Windows PowerShell - Permanent
[Environment]::SetEnvironmentVariable("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE", "User")

# Atau temporary (this session only)
$env:YOUTUBE_API_KEY = "YOUR_API_KEY_HERE"

# Verify
echo $env:YOUTUBE_API_KEY
```

**CARA C: Edit Script Directly**
```python
# Di streamlit_app_youtube.py, line ~93
YOUTUBE_API_KEY = "AIzaSyA34RxT4RvZmEDHNygacYLTldskgJe_Y3A"  # <-- REPLACE
```

---

## 📝 Usage

### Jalankan App
```powershell
# Option 1: Double-click run_streamlit.bat
# Option 2: Terminal
python -m streamlit run streamlit_app_youtube.py
```

### Use Fitur YouTube Fetch
1. **App Opens** → http://localhost:8501
2. **Select Mode:** 🎬 YouTube Comments Fetch
3. **Enter API Key** (di sidebar) atau sudah tersimpan
4. **Paste YouTube URL:**
   ```
   https://www.youtube.com/watch?v=dQw4w9WgXcQ
   ```
5. **Set Max Comments** (default: 100)
6. **Check:** "Auto Predict Spam"
7. **Click:** 🎬 Fetch Comments

### Wait & See Results
- Comments akan di-fetch
- Otomatis diprediksi
- Lihat statistics
- Download CSV

---

## 📊 Output

CSV file akan memiliki columns:
```
comment_text          - Isi komentar
author               - Nama penulis komentar
likes                - Jumlah likes
reply_count          - Jumlah replies
published_at         - Tanggal publish
pred_label           - 0 (Non-Spam) atau 1 (Spam)
pred_confidence      - Confidence score
spam_score           - Probability spam
label                - "Spam" atau "Non-Spam"
```

---

## ⚠️ Troubleshooting

### "Invalid YouTube URL"
- Paste FULL URL dengan https://
- Format: `https://www.youtube.com/watch?v=VIDEO_ID`
- Atau: `https://youtu.be/VIDEO_ID`

### "YouTube API Error"
- Check API key sudah benar
- Verify API sudah di-enable di console.cloud.google.com
- Check quota (YouTube API punya rate limit)

### "No comments found"
- Video mungkin disable comments
- API key tidak punya access
- Video tidak ada atau deleted

### "CORS Error" / "Forbidden"
- Biasanya karena API key configuration salah
- Re-generate API key di cloud console

---

## 🎯 API Key Dari User

**Anda sudah memberikan API Key:**
```
AIzaSyA34RxT4RvZmEDHNygacYLTldskgJe_Y3A
```

Untuk setup:

**Option 1: Masukkan di Sidebar (RECOMMENDED!)**
```
1. Run app
2. Di sidebar, ada text input "Enter YouTube API Key"
3. Paste: AIzaSyA34RxT4RvZmEDHNygacYLTldskgJe_Y3A
4. Selesai! Bisa langsung fetch comments
```

**Option 2: Environment Variable**
```powershell
$env:YOUTUBE_API_KEY = "AIzaSyA34RxT4RvZmEDHNygacYLTldskgJe_Y3A"
python -m streamlit run streamlit_app_youtube.py
```

---

## 🔐 Security Notes

⚠️ **PENTING:**
- API Key adalah sensitive data
- Jangan commit ke GitHub
- Use environment variables untuk production
- Set API key restrictions di cloud console (IP whitelist, referer, etc)

Best practice:
```python
# Di .env file (gitignore)
YOUTUBE_API_KEY=your_key_here

# Load di app
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")
```

---

## 📈 What Changed

**Removed:**
- ❌ Single text prediction mode
- ❌ Paste text manually

**Added:**
- ✅ YouTube API integration
- ✅ Video URL input
- ✅ Auto-fetch comments
- ✅ Comment metadata (author, likes, replies)

**Kept:**
- ✅ Batch CSV upload
- ✅ Statistics & visualization
- ✅ Download results

---

## 🚀 Quick Start

```powershell
# 1. Run app
python -m streamlit run streamlit_app_youtube.py

# 2. Paste API key di sidebar
AIzaSyA34RxT4RvZmEDHNygacYLTldskgJe_Y3A

# 3. Paste YouTube URL
https://www.youtube.com/watch?v=dQw4w9WgXcQ

# 4. Click "Fetch Comments"

# 5. Download results!
```

---

## 📝 Example YouTube URLs

**Long Format:**
- https://www.youtube.com/watch?v=dQw4w9WgXcQ

**Short Format:**
- https://youtu.be/dQw4w9WgXcQ

**Embed Format:**
- https://www.youtube.com/embed/dQw4w9WgXcQ

Semua format supported! ✓

---

## 💡 Tips

1. **Max 100 comments per fetch** (YouTube API limit per request)
2. **Set max comments ke 100-200** untuk balanced results
3. **Auto-prediction ON** untuk langsung lihat spam detection
4. **Download CSV** untuk analisis lebih lanjut
5. **Use Batch Upload** untuk comments yang sudah ada di CSV

---

## 📞 Support

Jika ada error:
1. Check YouTube URL valid
2. Verify API key sudah di-enable
3. Check quota di Google Cloud Console
4. Try URL berbeda
5. Lihat error message di terminal

---

**File:** `streamlit_app_youtube.py`  
**Status:** Ready to use ✅  
**API:** YouTube Data API v3  
**Language:** Indonesian-friendly 🇮🇩

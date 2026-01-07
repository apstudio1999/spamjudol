🎬 QUICK START - YouTube Spam Detector (API Version)
═════════════════════════════════════════════════════════════════════════════

Langsung bisa fetch comments dari YouTube dan check spam!

═════════════════════════════════════════════════════════════════════════════

⚡ MULAI DALAM 2 LANGKAH
═════════════════════════════════════════════════════════════════════════════

LANGKAH 1: Jalankan App
───────────────────────────────────
Double-click: run_youtube_api.bat

OR Terminal:
  python -m streamlit run streamlit_app_youtube.py

LANGKAH 2: Gunakan Di Browser
───────────────────────────────────
Browser terbuka di: http://localhost:8501

Pilih Mode:
  ✓ "🎬 Fetch YouTube Comments" 
  ✓ "📊 Batch Upload CSV"
  ✓ "📈 Statistics"

═════════════════════════════════════════════════════════════════════════════

🎬 CARA FETCH DARI YOUTUBE
═════════════════════════════════════════════════════════════════════════════

1. Pilih Mode: "🎬 Fetch YouTube Comments"

2. Paste YouTube URL:
   https://www.youtube.com/watch?v=VIDEO_ID
   
   Contoh:
   https://www.youtube.com/watch?v=dQw4w9WgXcQ

3. Masukkan jumlah comments:
   10 - 1000

4. Click button: "🔍 Fetch & Predict"

5. Tunggu 10-30 detik

6. Lihat results:
   • Tabel dengan semua comments
   • Prediction: Spam atau Non-Spam
   • Confidence score
   • Statistics (X% spam, Y% non-spam)

7. Download CSV untuk analisis lebih lanjut

═════════════════════════════════════════════════════════════════════════════

📝 CONTOH PENGGUNAAN
═════════════════════════════════════════════════════════════════════════════

Skenario 1: Cek spam di music video
  URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
  Comments to fetch: 50
  Result: Lihat spam comments, delete yang suspicious

Skenario 2: Monitor gaming video
  URL: https://www.youtube.com/watch?v=VIDEO_ID
  Comments: 100
  Result: Identify spam patterns, report to YouTube

Skenario 3: Batch check your own video
  URL: https://www.youtube.com/watch?v=YOUR_VIDEO_ID
  Comments: 500 (bisa sampai max available)
  Result: Mass moderate spam comments

═════════════════════════════════════════════════════════════════════════════

🔑 API KEY SETUP
═════════════════════════════════════════════════════════════════════════════

Your API Key:
  AIzaSyA34RxT4RvZmEDHNygacYLTldskgJe_Y3A

OPSI 1: Set di PowerShell (Recommended)
────────────────────────────────────────
$env:YOUTUBE_API_KEY = "AIzaSyA34RxT4RvZmEDHNygacYLTldskgJe_Y3A"
python -m streamlit run streamlit_app_youtube.py

OPSI 2: Set di .bat file
────────────────────────────────────────
Edit run_youtube_api.bat, uncomment line:
  set YOUTUBE_API_KEY=AIzaSyA34RxT4RvZmEDHNygacYLTldskgJe_Y3A

OPSI 3: Input langsung di sidebar
────────────────────────────────────────
Di app sidebar ada input field untuk API key
Paste di sana, click "Set API Key"

═════════════════════════════════════════════════════════════════════════════

✨ FITUR LENGKAP
═════════════════════════════════════════════════════════════════════════════

🎬 Fetch YouTube:
  ✓ Input YouTube URL
  ✓ Fetch top-level comments
  ✓ Auto predict spam
  ✓ Show results dengan metadata
  ✓ Download CSV

📊 Batch Upload:
  ✓ Upload CSV file
  ✓ Auto predict semua rows
  ✓ Download predictions
  ✓ Show statistics

📈 Statistics:
  ✓ Model info & performance
  ✓ Accuracy, Precision, Recall metrics
  ✓ Confusion matrix
  ✓ Baseline comparison

═════════════════════════════════════════════════════════════════════════════

⚠️ PENTING!
═════════════════════════════════════════════════════════════════════════════

⚡ Rate Limits:
  • YouTube API free tier: 10,000 units/hari
  • Fetch comments: ~1 unit per comment
  • Max ~10,000 comments per hari

🎬 Comments Available:
  • Only top-level comments (tidak replies)
  • Video harus enable comments
  • Private videos tidak bisa diakses

⏱️ Processing Time:
  • Fetch 50 comments: ~10-15 detik
  • Fetch 100 comments: ~20-30 detik
  • Batch 1000 CSV rows: ~1-2 menit

═════════════════════════════════════════════════════════════════════════════

❓ FAQ
═════════════════════════════════════════════════════════════════════════════

Q: Bagaimana cara mendapat VIDEO_ID?
A: Dari URL: https://youtube.com/watch?v=VIDEO_ID
   VIDEO_ID adalah string setelah v=

Q: Berapa max comments bisa fetch?
A: Tergantung API quota. Tapi usually bisa 50-500 tanpa issue.

Q: Akurat berapa% prediksinya?
A: ~88-92% accuracy pada test data YouTube comments.

Q: Bisa fetch dari channel lain?
A: Hanya dari specific video saja (pakai URL video).

Q: Apakah data aman?
A: Ya! Semua processing lokal, tidak ada upload ke cloud.

═════════════════════════════════════════════════════════════════════════════

🚀 NEXT STEPS
═════════════════════════════════════════════════════════════════════════════

1. Double-click: run_youtube_api.bat
2. Browser terbuka
3. Paste YouTube URL
4. Click Fetch & Predict
5. Review results
6. Download CSV jika perlu

That's it! Enjoy! 🎉

═════════════════════════════════════════════════════════════════════════════

Perlu help? Baca: YOUTUBE_API_USAGE.txt

@echo off
REM Run Streamlit App - YouTube API Version
REM Double-click untuk jalankan dengan YouTube API integration

echo ╔════════════════════════════════════════════════════════════╗
echo ║  YouTube Spam Detector - API Version                       ║
echo ║  Fetch comments dari YouTube langsung!                     ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Browser akan terbuka di: http://localhost:8501
echo.
echo Features:
echo   ✓ Fetch comments dari YouTube video menggunakan API
echo   ✓ Otomatis predict spam pada setiap comment
echo   ✓ Batch upload CSV untuk prediksi
echo   ✓ Statistics & visualization
echo.
echo Tekan Ctrl+C untuk stop app.
echo.

REM Uncomment dan set API key jika diperlukan
REM set YOUTUBE_API_KEY=AIzaSyA34RxT4RvZmEDHNygacYLTldskgJe_Y3A

python -m streamlit run streamlit_app_youtube.py

pause

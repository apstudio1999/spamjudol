@echo off
REM Run Streamlit App - YouTube API Version
REM Double-click atau jalankan: run_streamlit.bat

echo Starting YouTube Spam Detector with API...
echo Browser akan terbuka di: http://localhost:8501
echo.
echo Tekan Ctrl+C untuk stop.
echo.

REM Set YouTube API Key dari environment variable
REM Atau set di app sidebar
python -m streamlit run streamlit_app_youtube.py

pause

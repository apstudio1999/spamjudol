@echo off
REM Quick Install - Shortcut untuk install packages
REM Double-click untuk install

echo Installing dependencies...
echo.

pip install --upgrade pip setuptools wheel -q
pip install pandas numpy scikit-learn matplotlib seaborn joblib Sastrawi networkx -q
pip install torch torchvision -q
pip install streamlit -q

echo.
echo ✓ Install complete!
echo.
echo Sekarang jalankan: run_streamlit.bat
echo.

pause

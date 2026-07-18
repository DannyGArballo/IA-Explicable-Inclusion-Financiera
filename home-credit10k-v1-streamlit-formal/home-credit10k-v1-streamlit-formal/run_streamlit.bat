@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run streamlit_app.py
if errorlevel 1 (
  echo.
  echo No fue posible iniciar Streamlit. Instala primero las dependencias con:
  echo python -m pip install -r requirements.txt
  pause
)
endlocal
